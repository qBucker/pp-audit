"""Two-stage precision-passport commitments.

Stage 1 (before release):
    C_plan = SHA256("pp-audit/C_plan/v1" || canonical(plan))
    plan   = {v, round, profile, schedule}

Stage 2 (after release):
    H_RAW  = SHA256("pp-audit/H_RAW/v1" || canonical(data))
    data   = {v, round, plan_sha256, entries}
    C_data = SHA256("pp-audit/C_data/v1" || bytes.fromhex(H_RAW))

The canonical encoding is deterministic JSON (sorted keys, no whitespace,
UTF-8). The verification recomputes both commitments and audits the delivered
values against the committed profile.
"""

import hashlib
import json

import numpy as np

from . import core

PLAN_TAG = b"pp-audit/C_plan/v1"
HRAW_TAG = b"pp-audit/H_RAW/v1"
CDATA_TAG = b"pp-audit/C_data/v1"


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def plan_payload(profile, round_id, schedule=None):
    return {
        "v": "pp-audit/plan/v1",
        "round": round_id,
        "profile": profile,
        "schedule": schedule or {},
    }


def make_plan(profile, round_id, schedule=None):
    payload = plan_payload(profile, round_id, schedule)
    return payload, sha256_hex(PLAN_TAG + canonical(payload))


def entry_from_array(values, name):
    a = np.ascontiguousarray(values, dtype=np.float64)
    return {"path": name, "n": int(len(a)),
            "sha256": sha256_hex(a.tobytes())}


def data_payload(round_id, plan_sha256, entries):
    return {
        "v": "pp-audit/data/v1",
        "round": round_id,
        "plan_sha256": plan_sha256,
        "entries": sorted(entries, key=lambda e: e["path"]),
    }


def make_data(round_id, plan_sha256, entries):
    payload = data_payload(round_id, plan_sha256, entries)
    h_raw = sha256_hex(HRAW_TAG + canonical(payload))
    c_data = sha256_hex(CDATA_TAG + bytes.fromhex(h_raw))
    return payload, h_raw, c_data


def verify_passport(values, plan, data):
    """Check both commitments, the plan-data link, the delivered bytes, and
    the precision audit."""
    checks = {}

    payload = plan_payload(plan["profile"], plan["round"], plan.get("schedule"))
    c_plan = sha256_hex(PLAN_TAG + canonical(payload))
    checks["c_plan"] = {"recomputed": c_plan,
                        "committed": plan.get("c_plan"),
                        "ok": c_plan == plan.get("c_plan")}

    dpayload = data_payload(data["round"], data["plan_sha256"], data["entries"])
    h_raw = sha256_hex(HRAW_TAG + canonical(dpayload))
    c_data = sha256_hex(CDATA_TAG + bytes.fromhex(h_raw))
    checks["c_data"] = {"recomputed": c_data,
                        "committed": data.get("c_data"),
                        "h_raw_recomputed": h_raw,
                        "h_raw_committed": data.get("h_raw"),
                        "ok": (c_data == data.get("c_data")
                               and h_raw == data.get("h_raw"))}
    checks["plan_data_link"] = {
        "ok": data["plan_sha256"] == plan.get("c_plan"),
        "plan_sha256": data["plan_sha256"],
        "c_plan": plan.get("c_plan"),
    }

    entry = entry_from_array(values, data["entries"][0]["path"])
    checks["delivered"] = {
        "ok": (entry["sha256"] == data["entries"][0]["sha256"]
               and entry["n"] == data["entries"][0]["n"]),
        "entry": entry,
        "committed": data["entries"][0],
    }

    audit = core.verify(values, plan["profile"])
    checks["audit"] = {"verdict": audit["verdict"],
                       "issues": audit["issues"],
                       "b_global": audit["scan"]["b_global"],
                       "ok": audit["verdict"] == "pass"}

    ok = (checks["c_plan"]["ok"] and checks["c_data"]["ok"]
          and checks["plan_data_link"]["ok"] and checks["delivered"]["ok"]
          and audit["verdict"] == "pass")
    return {"verdict": "pass" if ok else "fail", "checks": checks,
            "audit": audit}
