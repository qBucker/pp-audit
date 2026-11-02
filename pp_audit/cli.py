"""Command-line interface for the precision-passport audit.

Usage:
  python -m pp_audit profile --input values.npy --out profile.json
  python -m pp_audit verify  --input values.npy --profile profile.json \
                             --out audit.json
  python -m pp_audit schema

Inputs are .npy arrays or .npz archives (use --key for the latter). All
values are float32/float64 azimuth-like quantities in [-pi, pi).
"""

import argparse
import glob as globmod
import json
import os
import sys

import numpy as np

from . import core
from . import passport


def load_values(path, key=None):
    if globmod.has_magic(path):
        arrays = []
        for p in sorted(globmod.glob(path)):
            data = np.load(p)
            k = key if key is not None else sorted(data.keys())[0]
            arrays.append(np.asarray(data[k], dtype=np.float64))
        if not arrays:
            raise SystemExit("no files match %s" % path)
        return np.concatenate(arrays)
    if path.endswith(".npz"):
        data = np.load(path)
        if key is None:
            key = sorted(data.keys())[0]
        return np.asarray(data[key], dtype=np.float64)
    return np.asarray(np.load(path), dtype=np.float64)


def write_json(obj, path):
    if path:
        with open(path, "w") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True)
            fh.write("\n")
    else:
        json.dump(obj, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")


def cmd_profile(args):
    values = load_values(args.input, args.key)
    out = core.measure_profile(values, name=args.name)
    out["n"] = int(len(values))
    write_json(out, args.out)
    if args.out:
        print("wrote %s (b_global=%s, n=%d)"
              % (args.out, out["mantissa_bits"], out["n"]))


def cmd_verify(args):
    values = load_values(args.input, args.key)
    with open(args.profile) as fh:
        profile = json.load(fh)
    out = core.verify(values, profile)
    out["n"] = int(len(values))
    out["input"] = os.path.basename(args.input)
    write_json(out, args.out)
    if args.out:
        print("wrote %s (verdict=%s, b_global=%s)"
              % (args.out, out["verdict"], out["scan"]["b_global"]))


def cmd_schema(args):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.json")
    with open(path) as fh:
        sys.stdout.write(fh.read())


def cmd_passport_commit(args):
    profile = json.load(open(args.profile))
    payload, c_plan = passport.make_plan(profile, args.round, args.schedule)
    payload["c_plan"] = c_plan
    write_json(payload, args.out)
    if args.out:
        print("wrote %s (C_plan=%s)" % (args.out, c_plan[:16]))


def cmd_passport_deliver(args):
    plan = json.load(open(args.plan))
    values = load_values(args.input, args.key)
    name = args.name or os.path.basename(args.input)
    entry = passport.entry_from_array(values, name)
    data, h_raw, c_data = passport.make_data(plan["round"], plan["c_plan"],
                                             [entry])
    data["h_raw"] = h_raw
    data["c_data"] = c_data
    write_json(data, args.out)
    if args.out:
        print("wrote %s (C_data=%s)" % (args.out, c_data[:16]))


def cmd_passport_verify(args):
    plan = json.load(open(args.plan))
    data = json.load(open(args.data))
    values = load_values(args.input, args.key)
    res = passport.verify_passport(values, plan, data)
    write_json(res, args.out)
    failed = [k for k, v in res["checks"].items()
              if isinstance(v, dict) and v.get("ok") is False]
    if args.out:
        print("wrote %s (verdict=%s, failed=%s)"
              % (args.out, res["verdict"], ",".join(failed) or "none"))


def _quantize(x, b):
    ax = np.abs(x)
    e = np.floor(np.log2(ax)).astype(int)
    step = np.exp2(e - b)
    return np.round(x / step) * step


def _truncate(x, b_from, b_to):
    ax = np.abs(x)
    e = np.floor(np.log2(ax)).astype(int)
    step = np.exp2(e - b_to)
    return np.trunc(x / step) * step


def cmd_passport_demo(args):
    rng = np.random.default_rng(args.seed)
    x = rng.uniform(-np.pi, np.pi, args.n)
    values = _quantize(x, 12)
    profile = {
        "product": "synthetic 12-bit product",
        "mantissa_bits": 12,
        "step_law": "u(e)=2^(e-b)",
        "quantizer": "float32 mantissa lattice, round-to-nearest",
        "alpha": 1e-06,
        "on_grid_threshold": 0.999,
        "b_range": [8, 16],
    }
    plan, c_plan = passport.make_plan(profile, "demo-2026-11-02")
    plan["c_plan"] = c_plan
    entry = passport.entry_from_array(values, "values.npy")
    data, h_raw, c_data = passport.make_data(plan["round"], c_plan, [entry])
    data["h_raw"] = h_raw
    data["c_data"] = c_data
    honest = passport.verify_passport(values, plan, data)

    t1 = _truncate(values, 12, 11)
    trunc = passport.verify_passport(t1, plan, data)

    t2 = values.copy()
    idx = len(t2) // 2
    e = int(np.floor(np.log2(abs(t2[idx]))))
    t2[idx] += 2.0 ** (e - 12) / 2.0
    single = passport.verify_passport(t2, plan, data)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        np.save(os.path.join(args.out, "values.npy"), values)
        np.save(os.path.join(args.out, "values_truncated.npy"), t1)
        np.save(os.path.join(args.out, "values_single_flip.npy"), t2)
        write_json(plan, os.path.join(args.out, "plan.json"))
        write_json(data, os.path.join(args.out, "data.json"))
        summary = {
            "profile": profile,
            "plan": plan,
            "data": data,
            "honest": {"verdict": honest["verdict"], "failed": []},
            "tamper_truncate": {
                "verdict": trunc["verdict"],
                "failed": [k for k, v in trunc["checks"].items()
                           if isinstance(v, dict) and v.get("ok") is False]},
            "tamper_single_flip": {
                "verdict": single["verdict"],
                "failed": [k for k, v in single["checks"].items()
                           if isinstance(v, dict) and v.get("ok") is False]},
        }
        write_json(summary, os.path.join(args.out, "demo.json"))
    print("C_plan=%s" % c_plan[:16])
    print("C_data=%s" % c_data[:16])
    print("%-22s %-6s %s" % ("step", "verdict", "failed checks"))
    for name, r in (("honest", honest), ("truncate 11 bits", trunc),
                    ("single-value flip", single)):
        failed = [k for k, v in r["checks"].items()
                  if isinstance(v, dict) and v.get("ok") is False]
        print("%-22s %-6s %s" % (name, r["verdict"], ",".join(failed) or "none"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pp-audit",
                                 description="Precision-passport audit")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("profile", help="measure a precision profile")
    p.add_argument("--input", required=True)
    p.add_argument("--key", default=None, help="array key for .npz input")
    p.add_argument("--name", default="product")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_profile)

    v = sub.add_parser("verify", help="audit values against a profile")
    v.add_argument("--input", required=True)
    v.add_argument("--key", default=None)
    v.add_argument("--profile", required=True)
    v.add_argument("--out", default=None)
    v.set_defaults(func=cmd_verify)

    s = sub.add_parser("schema", help="print the JSON schema")
    s.set_defaults(func=cmd_schema)

    pp = sub.add_parser("passport", help="two-stage precision passport")
    psub = pp.add_subparsers(dest="pcommand", required=True)
    pc = psub.add_parser("commit", help="commit the precision profile")
    pc.add_argument("--profile", required=True)
    pc.add_argument("--round", required=True)
    pc.add_argument("--schedule", default=None)
    pc.add_argument("--out", default=None)
    pc.set_defaults(func=cmd_passport_commit)
    pd = psub.add_parser("deliver", help="commit the delivered array")
    pd.add_argument("--input", required=True)
    pd.add_argument("--key", default=None)
    pd.add_argument("--name", default=None)
    pd.add_argument("--plan", required=True)
    pd.add_argument("--out", default=None)
    pd.set_defaults(func=cmd_passport_deliver)
    pv = psub.add_parser("verify", help="verify a passport")
    pv.add_argument("--input", required=True)
    pv.add_argument("--key", default=None)
    pv.add_argument("--plan", required=True)
    pv.add_argument("--data", required=True)
    pv.add_argument("--out", default=None)
    pv.set_defaults(func=cmd_passport_verify)
    pm = psub.add_parser("demo", help="run the synthetic passport demo")
    pm.add_argument("--n", type=int, default=200000)
    pm.add_argument("--seed", type=int, default=20261102)
    pm.add_argument("--out", default=None)
    pm.set_defaults(func=cmd_passport_demo)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
