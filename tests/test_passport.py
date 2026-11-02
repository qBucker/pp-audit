"""Tests for the two-stage precision-passport commitments."""

import math
import unittest

import numpy as np

from pp_audit import passport


def quantize(x, b):
    ax = np.abs(x)
    e = np.floor(np.log2(ax)).astype(int)
    step = np.exp2(e - b)
    return np.round(x / step) * step


def make_values(n=10000, seed=3):
    rng = np.random.default_rng(seed)
    return rng.uniform(-math.pi, math.pi, n)


PROFILE = {
    "product": "synthetic 12-bit product",
    "mantissa_bits": 12,
    "step_law": "u(e)=2^(e-b)",
    "quantizer": "float32 mantissa lattice, round-to-nearest",
    "alpha": 1e-06,
    "on_grid_threshold": 0.999,
    "b_range": [8, 16],
}


def build(values):
    plan, c_plan = passport.make_plan(PROFILE, "test-round")
    plan["c_plan"] = c_plan
    entry = passport.entry_from_array(values, "values.npy")
    data, h_raw, c_data = passport.make_data("test-round", c_plan, [entry])
    data["h_raw"] = h_raw
    data["c_data"] = c_data
    return plan, data


class TestPassport(unittest.TestCase):
    def test_honest_pass(self):
        values = quantize(make_values(), 12)
        plan, data = build(values)
        res = passport.verify_passport(values, plan, data)
        self.assertEqual(res["verdict"], "pass")
        for name in ("c_plan", "c_data", "plan_data_link", "delivered",
                     "audit"):
            self.assertTrue(res["checks"][name]["ok"], name)

    def test_truncation_fails_both_layers(self):
        values = quantize(make_values(), 12)
        plan, data = build(values)
        ax = np.abs(values)
        e = np.floor(np.log2(ax)).astype(int)
        step11 = np.exp2(e - 11)
        truncated = np.trunc(values / step11) * step11
        res = passport.verify_passport(truncated, plan, data)
        self.assertEqual(res["verdict"], "fail")
        self.assertFalse(res["checks"]["delivered"]["ok"])
        self.assertFalse(res["checks"]["audit"]["ok"])

    def test_single_flip_fails_hash_only(self):
        values = quantize(make_values(), 12)
        plan, data = build(values)
        t = values.copy()
        idx = len(t) // 2
        e = int(np.floor(np.log2(abs(t[idx]))))
        t[idx] += 2.0 ** (e - 12) / 2.0
        res = passport.verify_passport(t, plan, data)
        self.assertEqual(res["verdict"], "fail")
        self.assertFalse(res["checks"]["delivered"]["ok"])
        self.assertTrue(res["checks"]["audit"]["ok"])

    def test_wrong_commitment_detected(self):
        values = quantize(make_values(), 12)
        plan, data = build(values)
        plan["c_plan"] = "0" * 64
        res = passport.verify_passport(values, plan, data)
        self.assertEqual(res["verdict"], "fail")
        self.assertFalse(res["checks"]["c_plan"]["ok"])


if __name__ == "__main__":
    unittest.main()
