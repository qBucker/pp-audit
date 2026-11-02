"""Unit and regression tests for pp-audit.

Run from passport/:  python -m unittest discover -s tests -v
"""

import hashlib
import json
import math
import unittest

import numpy as np

from pp_audit import core


def quantize(x, b):
    ax = np.abs(x)
    e = np.floor(np.log2(ax)).astype(int)
    step = np.exp2(e - b)
    return np.round(x / step) * step


def make_values(n=20000, seed=1):
    rng = np.random.default_rng(seed)
    return rng.uniform(-math.pi, math.pi, n)


class TestAudit(unittest.TestCase):
    def test_12bit_roundtrip(self):
        values = quantize(make_values(), 12)
        profile = core.measure_profile(values)
        self.assertEqual(profile["mantissa_bits"], 12)
        res = core.verify(values, profile)
        self.assertEqual(res["verdict"], "pass")
        self.assertEqual(res["scan"]["b_global"], 12)
        for m in res["scan"]["octaves"].values():
            self.assertEqual(m["b"], 12)

    def test_truncation_detected(self):
        values = quantize(make_values(), 12)
        ax = np.abs(values)
        e = np.floor(np.log2(ax)).astype(int)
        step11 = np.exp2(e - 11)
        truncated = np.trunc(values / step11) * step11
        profile = core.measure_profile(values)
        res = core.verify(truncated, profile)
        self.assertEqual(res["verdict"], "fail")
        self.assertEqual(res["scan"]["b_global"], 11)

    def test_uniform_quantizer_detected(self):
        values = make_values()
        delta = 2 * math.pi / 4096
        k = np.round((values + math.pi) / delta)
        uniform = np.clip(k * delta - math.pi, -math.pi, math.pi - delta)
        profile = {"mantissa_bits": 12, "step_law": "u(e)=2^(e-b)",
                   "alpha": 1e-6, "on_grid_threshold": 0.999,
                   "b_range": [8, 16]}
        res = core.verify(uniform, profile)
        self.assertEqual(res["verdict"], "fail")
        self.assertTrue(any("no lattice fit" in s for s in res["issues"]))

    def test_regression_hash(self):
        values = quantize(make_values(n=20000, seed=7), 12)
        profile = core.measure_profile(values)
        res = core.verify(values, profile)
        summary = {
            "b_global": res["scan"]["b_global"],
            "verdict": res["verdict"],
            "per_octave": {k: v["b"]
                           for k, v in res["scan"]["octaves"].items()},
            "floor_b12": round(
                [r for r in res["floors"] if r["b"] == 12][0]["h_floor"], 9),
        }
        digest = hashlib.sha256(
            json.dumps(summary, sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest, EXPECTED_HASH)


EXPECTED_HASH = "7a7fb10a172bf63b422bb4ba57c22c526d8e481974041d896aa3b73d82e78a87"


if __name__ == "__main__":
    unittest.main()
