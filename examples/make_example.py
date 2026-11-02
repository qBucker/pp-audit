#!/usr/bin/env python3
"""Generate a small synthetic 12-bit product for the pp-audit example."""

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    rng = np.random.default_rng(20261102)
    x = rng.uniform(-math.pi, math.pi, 200000)
    ax = np.abs(x)
    e = np.floor(np.log2(ax)).astype(int)
    step = np.exp2(e - 12)
    values = (np.round(x / step) * step).astype(np.float32)
    np.save(os.path.join(HERE, "values.npy"), values)
    profile = {
        "product": "synthetic 12-bit product",
        "mantissa_bits": 12,
        "step_law": "u(e)=2^(e-b)",
        "quantizer": "float32 mantissa lattice, round-to-nearest",
        "alpha": 1e-06,
        "on_grid_threshold": 0.999,
        "b_range": [8, 16],
    }
    with open(os.path.join(HERE, "profile.json"), "w") as fh:
        json.dump(profile, fh, indent=2)
        fh.write("\n")
    print("wrote values.npy (%d values) and profile.json" % len(values))


if __name__ == "__main__":
    main()
