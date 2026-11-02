"""Core routines of the precision-passport audit.

The audit treats a delivered array of values as a measurement:

  * partition the values by binary octave |x| in [2^e, 2^(e+1));
  * for each octave, find the coarsest explicit mantissa count b whose grid
    u(e) = 2^(e-b) holds at least ``threshold`` of the unique values;
  * bin the values into 2^b equal bins over [-pi, pi) and certify the
    min-entropy floor with a Clopper-Pearson upper limit at alpha/2^b;
  * report the floor, the cap b - 1 + log2(2*pi), and the efficiency eta.

Only numpy and scipy are required.
"""

import math

import numpy as np
from scipy import stats

DEFAULTS = {
    "alpha": 1e-6,
    "b_min": 8,
    "b_max": 24,
    "b_range": [8, 16],
    "on_grid_threshold": 0.999,
    "on_grid_tolerance": 1e-3,
    "min_unique": 30,
}


def _octave_index(values):
    ax = np.abs(values)
    nz = ax > 0
    e = np.full(len(values), -999, dtype=int)
    e[nz] = np.floor(np.log2(ax[nz])).astype(int)
    return ax, nz, e


def on_grid_fraction(values, b, tolerance=1e-3):
    """Fraction of nonzero values within ``tolerance`` of the b-bit grid."""
    values = np.asarray(values, dtype=np.float64)
    ax, nz, e = _octave_index(values)
    if not np.any(nz):
        return 0.0
    step = np.exp2(e[nz] - b)
    r = np.abs(values[nz] / step - np.round(values[nz] / step))
    return float(np.mean(r < tolerance))


def octave_scan(values, b_min=DEFAULTS["b_min"], b_max=DEFAULTS["b_max"],
                threshold=DEFAULTS["on_grid_threshold"],
                tolerance=DEFAULTS["on_grid_tolerance"],
                min_unique=DEFAULTS["min_unique"]):
    """Coarsest passing b per octave, plus the global read-out."""
    values = np.asarray(values, dtype=np.float64)
    ax, nz, e = _octave_index(values)
    octaves = {}
    for ee in range(-8, 2):
        sel = nz & (e == ee)
        vals = np.unique(values[sel])
        if len(vals) < min_unique:
            continue
        best = None
        for b in range(b_min, b_max + 1):
            step = math.exp2(ee - b)
            r = np.abs(vals / step - np.round(vals / step))
            frac = float(np.mean(r < tolerance))
            if frac >= threshold:
                best = {"b": b, "on_grid_fraction": frac,
                        "n_unique": int(len(vals))}
                break
        if best is None:
            scored = []
            for b in range(b_min, b_max + 1):
                step = math.exp2(ee - b)
                r = np.abs(vals / step - np.round(vals / step))
                scored.append((float(np.mean(r < tolerance)), b))
            frac, b = max(scored)
            best = {"b": b, "on_grid_fraction": frac,
                    "n_unique": int(len(vals)), "no_pass": True}
        octaves[str(int(ee))] = best
    counts = {}
    for ee, m in octaves.items():
        if int(ee) >= -4 and not m.get("no_pass"):
            counts[m["b"]] = counts.get(m["b"], 0) + 1
    b_global = max(counts, key=counts.get) if counts else None
    return {"octaves": octaves, "b_global": b_global}


def cp_upper(k, n, alpha):
    """Clopper-Pearson upper limit for k successes in n trials."""
    if k >= n:
        return 1.0
    return float(stats.beta.ppf(1 - alpha, k + 1, n - k))


def floor_scan(values, b_range=DEFAULTS["b_range"], alpha=DEFAULTS["alpha"]):
    """Certified floor, cap and eta for each analysis resolution b."""
    values = np.asarray(values, dtype=np.float64)
    n = int(len(values))
    rows = []
    for b in range(int(b_range[0]), int(b_range[1]) + 1):
        nb = 1 << b
        k = np.minimum(((values + math.pi) / (2 * math.pi) * nb).astype(np.int64),
                       nb - 1)
        counts = np.bincount(k, minlength=nb)
        nmax = int(counts.max())
        h = -math.log2(cp_upper(nmax, n, alpha / nb))
        cap = b - 1 + math.log2(2 * math.pi)
        rows.append({"b": b, "n": n, "n_max": nmax,
                     "h_floor": h, "h_cap": cap, "eta": h / cap})
    return rows


def measure_profile(values, name="product"):
    """Measure a precision profile from the delivered values."""
    scan = octave_scan(values)
    return {
        "product": name,
        "mantissa_bits": scan["b_global"],
        "step_law": "u(e)=2^(e-b)",
        "quantizer": "float mantissa lattice",
        "alpha": DEFAULTS["alpha"],
        "on_grid_threshold": DEFAULTS["on_grid_threshold"],
        "b_range": DEFAULTS["b_range"],
        "scan": scan,
    }


def verify(values, profile):
    """Audit the delivered values against a declared profile."""
    declared = int(profile["mantissa_bits"])
    threshold = float(profile.get("on_grid_threshold",
                                  DEFAULTS["on_grid_threshold"]))
    scan = octave_scan(values, threshold=threshold)
    issues = []
    for ee, m in sorted(scan["octaves"].items(), key=lambda kv: int(kv[0])):
        if m.get("no_pass"):
            issues.append("octave %s: no lattice fit at b <= %d"
                          % (ee, DEFAULTS["b_max"]))
        elif m["b"] != declared:
            issues.append("octave %s: measured b=%d, declared %d"
                          % (ee, m["b"], declared))
    floors = floor_scan(values, b_range=profile.get("b_range",
                                                    DEFAULTS["b_range"]),
                        alpha=float(profile.get("alpha", DEFAULTS["alpha"])))
    verdict = "pass" if not issues else "fail"
    return {"verdict": verdict, "issues": issues,
            "declared_mantissa_bits": declared,
            "scan": scan, "floors": floors}
