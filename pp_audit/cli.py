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

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
