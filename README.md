# pp-audit — precision-passport audit for public data products

`pp-audit` is the reference implementation of the storage-precision audit
described in *A computational-physics audit of storage precision in public HEP
data products*. It recovers the storage lattice of a delivered array, certifies
a min-entropy floor, and verifies the values against a declared precision
profile.

The package accompanies the manuscript *A computational-physics audit of
storage precision in public HEP data products* (EPJ C, Computing, Software and
Data Science section). The measured profiles for the two products are in
`profiles/` (`cms_lead.json`, `atlas_lead.json`).

## Install

```
pip install -e .
```

Dependencies: numpy, scipy (Python >= 3.9).

## Usage

```
# measure a profile from delivered values
python -m pp_audit profile --input values.npy --out profile.json

# audit values against a declared profile
python -m pp_audit verify --input values.npy --profile profile.json --out audit.json

# pool a product made of many arrays (e.g. one branch over 67 files)
python -m pp_audit verify --input 'c5G/*.npz' --key lead_phi \
       --profile examples/profile_cms_lead.json --out audit.json

# print the JSON schema
python -m pp_audit schema
```

Inputs are one-dimensional arrays of values in `[-pi, pi)` (`.npy`, or `.npz`
with `--key`). A shell-style glob pools the same key from several archives.

## Example

```
cd examples
python make_example.py          # writes values.npy and profile.json
python -m pp_audit verify --input values.npy --profile profile.json --out audit.json
```

`profiles/cms_lead.json` is the profile of the CMS PFNanoAOD leading-jet
azimuth branch (record 31316, Run2016G), measured in the paper;
`profiles/atlas_lead.json` is the ATLAS Open Data profile (record 93933,
full float32).

## Passport protocol

The two-stage commitment is also implemented:

```
python -m pp_audit passport commit  --profile profiles/cms_lead.json \
    --round R --out plan.json
python -m pp_audit passport deliver --input values.npy --plan plan.json \
    --out data.json
python -m pp_audit passport verify  --input values.npy --plan plan.json \
    --data data.json --out passport.json
python -m pp_audit passport demo    --out demo/
```

`C_plan` is committed before release; `C_data` binds the delivered array;
`verify` recomputes both commitments, checks the delivered hash, and audits
the values against the committed profile. The synthetic demo exercises the
honest release, a mantissa truncation (hash + audit fail) and a single-value
flip (hash only).

## Output

The audit result contains:

- `verdict`: `pass` or `fail`;
- `issues`: the per-octave mismatches, if any;
- `scan.octaves`: the coarsest passing mantissa count per octave, with its
  on-grid fraction;
- `floors`: for each analysis resolution `b`, the maximum bin count, the
  Clopper-Pearson floor, the cap `b - 1 + log2(2*pi)`, and `eta`.

The JSON schema is `pp_audit/schema.json`.

## Scope

The audit certifies the precision profile of the delivered values: mantissa
count, step law and quantizer. It is a statistical profile check, not a
per-value integrity check; sparse changes below `(1 - threshold) n` are outside
its scope and are covered by a content hash. See the paper for the detection
limit and the tamper tests.

## Tests

```
python -m unittest discover -s tests -v
```

The suite includes a synthetic 12-bit round trip, a truncation test, a
uniform-quantizer test, and a fixed-seed regression hash.

## Docker

```
docker build -t pp-audit .
docker run --rm -v "$PWD:/data" pp-audit verify \
  --input /data/values.npy --profile /data/profile.json --out /data/audit.json
```

## Data

The paper uses public CERN Open Data records: CMS 31316 (PFNanoAOD ZeroBias
Run2016G, CC0-1.0) and ATLAS 93933 (Run 2 2015+2016, CC0-1.0). The audit
does not redistribute the products; the scripts read local arrays.

## License

MIT (repository license).
