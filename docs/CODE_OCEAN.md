# Code Ocean setup

## Export

```bash
python tools/verify_bundle.py
python tools/build_capsule.py
```

The exporter creates `outputs/codeocean/capsule/` and a ZIP with the same files. It copies scientific inputs from the manifest, puts the library in `code/src/herald/`, and places checkpoints in `data/models/`.

```text
code/          run, reproduce.py, train.py, input_manifest.json, src/, docs/
data/          reproduction/, processed/, analysis/, elements/, models/
environment/   dependency pins, postInstall, Dockerfile
metadata/      capsule.json
```

## Hosted run

1. Create a capsule and upload exported folders into the matching locations. Use `metadata/capsule.json` for the title, description and authors.
2. Select Linux with Python 3.13.9 and at least 4 GB RAM. Install CPU PyTorch 2.12.1 and the pinned dependencies using `postInstall`. The Dockerfile provides an equivalent local recipe.
3. Set `code/run` as the run script. It writes to `/results/nc_reproduction/` and requires no runtime network access.
4. Start a Reproducible Run. Check that `report.json` reports `status: pass`, `all_checks_passed: true` and 201 checks. Review `manuscript_checks.csv` and the output tables.
5. Complete publication metadata, including the author-selected license, then share or publish the capsule for submission.

The [Code Ocean reproducible-run documentation](https://docs.codeocean.com/user-guide/v4.2.0/compute-capsule-basics/reproducible-runs) specifies headless execution and the `/code`, `/data`, `/results` locations. The analysis uses only its packaged code and data at runtime.

## Local capsule run

After installing dependencies, run `bash outputs/codeocean/capsule/code/run` from the repository root. For a repeat, provide a new `--output` directory. Local validation is recorded in `provenance/validation_report.json`; a hosted run produces its own report and Code Ocean execution record.
