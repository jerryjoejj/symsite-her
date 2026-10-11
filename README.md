# SymSite-HER

Data, trained models and executable analyses for **Multimetallic Single-Atom Catalysts for Hydrogen Evolution: Symmetry-Guided Discovery and Cooperative Site Effects**, prepared for Nature Communications.

## Reproduce the results

Use Python 3.13 on Linux:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.12.1
python -m pip install -r environment/requirements-reproduce.txt
bash code/run
```

The run rebuilds the training labels and features, predicts all 28,560 compositions with the supplied ten-member ensemble, repeats statistical comparisons, refits the empirical expressions, and calculates electronic and bonding responses from final result tables. It writes 201 numerical checks and 43 CSV tables to `outputs/nc_reproduction/`, together with fitted coefficients and `report.json`. Failed checks return a nonzero exit status. Use `--output outputs/another_run` to repeat the run in a new directory.

## Included data

| Data | Size | Location |
|---|---:|---|
| Complete training source records | 1,206: 500 initial + 98 adaptive + 608 reference | [Training records](data/reproduction/training_records_1206.csv) |
| Deduplicated training compositions | 995 | [Training](data/processed/training/) |
| Initial and adaptive selections | 500 and 100; 98 valid adaptive results | [Initial list](data/processed/cohorts/round_0/selection/selected_500_structures.csv), [adaptive list](data/reproduction/adaptive_selection_100.csv) |
| Held-out compositions | 110 | [Predictions and labels](data/processed/architecture/blindtest_v2_predictions_v3.csv) |
| DFT-validated candidates | 60 | [Final energies](data/processed/dft_validation/verification_results_all60.csv) |
| Full-space predictions / final all-3d candidates | 28,560 / 137 | [Screening](data/processed/screening/) |
| Paired Co electronic endpoints | 129 pairs, 107 compositions | [Endpoints](data/reproduction/paired_electronic_endpoints.csv) |

The data include final energies, charges, orbital populations, bond populations, processed spectra, model outputs and evaluation records. Trained models are in `models/`.

## Code Ocean

```bash
python tools/verify_bundle.py
python tools/build_capsule.py
```

This creates `outputs/codeocean/capsule/` and `outputs/codeocean/symsite-her-codeocean.zip`. The capsule run command is `bash run`. See [Code Ocean setup](docs/CODE_OCEAN.md).

## Navigation

- [Result-to-input mapping and procedures](docs/REPRODUCIBILITY.md)
- [Data definitions, units and identities](docs/DATA_DICTIONARY.md)
- [Model training](docs/TRAINING.md)
- [NC methods](docs/methods.md), [supplementary methods](docs/supplementary_methods.md), [references](docs/references.md)
- [Input checksums](code/input_manifest.json), [file inventory](provenance/file_manifest.csv)
- [License status](docs/LICENSE_STATUS.md)

`code/` contains entry points, `src/herald/` provides feature and model implementations, and `tools/` provides verification and capsule export. Runtime outputs go to a separate results directory.
