# SymSite-HER

Code, models and result data for **Multimetallic Single-Atom Catalysts for Hydrogen Evolution: Symmetry-Guided Discovery and Metal–N Bonding Modulation**.

SymSite-HER represents the adsorption center, neighboring metal and unordered remote-metal pair separately. It includes composition features, ensemble prediction, candidate screening, DFT validation results, and site-resolved electronic and bonding analysis.

[Content guide](docs/CORE_CONTENTS.md) · [Data guide](docs/DATA_DICTIONARY.md) · [Usage](docs/REPRODUCIBILITY.md)

## Quick start

```bash
python tools/verify_bundle.py
python -m pip install -e .
python examples/predict.py --check-model
python examples/predict.py --input examples/compositions.csv --output outputs/example_predictions.csv
```

The commands check the supplied files, install the package, load the production model on CPU, and predict adsorption free energies for example compositions. The `herald` Python namespace provides checkpoint compatibility. See [model files](docs/MODELS.md) and [environment setup](environment/README.md).

## Main datasets

| Dataset | Records | Location |
|---|---:|---|
| Training compositions | 995 | [Training](data/processed/training/) |
| Held-out predictions | 110 | [Evaluation](data/processed/architecture/blindtest_v2_predictions_v3.csv) |
| Candidate validation | 60 | [Membership](data/processed/dft_validation/validation_membership.csv) |
| Full composition-space predictions | 28,560 | [Screening](data/processed/screening/activity_map_28560_v3.csv) |
| All-3d candidates | 137 | [Candidate table](data/processed/screening/SI_table_cheap_candidates.csv) |
| Paired electronic records | 129 across 107 compositions | [Electronic changes](data/analysis/electronic_structure/paired_responses/paired_electronic_changes.csv) |

Data are provided as processed result tables and numerical arrays. Definitions, units and dataset memberships are described in the [data guide](docs/DATA_DICTIONARY.md).

## Repository layout

```text
src/herald/       Feature, symmetry, model and acquisition library
scripts/          Model training, screening and manuscript comparisons
models/           Ensemble checkpoints and empirical expressions
data/             Result tables, element properties and composition features
docs/             Methods, supporting information and user guides
examples/         Model-loading and prediction examples
tools/            File and dataset verification
environment/      Dependencies and recorded package versions
provenance/       Source mappings, table index and file hashes
```

## Methods and data

- [Methods](docs/methods/methods_EN.md) and [Supporting Information](docs/methods/supporting_information_EN.md).
- [Empirical expressions](data/analysis/descriptors/README.md) and [electronic structure](data/analysis/electronic_structure/README.md).
- [File manifest](provenance/file_manifest.csv) and [table index](provenance/table_index.csv).
- [License information](docs/LICENSE_STATUS.md).
