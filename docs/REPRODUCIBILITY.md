# Using the data and code

## Model prediction

From the repository root:

```bash
python -m pip install -e .
python examples/predict.py --check-model
python examples/predict.py --input examples/compositions.csv --output outputs/example_predictions.csv
```

The example loads the ten-member production ensemble on CPU. Input rows identify the four metals as `M1`, `M2`, `M3` and `M4`; the supplied example shows the format. Predictions use the saved feature and target normalization. Output files use a new path under `outputs/`. See [model files](MODELS.md) and [environment setup](../environment/README.md).

## Result tables

The [data guide](DATA_DICTIONARY.md) defines units, sample membership and table-reading conventions. The [table index](../provenance/table_index.csv) lists columns and record counts. Empirical-expression definitions and full-precision coefficients are linked from the [descriptor guide](../data/analysis/descriptors/README.md).

## Training and data preparation

The [code guide](../scripts/README.md) maps the 18 programs for training, screening, manuscript comparisons, validation statistics and the D01 expression. The [script catalog](../provenance/script_catalog.csv) records which programs fit models or write results. Use a working copy when running programs that write to result or model directories. In particular, `scripts/modeling/screen_fullspace_v2.py` trains and writes to the production checkpoint path.

Electronic-structure and bonding results are supplied as processed numerical tables, indexed by the [electronic-structure guide](../data/analysis/electronic_structure/README.md). Calculation methods and parameters are described in [Methods](methods/methods_EN.md).

## File verification

```bash
python tools/verify_bundle.py
```

This checks file hashes, Python syntax, dataset membership and model metadata. `SHA256SUMS` describes the files in this repository; [file_manifest.csv](../provenance/file_manifest.csv) records their source mappings.
