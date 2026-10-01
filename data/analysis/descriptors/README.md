# Site-resolved empirical expression

This directory contains the D01 expression and the results used in manuscript Section R7, Figure 8 and Figure S12.

| Directory | Supplied result |
|---|---|
| [expression_definition](expression_definition/) | Formula definitions, input conventions and worked example |
| [expression_validation](expression_validation/) | The 17 center-specific coefficient sets and the 110-record evaluation summary |
| [site_role_controls](site_role_controls/) | Original/unordered components, two grouped evaluations, 20 fold models, predictions and per-center errors |
| [validation_protocol](validation_protocol/) | Composition-grouped splits and evaluation settings |
| [frozen_expression_validation](frozen_expression_validation/) | 110 evaluation records and the 220 predictions from D01 and the center-median baseline |

## Manuscript results

| Comparison | D01 | Reference |
|---|---:|---:|
| First grouped scheme, macro MAE (eV) | 0.118106 | 0.132774, unordered representation |
| Second grouped scheme, macro MAE (eV) | 0.118411 | 0.131039, unordered representation |
| 110-record evaluation, MAE (eV) | 0.125739 | 0.148560, center-median baseline |
| Qualifying compositions among the top 22 | 10 | 10.2, tie-averaged center-median baseline |

The two grouped schemes use the same 995 compositions. Each representation has 17 intercepts and 34 slopes. The full-data expression is `All_De1_original`, with Ridge `alpha=1`; its 110-record evaluation uses the saved coefficients. Local Cr/Fe/Ni results are provided in the [comparison table](../electronic_structure/model_link/S2I_evidence_gate/analysis/composition_predictions.csv).

## Use

- [Formula, example and coefficients](expression_definition/section_detailed_EN.md).
- [Final model and element properties](../../../models/descriptors/frozen_expressions.json).
- [Prediction program](../../../scripts/mechanistic_analysis/candidate_predictor.py).
- [Site-role functions and result summary](../../../scripts/mechanistic_analysis/site_role_controls.py).

From the repository root, summarize the supplied grouped predictions:

```bash
python scripts/mechanistic_analysis/site_role_controls.py
```

Predict with the final formula:

```bash
python scripts/mechanistic_analysis/candidate_predictor.py \
  --lock models/descriptors/frozen_expressions.json \
  --model All_De1_original --input examples/compositions.csv \
  --output descriptor_predictions.csv
```

The prediction command creates a new output file. Full-precision coefficients multiply the stated inputs directly.
