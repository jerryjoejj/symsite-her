# Model files

| File | Role |
|---|---|
| `models/ensemble/symmetry_ensemble_v2_production_v3.pt` | Current ten-member, 87D production model used for the v3 full-space predictions |
| `models/ensemble/symmetry_ensemble_v2_production_v3_meta.json` | Human-readable production configuration and archived training-fit summary |
| `models/ensemble/symmetry_ensemble_v2_for_shap.pth` | Retained symmetry-model checkpoint used for neural attribution |
| `models/ensemble/flat_mlp_nll_for_shap.pt` | Retained flat-NLL checkpoint used for neural attribution |
| `models/ensemble_iter0.pt` | Checkpoint used for the executed acquisition round |
| `models/descriptors/frozen_expressions.json` | Final D01 expression `All_De1_original`, element inputs and training reference values |
| `data/analysis/descriptors/expression_validation/final_coefficients.csv` | Full-precision expression coefficients, with explicit model identifiers |

The 20 fold-specific D01/control models are stored in `data/analysis/descriptors/site_role_controls/outer_models.json`; the full-data D01 coefficients are stored in the final expression model.

Production checkpoint contents are configuration, per-block feature standardization, target normalization and member state dictionaries. Their Python configuration class remains in `herald.models.symmetry_aware_ensemble_v2`; the package namespace is therefore preserved.

The CPU-loading example checks the supplied checkpoint hash and loads its ten members on CPU.

## XGBoost attribution data

The XGBoost attribution is supplied as summary tables and analysis code. The original fitted 87D XGBoost checkpoint and per-sample SHAP matrix are unavailable. Calculation details are recorded in [shap_explainer_audit.json](../data/processed/design_principle/shap_explainer_audit.json).
