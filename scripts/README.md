# Code guide

The scripts support the manuscript's model, screening and expression results.

| Purpose | Programs | Result data |
|---|---|---|
| Composition space and training features | [generate_phase_space_v2.py](modeling/generate_phase_space_v2.py), [recompute_training_features.py](modeling/recompute_training_features.py) | `data/space/`, `data/processed/training/` |
| Production ensemble and full-space screening | [screen_fullspace_v2.py](modeling/screen_fullspace_v2.py), [robust_shortlist_v3.py](modeling/robust_shortlist_v3.py) | `models/ensemble/`, `data/processed/screening/` |
| Architecture, loss and model comparisons | [ablation_strict_2x2.py](modeling/ablation_strict_2x2.py), [model_comparison_v2.py](modeling/model_comparison_v2.py), [arch_advantage_blindtest.py](modeling/arch_advantage_blindtest.py) | `data/processed/architecture/` |
| Chemically guided selection | [ablation_acquisition.py](modeling/ablation_acquisition.py) | `data/processed/acquisition/` |
| Held-out uncertainty evaluation | [uncertainty_calibration_v3.py](modeling/uncertainty_calibration_v3.py) | `data/processed/architecture/` |
| SHAP model attribution | [_shap_common.py](modeling/_shap_common.py), [shap_xgb_87d_control.py](modeling/shap_xgb_87d_control.py), [shap_flat_mlp_baseline.py](modeling/shap_flat_mlp_baseline.py), [shap_three_way_comparison.py](modeling/shap_three_way_comparison.py), [train_symmetry_ensemble_v2_for_shap.py](modeling/train_symmetry_ensemble_v2_for_shap.py) | `data/processed/design_principle/` |
| Validation membership and statistics | [prepare_validation_membership.py](data_preparation/prepare_validation_membership.py) | `data/processed/dft_validation/` |
| D01 expression and site-role comparison | [candidate_predictor.py](mechanistic_analysis/candidate_predictor.py), [site_role_controls.py](mechanistic_analysis/site_role_controls.py), [parameter_sharing.py](mechanistic_analysis/parameter_sharing.py) | `data/analysis/descriptors/`, `models/descriptors/` |

For prediction with the supplied production checkpoint, use [examples/predict.py](../examples/predict.py). The [descriptor guide](../data/analysis/descriptors/README.md) gives the D01 prediction command and the saved-result summary command.

The training and comparison scripts write to their configured paths, commonly `data/analysis/modeling/`; the corresponding published result tables are organized under `data/processed/`. Run programs that write files in a working copy. `screen_fullspace_v2.py` trains the production ensemble and writes its checkpoint.

`model_comparison_v2.py` includes the model selection and five-fold comparisons reported in the manuscript. SHAP programs use the common 87-dimensional input and the recorded neural-network background selection. Scientific settings are described in [Methods](../docs/methods/methods_EN.md).

See [usage](../docs/REPRODUCIBILITY.md), [dependencies](../environment/README.md) and the [script catalog](../provenance/script_catalog.csv).
