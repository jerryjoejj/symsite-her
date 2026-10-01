# Modeling and attribution results

## Model comparisons and screening

| Files | Contents |
|---|---|
| `ablation_strict_2x2*.csv`, `ablation_architecture_summary.csv` | Architecture/loss comparisons on the shared 87D input |
| `ablation_baselines.csv`, `model_comparison_v2.csv` | Baseline model errors |
| `xgb_vs_symmetry_paired_cv.csv`, `plan_a_parallel_5fold_cv.csv` | Matched-fold comparisons under the recorded validation protocols |
| `blindtest_v2_predictions_v3.csv`, `arch_advantage_blindtest.csv` | Held-out predictions and paired errors |
| `uncertainty*.csv` | Prediction dispersion, error ranking and calibration summaries |
| `ablation_acquisition*.csv`, `prior_roc*.csv` | Retrospective acquisition comparisons and prior calibration |
| `initial_500*.csv` | Initial-sampling membership and available labels |
| `M3M4_symmetry*.csv` and `.txt` | Remote-exchange pairs and summary statistics |
| `activity_map_28560_v3.csv`, `robust_shortlist_v3.csv`, `top_candidates_v3.csv` | Current production screening results |
| `verification_results*.csv`, `verification_v2top*.txt` | Candidate DFT results at successive validation stages |

Current manuscript datasets are collected under [data/processed](../../processed/). [Validation membership](../../processed/dft_validation/validation_membership.csv) identifies the final 60 compositions and their evaluation cohorts. Earlier result versions retain their model and batch identifiers.

## SHAP tables

| Table | Model/input and evaluation setup |
|---|---|
| [shap_three_way_site_attribution.csv](shap_three_way_site_attribution.csv) | Three-model comparison on 87D input; neural explainers use 100 background samples and evaluate all 995 training compositions |
| [shap_xgb_87d_by_site.csv](shap_xgb_87d_by_site.csv) | 87D XGBoost site attribution |
| [shap_grouped_site_attribution.csv](shap_grouped_site_attribution.csv) | Symmetry-model attribution for 122 high-activity and 873 other compositions, using all 995 as background |
| [shap_attribution_by_site.csv](shap_attribution_by_site.csv) | Historical 96D XGBoost feature-set comparison |

In the common-input comparison, the $M_1$ shares are 79.01% for XGBoost, 31.48% for the flat MLP and 39.95% for the symmetry ensemble. The neural global-feature shares are 60.14% and 53.64%, respectively. These quantities describe contributions to model predictions. Electronic and bonding responses are evaluated in the separate electronic-structure datasets.

The earlier `shap_symmetry_ensemble_by_site.csv` table lacks recorded background and evaluation membership. The manuscript comparison uses `shap_three_way_site_attribution.csv`. Available checkpoints and the XGBoost summary files are described in [model files](../../../docs/MODELS.md).
