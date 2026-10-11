# Reproducing the NC results

The manuscript title, date and source hashes are in `provenance/manuscript_snapshot.json`. Reference values and tolerances are explicit in `code/reproduce.py`; every comparison is written to `manuscript_checks.csv`.

## Result map

All input paths below are relative to `data/`, except model checkpoints.

| Manuscript result | Inputs | Executed operation | Main output |
|---|---|---|---|
| R1: 1,206 records to 995 compositions | `reproduction/training_records_1206.csv`, element properties | Group remote exchanges; take clean/H minima separately; recalculate labels and features | `training_995_rebuilt.csv` |
| R1; SI S1.2–S1.4: exchange symmetry | `processed/symmetry/` | Recalculate pair deviations, threshold counts and Mann–Whitney test | `remote_exchange_deviations.csv` |
| R2; Tables S3–S4: model comparison | Fold errors, 110 paired predictions | Count network parameters; aggregate MAEs; run paired tests | `architecture_fold_summary.csv` |
| R3; Table S5: prior and acquisition | 995 labels, 80 round records | Recalibrate prior; calculate paired statistics for eight repeats | `chemical_prior_scores.csv`, `retrospective_paired_summary.csv` |
| R4; Table S1: candidate screening | Element properties, production checkpoint | Enumerate 28,560 compositions; generate features; infer and rank candidates | `full_space_predictions.csv`, `all_3d_candidates.csv` |
| R2/R5: uncertainty and validation | Held-out labels, 60 final DFT results, new predictions | Recalculate uncertainty ranking, overlaps, correlations, errors and hit counts | `validation_membership_and_predictions.csv` |
| R6: center properties and SHAP | Labels, SHAP arrays, Co/Sc observations | Refit five-attribute Ridge; recompute variance, attribution shares and electronic means | `five_attribute_predictions.csv`, `Sc_Co_electronic_summary.csv` |
| R7; Figure 8/S12: empirical expression | Saved grouped splits, selected penalties, labels | Refit all 20 fold models and full-data expression; recalculate macro MAE and top-22 hits | `D01_grouped_predictions.csv`, `D01_refitted_full_model.json` |
| R7; SI S1.10: Co–N response | Clean/H charges, orbital/bond populations, distances and energies | Calculate differences, atom/bond sums, orbital contrasts, fixed-geometry J and state sensitivity | `CoN_bonding_response.csv`, `CoN_shell_contrasts.csv`, `fixed_geometry_J.csv` |

## Execution

`bash code/run` verifies SHA-256 checksums, loads the production model on CPU, and runs without network access. The ten supplied members generate new predictions. The five-attribute Ridge and grouped empirical models are fitted during each run. Architecture fold errors, acquisition-round observations and SHAP values are supplied result data from which their reported statistics are recalculated. [Separate training commands](TRAINING.md) execute the neural-model and acquisition training procedures.

Production predictions are compared against every stored value with an absolute tolerance of 0.000003 eV. Empirical refits and saved predictions are compared at 1e-10 eV. Rounded manuscript statistics use their displayed precision. The output records observed values, expected values, tolerances, input hashes, versions and elapsed time.

## Five-attribute Ridge preprocessing

The original cross-validation calculation standardizes attributes within each training fold before quadratic expansion. The original held-out calculation constructs quadratic terms in native units. Its evaluation attributes were stored as float32 and promoted to float64 before expansion. Both use Ridge alpha 0.1. These execution paths reproduce their respective published values: cross-validated R² 0.8938 and held-out R² 0.8927 / MAE 0.1554 eV. Removing the d-band attribute gives cross-validated R² 0.8585.

## Empirical fitting

The default run uses the published two descriptor components and recorded penalty choices. Each outer model is fitted using only its training partition; composition-group separation is checked. Newly fitted coefficients are saved in `D01_refitted_fold_models.json`. The full-data fit is reconstructed and evaluated on the 110 supplementary records. The median baseline uses fractional inclusion of tied scores at the top-22 threshold.

## Endpoint definitions

Training free energies use minimum clean and minimum H energies independently within each physical ID. Electronic responses retain explicit paired clean/H identifiers. Fixed-geometry contrasts use energy excluding the entropy term. See the [data dictionary](DATA_DICTIONARY.md) for units and signs.

## Runtime artifacts

`report.json` records completion and `manuscript_checks.csv` contains each check. CSV outputs use stable ordering and 12 significant digits. Timing and platform metadata vary between machines. Existing result directories are protected from overwrite; choose a new `--output` path for another run.
