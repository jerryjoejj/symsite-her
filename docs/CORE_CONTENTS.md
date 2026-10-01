# Core content and entry points

This guide locates code and data by scientific topic. File origins, public paths and hashes are listed in [file_manifest.csv](../provenance/file_manifest.csv).

| Topic | Main code | Data and documentation |
|---|---|---|
| A. Compositions and features | [Composition space](../src/herald/phase_space/), [features](../src/herald/features/), [space generation](../scripts/modeling/generate_phase_space_v2.py) | [Element properties](../data/elements/element_properties.json), [feature dictionary](../data/processed/features/feature_dictionary.csv), [87D features](../data/processed/features/training_87d.csv) |
| B. Labels and dataset membership | [Physical IDs](../src/herald/phase_space/generator.py), [membership preparation](../scripts/data_preparation/prepare_validation_membership.py) | [Training data](../data/processed/training/consolidated_ground_state.csv), [held-out predictions](../data/processed/architecture/blindtest_v2_predictions_v3.csv), [60 validation records](../data/processed/dft_validation/validation_membership.csv), [cohort results](../data/processed/cohorts/) |
| C. Models and training | [V2 model](../src/herald/models/symmetry_aware_ensemble_v2.py), [training and screening](../scripts/modeling/screen_fullspace_v2.py), [prediction example](../examples/predict.py) | [Production weights](../models/ensemble/symmetry_ensemble_v2_production_v3.pt), [configuration](../models/ensemble/symmetry_ensemble_v2_production_v3_meta.json) |
| D. Comparisons, statistics and acquisition | [2×2 comparison](../scripts/modeling/ablation_strict_2x2.py), [model comparison](../scripts/modeling/model_comparison_v2.py), [chemical prior](../src/herald/acquisition/physics_guided.py), [retrospective acquisition](../scripts/modeling/ablation_acquisition.py) | [Model comparison data](../data/processed/architecture/), [acquisition data](../data/processed/acquisition/), [sampling module](../src/herald/sampling/initial.py) |
| E. Full-space screening and validation | [Screening rules](../scripts/modeling/robust_shortlist_v3.py), [validation membership](../scripts/data_preparation/prepare_validation_membership.py) | [Full-space predictions](../data/processed/screening/activity_map_28560_v3.csv), [137 candidates](../data/processed/screening/SI_table_cheap_candidates.csv), [validation membership](../data/processed/dft_validation/validation_membership.csv) |
| F. Element properties, interpretation and electronic structure | [SHAP comparison](../scripts/modeling/shap_three_way_comparison.py) | [Interpretation tables](../data/processed/design_principle/), [electronic structure tables](../data/processed/electronic_structure/), [SHAP calculation metadata](../data/processed/design_principle/shap_explainer_audit.json) |
| G. Empirical expressions and bonding | [Fixed-expression evaluation](../scripts/mechanistic_analysis/candidate_predictor.py), [site-role comparisons and result summary](../scripts/mechanistic_analysis/site_role_controls.py) | [Coefficients](../data/analysis/descriptors/expression_validation/final_coefficients.csv), [grouped splits](../data/analysis/descriptors/validation_protocol/splits.json), [bonding data guide](../data/analysis/electronic_structure/README.md), [electronic-state results](../data/analysis/electronic_structure/) |
| H. Environment, documentation and verification | [Data preparation](../scripts/data_preparation/), [verification tool](../tools/verify_bundle.py) | [Data dictionary](DATA_DICTIONARY.md), [table index](../provenance/table_index.csv), [environment setup](../environment/README.md) |

## Methods and data definitions

- [Methods](methods/methods_EN.md).
- [Supporting Information](methods/supporting_information_EN.md): computational methods, sample definitions, equations and supplementary results.
- [Bonding analysis](methods/bonding_analysis_EN.md).
- [D01 definition and worked example](../data/analysis/descriptors/expression_definition/section_detailed_EN.md): the final expression is `All_De1_original`; calculations use its full-precision coefficients.

See the [usage guide](REPRODUCIBILITY.md) for working with result tables and models.
