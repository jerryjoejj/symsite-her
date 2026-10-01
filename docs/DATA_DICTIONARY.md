# Data guide

[table_index.csv](../provenance/table_index.csv) lists the CSV tables, record counts and column names.

## Main conventions

| Field or quantity | Meaning |
|---|---|
| `M1`, `M2`, `M3`, `M4` | Adsorption center, neighbor and two remote metal sites |
| `phys_id`, `physical_id` | M1/M2 identity followed by sorted M3/M4, subject to each archived table's text serialization |
| `E_clean`, `E_H` | Retained clean/H total-energy labels in eV; consult the specific protocol for energy definition |
| `delta_G_H`, `dft` | Signed hydrogen adsorption free energy in eV, for the corresponding dataset |
| `pred_dG`, `pred`, `dG_SymEns` | Signed model prediction in eV; distinguish production data from historical rounded SI tables |
| `abs_dG` | Absolute predicted free energy in the screening tables |
| `epistemic`, `epi` in current v3 screening/held-out CSVs | Epistemic standard deviation in eV; the Python model object's `.epistemic` field instead contains variance |
| `seen_training`, `in_blind110` | Training and held-out membership flags |
| `pre_specified`, `cohort`, `in_top30` | Selection/evaluation cohort flags |
| `DFT_hit_0p1` | Archived indicator for absolute DFT free energy at most 0.1 eV |
| `source`, `protocol`, `source_family`, `mesh` | Calculation batch, protocol, source group and reciprocal-space mesh |
| ICOHP | Integrated crystal orbital Hamilton population; negative denotes net bonding in the retained convention |
| B | Sum of H-minus-clean Co–N ICOHP changes in a composition, in eV |
| J | Fe-minus-Cr adsorption electronic-energy difference at fixed reference geometry; retained records specify energy without entropy |

NaN and empty cells denote unavailable values. Technical-completion and electronic-state comparison results are recorded in separate fields.

## Dataset membership and energy labels

- The 995 training compositions and 110 held-out compositions have disjoint physical IDs.
- The 60 candidate-validation records include two training compositions and one held-out composition. `validation_membership.csv` identifies each cohort and intersection.
- The expanded electronic dataset contains 129 clean/H record pairs for 107 compositions, with batch identifiers distinguishing repeated calculations.
- Training labels use the lowest clean and H energies selected separately within each physical ID, followed by the +3.481 eV conversion to hydrogen adsorption free energy.

## Key files

- `data/processed/features/training_87d.csv`: 995 records, six identity/label fields and 87 features. The feature dictionary identifies each column; the input groups are M1/M2/M34/global with 16/13/6/52 features, respectively.
- `data/processed/dft_validation/validation_membership.csv`: all 60 observations and their intersecting cohorts. Use this file to select the evaluation denominator.
- `data/processed/screening/activity_map_28560_v3.csv`: current full-space predictions. Earlier comparison files retain their model-version identifiers.
- `data/analysis/electronic_structure/`: paired electronic responses, Co–N bonding and state comparisons; see the [electronic-structure guide](../data/analysis/electronic_structure/README.md).
- `data/processed/electronic_structure/orbital_pdos.csv`: 12,006 energy-grid records for six clean/H electronic states, with Fermi levels, orbital occupations and spin-resolved PDOS.
- `data/analysis/descriptors/expression_validation/final_coefficients.csv`: use `All_De1_original` for the frozen D01 expression. Its full-precision coefficients multiply the stated inputs directly.

For formulas, cohort definitions and electronic-state comparisons, see Methods, SI Sections S1.10–S1.12, and [core content map](CORE_CONTENTS.md).
## Reading CSV files

The table index counts nonempty records read with Python's CSV reader. Two tables use the following text conventions:

- `data/processed/design_principle/dband_collinearity_partial.csv`: four metric labels contain unquoted commas, so those records have four fields under a three-column header. Treat the final two fields as value and p-value and the preceding text as the metric label.
- `data/processed/screening/m1_m2_hitrate.csv`: the final line begins with `# Co_row_chi2_test` and contains summary metadata. Use a comment-aware reader or omit that footer when reading the 17-by-17 matrix.

Rows with differing field counts are identified by `unequal_width_rows` in `provenance/table_index.csv`.
