# Data dictionary

## Identities and datasets

`M1` is the adsorption center, `M2` the neighbor, and `M3`/`M4` the remote metals. A physical ID joins M1, M2 and the lexicographically sorted remote pair with hyphens. All four elements differ. The 17 elements define 57,120 ordered arrangements and 28,560 physical IDs.

`training_records_1206.csv` retains final energy records before deduplication. `record_id` identifies a row in its cohort: `initial` (500), `adaptive` (98), or `reference` (608). The generated `training_995_rebuilt.csv` identifies the source records furnishing each composition's minimum clean and H energies.

The adaptive inventory includes all 100 selections. Two have blank energies and `result_status=no_valid_result`; 98 supply training labels. `ordered_space_index` identifies the selected ordering. The initial 500-list identifies the cluster, physics and random selection strategies.

The eight-repeat retrospective comparison uses the 995 compositions separately. Seed 42+r assigns the first 200 permuted rows to the initial set and 795 to the candidate pool. The default run exports those memberships; fresh acquisition training exports every newly selected batch.

The 110 held-out physical IDs do not overlap training. The 60 DFT validations contain 40 prespecified and 20 additional candidates; two share training IDs. Output membership columns identify overlaps and the all-3d top-30 subset.

## Units and signs

| Quantity | Definition |
|---|---|
| `E_clean`, `E_H` | eV; final energies used for adsorption labels |
| `delta_G_H` | eV; `E_H - E_clean + 3.481`, including H2 and thermodynamic corrections |
| `pred_dG`, `prediction` | eV; predicted adsorption free energy |
| `epistemic` / `epistemic_std` in tables | eV; standard deviation of member means. The ensemble API's `prediction.epistemic` is variance in eV² |
| `aleatoric` in model API | eV²; mean predicted variance across members |
| `total_std` in model API | eV; square root of summed aleatoric and epistemic variances |
| `q_clean_e`, `q_H_e` | e; positive Bader charge relative to valence electrons; positive H-minus-clean change means electron loss |
| `M1_d_e_*`, `M1_dz2_e_*` | electrons; projected clean/H populations |
| `M1_moment_abs_muB_*` | Bohr magnetons; absolute local Co moment |
| `ICOHP_eV` | eV; signed bond population; negative means bonding |
| `B_eV` | eV; summed H-minus-clean ICOHP change over four Co–N bonds and both spins |
| `r_clean_A`, `r_H_A` | Å; paired Co–N distances |
| `without_entropy_eV` | eV; final total energy excluding the entropy term |
| `J_meV` | meV; 1000 times Fe-minus-Cr difference in `E_H - E_clean` at fixed geometry |
| `D_r`, `D_e` | Dimensionless; radii in pm are divided by 100 pm; d counts are tabulated elemental counts |

`standard` and `pv4p` identify the two orbital-projection bases. Bond rows distinguish atoms, periodic cells, orbitals and spins; total and resolved rows are aggregated separately. `source`, `clean_key` and `H_key` distinguish repeated electronic evaluations of one composition. There are 129 pairs for 107 physical IDs.

The elemental property file supplies model inputs. The feature dictionary maps 96 descriptors to the 87-feature input, in blocks of 16, 13, 6 and 52. Processed PDOS tables contain energy grids and spin/orbital projections used in the manuscript.

`provenance/table_index.csv` lists every CSV's row count and columns. `code/input_manifest.json` identifies scientific inputs and models by SHA-256 and byte count.
