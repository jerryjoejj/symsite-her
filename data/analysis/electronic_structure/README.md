# Electronic structure and bonding

The tables describe clean-to-H responses at metal sites and within the Co–N₄ coordination unit. Definitions and calculation methods are given in [SI Section S1.12](../../../docs/methods/supporting_information_EN.md#s1-12).

## Data navigation

| Directory | Contents |
|---|---|
| [paired_responses](paired_responses/) | Integrated clean/H responses and composition/batch membership |
| [coordination_response](coordination_response/) | Atom and coordination-unit charge responses |
| [charge_cancellation](charge_cancellation/) | Net charge response, atomic redistribution and bond changes |
| [geometry_bonding](geometry_bonding/) | Co–N distances, orbital-resolved bonding and spectral integrals |
| [fixed_geometry](fixed_geometry/) | Cross-geometry electronic-energy comparisons |
| [neighbor_interaction](neighbor_interaction/) | Cu/Mn comparisons at common fixed geometry |
| [restart_sensitivity](restart_sensitivity/) | Energy and magnetic response to endpoint restart |
| [noncollinear_sensitivity](noncollinear_sensitivity/) | Noncollinear energy, spin-vector and occupation comparisons |
| [copper_magnetism](copper_magnetism/) | Cu-background spin and orbital populations |
| [coordination_controls](coordination_controls/) | Ni comparison and charge-balance candidate |
| [model_link](model_link/), [data_coverage](data_coverage/) | Composition predictions, electronic records and sample coverage |

## Quantities and results

The paired orbital table contains 129 clean/H pairs for 107 Co compositions. Each record retains its calculation batch: 128 pairs show lower Co $d_{z^2}$ projected occupation after adsorption, 127 show higher total d occupation, and 127 show smaller absolute Co local moments. Multiple batches can represent the same composition.

Charge response is H minus clean. Bader net charge is valence-electron count minus Bader electron population. The signed Co–N₄ sum measures net charge response; the sum of absolute atomic changes measures redistribution within the unit.

$B$ is the sum of H-minus-clean ICOHP over four Co–N bonds, in eV. Positive $B$ denotes bond weakening. Orbital terms are resolved in the specified LOBSTER basis. Cu and Mn neighbors give different total responses to remote Cr→Fe replacement; s/p and d contributions are tabulated separately.

$J$ is the Fe-minus-Cr adsorption electronic-energy response at fixed geometry, calculated from energies without entropy. $I=J_{\mathrm{Mn}}-J_{\mathrm{Cu}}$ compares neighboring environments. The endpoint and sensitivity results are:

| Comparison | Result |
|---|---|
| Original common-geometry endpoints | $J_{\mathrm{Cu}}=+54.55077$ meV, $J_{\mathrm{Mn}}=-126.41696$ meV, $I=-180.96773$ meV |
| MnFe clean restart | Energy decreases by 168.80494 meV; the Co moment changes from −0.531 to +0.703 μB |
| Substitution of that restarted endpoint | $J_{\mathrm{Mn}}=+42.38798$ meV and $I=-12.16279$ meV, with the other endpoints fixed |
| Noncollinear comparison | Further energy decrease of 1.37447 meV exceeds the specified 1 meV tolerance; vector/density checks pass, while XML electron-count and highest-eight-band occupation checks are indeterminate at the recorded precision |
| Ni bonding comparison | Clean-state branch compatibility fails; the returned branch gives $B_{\mathrm{Ni}}=0.26202$ eV versus 0.44179 eV for Cr and 0.35268 eV for Fe, opposite to the proposed ordering |

The original, restarted and noncollinear endpoints have distinct protocol identifiers. SI Section S1.12 describes their state comparisons and the Ni branch assignment.

The [paired electronic table](paired_responses/paired_electronic_changes.csv), [Co–N bonding table](geometry_bonding/analysis/bonding_summary.csv) and [orbital-channel contrasts](geometry_bonding/orbital_analysis/Fe_Cr_channel_contrasts.csv) provide the numerical results. File hashes are recorded in the [manifest](../../../provenance/file_manifest.csv).
