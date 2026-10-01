# Terminology and notation

| Quantity or concept | Symbol or name | Definition |
|---|---|---|
| Material class | Multimetallic single-atom catalysts | Multimetallic single-atom catalysts on N-doped graphene |
| Study platform | $(M_1,M_2,M_3,M_4)$–N₄/G | Four-metal single-atom model on N-doped graphene; G denotes graphene |
| Generic coordination platform | M–N₄/G | Used when the elements at the four sites are unspecified |
| Specific composition | (Co,Cu,Fe,Zn)–N₄/G; abbreviated Co–Cu–Fe–Zn | Elements are listed in $M_1$, $M_2$, $M_3$, $M_4$ order; hyphenated IDs in the data retain this order |
| Site roles | $M_1$; $M_2$; $\{M_3,M_4\}$ | Adsorption center; neighbor; unordered pair of remote sites |
| Remote exchange | $M_3\leftrightarrow M_4$ | Site-exchange operation associated with approximate adsorption-energy symmetry |
| Remote feature block | $M_{34}$ | Means and absolute differences of electronegativity, d-electron count and atomic radius; six features in total |
| Physical ID | $(M_1,M_2,\{M_3,M_4\})$ | The first two sites are fixed and the remote pair is unordered; remote elements are sorted alphabetically when serialized. Deduplicated labels use the lowest clean-state and H-state energies separately for each ID |
| Hydrogen adsorption free energy | $\Delta G_{\mathrm{H}}$ | H is upright; screening uses the absolute value. A hat denotes an empirical prediction |
| Exchange deviation | $\Delta E_{ab}$; $\Delta E_{34}$ | Absolute difference in hydrogen adsorption free energy before and after exchange; nonnegative |
| Atomic properties | $r_i$, $\chi_i$, $n_{d,i}$ | Tabulated radius, electronegativity and d-electron count at site $i$; d-electron count differs from total valence-electron count |
| Nine-valence-electron metals | Co, Rh, Ir | Elements with a valence-electron count of nine |
| Ensemble member | Constituent model | An independently trained neural network in the ensemble |
| Aleatoric and epistemic variance | $\sigma_{\mathrm{ale}}^2$; $\sigma_{\mathrm{epi}}^2$ | Mean of member predictive variances; population variance of member predictive means |
| Screening uncertainty | $\sigma_{\mathrm{epi}}$ | Epistemic standard deviation, in eV |
| Acquisition uncertainty | $U=\sqrt{\sigma_{\mathrm{ale}}^2+\sigma_{\mathrm{epi}}^2}$; $U_{\mathrm{norm}}$ | Total predictive standard deviation and its min–max normalization over the remaining pool in the current round; the acquisition score uses the normalized value |
| Chemical prior | $P_{\mathrm{phys}}=0.6\pi_{n_d}+0.4\pi_\chi$ | Components based on d-electron count and electronegativity, respectively |
| Electronegativity Gaussian component | $\bar{\chi}$, $\sigma_\chi$ | Mean and standard deviation of adsorption-center electronegativity among hits in the calibration set |
| Acquisition strategies | Physics-guided; pure-UCB | UCB names the comparison baseline |
| Orbital and Fermi energy | $d_{z^2}$; $E_{\mathrm{F}}$ | Orbital label and Fermi energy |
| Empirical expression | $D_r$, $D_e$ | Size and electronic components; D01 is the model identifier |
| Bader net charge | $q_i=Z_i^{\mathrm{val}}-N_i^{\mathrm{Bader}}$; $\Delta q_i=q_i^{\mathrm{H}}-q_i^{\mathrm{clean}}$ | Valence-electron count minus Bader electron population; the response is the adsorbed state minus the clean state |
| Bonding response | $B$ | Sum of H-state minus clean-state ICOHP changes for the four Co–N bonds, in eV |
| Fixed-geometry response | $J$; $G_A$, $G_B$ | Fe-minus-Cr contrast in adsorption electronic energy; two reference geometries. This quantity differs from $\Delta G_{\mathrm{H}}$ |
| Capacity-balance candidate | $Q_{\mathrm{bal}}$ | Candidate quantity defined in the SI; electronic-branch comparisons are given in SI S1.12 |

## Abbreviations

| Abbreviation | Full name |
|---|---|
| HER | hydrogen evolution reaction |
| SAC / SACs | single-atom catalyst / catalysts |
| ML | machine learning |
| DFT | density functional theory |
| PAW | projector augmented-wave |
| PBE | Perdew–Burke–Ernzerhof |
| GGA | generalized gradient approximation |
| CHE | computational hydrogen electrode |
| MLP | multilayer perceptron |
| NLL | negative log-likelihood |
| MSE | mean squared error |
| MAE | mean absolute error |
| RMSE | root mean squared error |
| CV | cross-validation |
| SD | standard deviation |
| OOD | out-of-distribution |
| UCB | upper confidence bound |
| SHAP | SHapley Additive exPlanations |
| DOS / PDOS | density of states / projected density of states |
| ICOHP | integrated crystal orbital Hamilton population |
| SCF | self-consistent field |
| SOC | spin–orbit coupling |
| SI | Supporting Information |

## Units and supplementary tables

Energy is reported in eV or meV, length in Å or pm, charge in e and magnetic moment in $\mu_{\mathrm{B}}$.

Supplementary notes are numbered S1.1–S1.12 and supplementary tables Table S1–S15. Table S5 and Table S12 each contain parts (a) and (b).
