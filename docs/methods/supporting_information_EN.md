# Supporting Information

Last modified: 2026-09-25

<!-- BEGIN SI CONTENTS -->
## Contents

- [S1. Supplementary Methods and Analyses](#s1)
  - [S1.1. Material Platform and Metal Compositions](#s1-1)
  - [S1.2. Composition Space and Physical IDs](#s1-2)
  - [S1.3. DFT Verification of Remote-Site Exchange Symmetry](#s1-3)
  - [S1.4. Site Features and Exchange Invariance](#s1-4)
  - [S1.5. Ensemble Models, Training and Baseline Comparisons](#s1-5)
  - [S1.6. DFT Sampling and Chemically Guided Selection](#s1-6)
  - [S1.7. DFT Validation of Candidate Compositions](#s1-7)
  - [S1.8. Electronic-Structure Analysis of Adsorption Centers](#s1-8)
  - [S1.9. SHAP Attribution of Model Features](#s1-9)
  - [S1.10. Software and Data Records](#s1-10)
  - [S1.11. Evaluation Sets and Statistical Interpretation](#s1-11)
  - [S1.12. Site-Resolved Expressions and Coordination Responses](#s1-12)
- [S2. Supplementary Tables](#s2)
  - [Table S1. Predicted Free Energies and Epistemic Uncertainties of 137 All-3d Candidates](#table-s1)
  - [Table S2. Descriptor Dictionary and Construction of the 87-Dimensional Input](#table-s2)
  - [Table S3. 2×2 Ablation of Model Architecture and Loss Function](#table-s3)
  - [Table S4. Five-Fold Cross-Validation of 13 Models](#table-s4)
  - [Table S5. Paired Retrospective Comparison of Chemically Guided Selection](#table-s5)
  - [Table S6. DFT Reassessment of 56 Pairs with Large Exchange Deviations](#table-s6)
  - [Table S7. Independent Validation of 40 Early-Transition-Metal Exchange Pairs](#table-s7)
  - [Table S8. Effects of Descriptor Count and Regression Form on Predictive Performance](#table-s8)
  - [Table S9. DFT Validation of 40 Pre-Specified Candidate Compositions](#table-s9)
  - [Table S10. DFT Validation of 20 Additional Compositions in the All-3d Top-30](#table-s10)
  - [Table S11. Composition Overlap and Training-Unseen Validation Results](#table-s11)
  - [Table S12. Comparison of SHAP Site Attributions across Models](#table-s12)
  - [Table S13. Projected $d_{z^2}$ Occupancies of Three Co-Centered Compositions](#table-s13)
  - [Table S14. Projected $d_{z^2}$ Occupancies of Fe, Ru, Ni, Pd and Pt Centers](#table-s14)
  - [Table S15. Composition-Selection Criteria for Multimetallic M–N₄/G Catalysts](#table-s15)
<!-- END SI CONTENTS -->

<a id="s1"></a>
## S1. Supplementary Methods and Analyses

<a id="s1-1"></a>
### S1.1. Material Platform and Metal Compositions

The catalyst platform is denoted $(M_1,M_2,M_3,M_4)$–N₄/G. The initial clean model contains 48 C atoms, 16 N atoms and four distinct metal atoms, giving 68 atoms; the H-adsorbed state contains 69. Its lattice parameters are $a=b=14.76$ Å and $\gamma=120^\circ$, with a cell length of 20 Å normal to the substrate. The metals occupy four N₄ coordination sites, and H is initialized atop $M_1$. Local geometries are determined by relaxation for each composition.

$M_1$ denotes the adsorption site examined, while $M_2$–$M_4$ describe its metal environment. Metal occupation defines the local coordination environment; site effects are examined through exchange comparisons and electronic-structure analysis. Approximate $M_3\leftrightarrow M_4$ energy-permutation symmetry is tested through adsorption-energy differences between swapped pairs. Electronic-structure comparisons use the compositions listed in Figure 7 and Tables S13 and S14.

The 17 metals were chosen to cover the d-electron counts d¹ through d¹⁰, which spans the d-band center range needed for the Sabatier search, and to include noble-metal benchmarks alongside earth-abundant candidates. The pool holds the 3d metals Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, and Zn, the 4d metals Ru, Rh, Pd, and Ag, and the 5d metals Ir, Pt, and Au. Radioactive and rare-earth elements are excluded. Scandium supplies the d¹ endpoint of the series.

---


DFT, density functional theory; HER, hydrogen evolution reaction; SACs, single-atom catalysts; ML, machine learning; MLP, multilayer perceptron; NLL, negative log-likelihood; MSE, mean squared error; MAE, mean absolute error; RMSE, root mean squared error; CV, cross-validation; SD, standard deviation; OOD, out-of-distribution; UCB, upper confidence bound; SHAP, SHapley Additive exPlanations; DOS/PDOS, density of states/projected density of states; ICOHP, integrated crystal orbital Hamilton population; SCF, self-consistent field; SOC, spin–orbit coupling. Symbols are defined in the relevant notes: $M_3\leftrightarrow M_4$ denotes the exchange, $\{M_3,M_4\}$ the unordered remote pair, and $M_{34}$ the six-dimensional remote feature block.

<a id="s1-2"></a>
### S1.2. Composition Space and Physical IDs

The 17 metals arranged over four distinct sites without repetition give P(17,4) = 17 × 16 × 15 × 14 = 57120 ordered ($M_1$, $M_2$, $M_3$, $M_4$) structures. Arrangements are grouped by the approximate $M_3\leftrightarrow M_4$ permutation relation in S1.3. Each physical ID is defined as $(M_1,M_2,\{M_3,M_4\})$ and stored in its lexicographically smaller ordering $M_3$ ≤ $M_4$. The median deviation of 5.6 meV across 124 exchange pairs characterizes the energy scale of this grouping approximation. This encoding represents the space as 28560 physical IDs, equivalently C(17,4) × 4!/2 = 2380 × 12, or 17 × 16 × C(15,2) = 17 × 16 × 105. The deduplicated space contains no $M_3\leftrightarrow M_4$ exchange pairs by construction, and a hash check of the feature matrix confirms zero feature collisions, where the legacy 57120-space encoding produced 89 collisions and reduced the distinct feature vectors to 9520.

The 124 training exchange pairs characterize the composition dependence of approximate symmetry. Their archived DFT results give a median swap deviation $\Delta E_{34}$ of 5.6 meV, with 54.8% of pairs below 10 meV, 16.9% between 10 and 50 meV, and 28.2% above 50 meV. Element medians range from 0.81 meV for Ir and 0.99 meV for Pt to 24.6 meV for Au and 84.9 meV for Fe, so breaking is element-specific rather than diffuse. The nine-valence-electron triplets Co, Rh, and Ir, which dominate the near-thermoneutral region in Figure 5, have a median of 1.81 meV with 70.8% of pairs below 10 meV. The 15 pairs whose smaller absolute adsorption free energy is at most 0.1 eV have median swap deviation 3.4 meV, compared with 6.7 meV for the other 109. A Mann–Whitney test detects no difference between the two swap-deviation distributions (p = 0.99), assessing their relation to the near-thermoneutral window. Relaxed structures belong to space group P1 by automated space-group analysis at 0.01 Å tolerance, so the interchangeability is an approximate energy symmetry rather than a crystallographic equivalence; the space size 28560 is a pure combinatorial count and does not depend on whether the symmetry holds.

---

<a id="s1-3"></a>
### S1.3. DFT Verification of Remote-Site Exchange Symmetry

Two dedicated DFT campaigns map the swap symmetry across the compositional space. Campaign 1 (56 pairs, 112 structures, 224 single points) targeted the historical outliers, the 124 training exchange pairs whose initial DFT deviation exceeded 10 meV, distributed over the deviation bins 10–50 meV (21 pairs), 50–100 meV (8 pairs), 100–200 meV (22 pairs), and above 200 meV (5 pairs). These were recalculated with the same INCAR settings as the regular batches, EDIFFG = −0.02 eV/Å and EDIFF = 10⁻⁵ eV, so the recalculations retained the same nominal stopping criteria. Deviations remained above the comparison threshold in 84% of pairs, 47 of 56. The persistent deviations identify compositions sensitive to remote exchange; electronic-state selection may contribute to this response. The breaking is element-specific and concentrated in structures with Fe or Au at $M_1$, the median deviation for Fe-$M_1$ pairs dropping from 145 meV historically to 101 meV after recalculation, a 31% improvement that still leaves 22 of 26 pairs above the 10 meV threshold. In contrast, the late transition metals converge well, Ir, Pd, and Pt at $M_1$ showing convergence rates of 100%, 67%, and 33% respectively, for the pairs sampled in this campaign.

Campaign 2 (40 pairs, 80 structures, 160 single points) filled the verification blind spot, the early transition metals Sc, Ti, V, Cr, and Mn at $M_1$, which occupy about 29% of the 28560 space but have no exchange pairs in the training set. This campaign drew 40 unbiased pairs, 8 per element, from the full phase space and used the same computational settings. The result is that 72.5% are symmetric, 29 of 40 below 10 meV, with a median deviation of 2.9 meV, supporting approximate symmetry for most sampled early-TM pairs. Element-level breakdown gives Ti 7 of 8 with a median of 0.8 meV, Cr 7 of 8 with 1.6 meV, Mn 6 of 8 with 3.8 meV, V 5 of 8 with 6.4 meV, and Sc 4 of 8 with 10.6 meV, the weakest, with half of the sampled pairs below the threshold. The 11 breaking pairs of campaign 2 are enriched in Fe and Au, 6 of 11 containing Fe or Au against 8 of 29 in the symmetric subset, showing an association between larger swap deviations and Fe/Au at auxiliary sites $M_2$–$M_4$.

The three-zone symmetry map follows (Tables S6 and S7, Figure S6). The post-transition-metal zone, Co, Ni, Ru, Rh, Pd, Ir, Pt, Cu, Ag, and Zn at $M_1$, covers 59% of the space at a median below 3 meV, supporting approximate grouping. The early-transition-metal zone, Sc, Ti, V, Cr, and Mn at $M_1$, covers 29% at a measured median of 2.9 meV, again supporting an approximate compression. The Fe/Au zone covers 12% at a median of 25–85 meV, where the compression loses information, but these elements rarely produce high-activity candidates, Figures 5 and 7.

The 28560 physical IDs follow from counting unordered remote-site pairs. Swap deviations quantify the accuracy of this grouping approximation. Models with the shared invariant input assign the same prediction to both remote orderings, and model comparisons use this common representation.

---

<a id="s1-4"></a>
### S1.4. Site Features and Exchange Invariance

Each ($M_1$, $M_2$, $M_3$, $M_4$) structure is represented by 96 descriptors, an $M_1$ block of 16 columns, 13 base atomic properties plus the derived H-bond polarity, d-band filling, and work-function deviation, an $M_2$ block of 13 base properties, a 6-column $M_{34}$ symmetric block, 52 global statistics, and 9 derived columns that the symmetric input drops (Table S2). The 13 base properties are atomic number, atomic weight, atomic radius, Pauling electronegativity, d-electron count, valence-electron count, d-band center, d-band width, work function, surface energy, lattice constant, melting point, and cohesive energy. The global block holds four statistics per property over the unordered set {$M_1$, $M_2$, $M_3$, $M_4$}, the weighted mean with $M_1$ weight 0.5, the standard deviation, the range, and the $M_1$ deviation.

H adsorption assigns distinct roles to $M_1$ and its neighboring site $M_2$. In the analyzed structure, the adsorption-induced Bader charge change is about −0.04 |e| at $M_2$ and below 0.01 |e| at the remote $M_3$ and $M_4$ sites, supporting separate representations of the neighboring and remote sites. The $M_{34}$ block holds the mean and the absolute difference of electronegativity, d-electron count, and atomic radius over the distal pair, invariant to $M_3\leftrightarrow M_4$ swap by construction. The feature dictionary evolved from the 77-dimensional Phase-D set used for initial sampling, which pooled $M_2$, $M_3$, and $M_4$ into aggregate statistics and masked $M_2$'s distinct role, to the present 96-dimensional Phase-E set, which adds the 13-column $M_2$ block and the 6-column $M_{34}$ block. This refinement eliminated the 89 feature collisions of the legacy encoding, 57120 structures compressing to only 9520 distinct vectors, and lowered the irreducible model error floor from 0.028 eV to about 0.003 eV.

For every compared model, the 96D vector is transformed into an 87D symmetric input that satisfies x($M_1$, $M_2$, $M_3$, $M_4$) = x($M_1$, $M_2$, $M_4$, $M_3$) by construction. The transformation groups the features into the 16D $M_1$ block, the 13D $M_2$ block, the 6D $M_{34}$ symmetric block, and the 52D global block, 16 + 13 + 6 + 52 = 87 dimensions in total. The remaining 9 of the 96 columns are derived descriptors, property ratios, size mismatch, strain, and dipole proxies, discussed in S1.9, and are dropped from the symmetric input.

Theorem 1 (Exact scalar invariance). Let $f_\theta: \mathbb{R}^{87} \to \mathbb{R}$ be any function mapping symmetric features to a scalar prediction, for example a neural network. Let $\tau$ denote the swap operation $(M_1, M_2, M_3, M_4) \mapsto (M_1, M_2, M_4, M_3)$, and let $\mathbf{x}(\cdot)$ be the 87D symmetric feature transform. Then for all structures $s$ and all parameter vectors $\theta$,

$$f_\theta(\mathbf{x}(s)) = f_\theta(\mathbf{x}(\tau s)).$$

*Proof.* The construction of $\mathbf{x}$ gives $\mathbf{x}(\tau s) = \mathbf{x}(s)$, since the $M_{34}$ block uses the mean and the absolute difference of the shared properties, both invariant under the $M_3\leftrightarrow M_4$ swap, and the global block aggregates over an unordered set. Therefore $f_\theta(\mathbf{x}(\tau s)) = f_\theta(\mathbf{x}(s))$ for any $f_\theta$, independent of $\theta$. $\square$

The shared input guarantees scalar invariance for flat MLPs, trees and site-role models. The ablation compares the combined effects of site grouping and network architecture on identical inputs; parameter counts also differ (Table S3).

Features were z-score normalized with statistics computed on the training partition only. Within each five-fold cross-validation trial, the training partition of 796 samples and the validation partition of 199 samples were normalized with the training mean and standard deviation, and the full-space prediction used statistics from a model retrained on all 995 samples, so test and prediction data do not enter the normalization statistics.

---

<a id="s1-5"></a>
### S1.5. Ensemble Models, Training and Baseline Comparisons

The ensemble comprises 10 independent MLPs. Each constituent model takes the 87D symmetric input, passes the three site blocks through independent encoders, $M_1$, $M_2$, and $M_{34}$, each producing a 32-dimensional embedding, concatenates the three embeddings with the 52D global statistics into a 148D vector, and processes it through a shared fusion trunk with two hidden layers [128, 64], ReLU activations, and dropout 0.1. Each model has a heteroscedastic dual head predicting the mean μ and log-variance log σ² of $\Delta G_{\mathrm{H}}$, trained with the Gaussian negative log-likelihood loss in Figure 3.

Training used Adam at learning rate 10⁻³ with weight decay 10⁻⁵, batch size 64, up to 300 epochs, early stopping on validation loss at patience 30, and bootstrap resampling with different random seeds per model, 42, 123, 456, and so on through 999. Cross-validation used KFold with shuffle and random seed 42. Each model contains 32418 parameters, verified against the trained weights, versus 64706 for the flat MLP baseline, a 49.9% reduction.

The ensemble prediction is the arithmetic mean of the 10 constituent-model predictions, the aleatoric variance is the mean of the model variances, and the epistemic uncertainty is the population standard deviation of the model means, the quantity used for full-space uncertainty filtering.

The flat-MLP baseline (ablation control) has 3 hidden layers [256, 128, 64], receives the same 87D input, and contains 64706 parameters per model, so the comparison fixes the features while changing site grouping and network capacity. The XGBoost baseline uses 300 trees, maximum depth 7, learning rate 0.04, subsample 0.8, column subsample 0.8, and minimum child weight 5, tuned on the 87D features under the shared five-fold protocol; it provides no uncertainty estimates and serves only for point-prediction benchmarking (Table S4, Figure S3). Ten further traditional machine-learning baselines were trained and evaluated under the same unified protocol, 87D input and shared five-fold CV with fold-internal standardization. Two anomalies were diagnosed and fixed by retraining, LightGBM, whose initial hyperparameter grid was over-regularized, improving from 0.2596 to 0.1082 eV on a 576-combination grid, and GPR, initially subsampled to 300 structures, improving from 0.2278 to 0.1385 eV on the full 995 samples. The symmetry-aware ensemble ranks third among 13 archived five-fold mean MAEs, behind XGBoost (0.1058 eV) and LightGBM (0.1082 eV), with its own mean 0.1095 eV. Fold standard deviations are archived for only three models; missing deviations for the other ten are not imputed (SI Figure S3; Table S4).

The 110 held-out structures comprise an initial 30 selected as 12 top-ranked and high-confidence, 6 promising but uncertain, 6 boundary and 6 blind-spot candidates, followed by 80 recorded as 40 in-distribution and 40 out-of-distribution structures (SI Figure S4). Physical ID matching against the 995 training compositions gives zero overlap. Out-of-distribution evaluation uses the 40 structures explicitly assigned to that stratum. On these 110 structures, symmetry-aware and flat MAEs are 0.1095 and 0.1302 eV, with structure-paired t-test p = 0.0012. Mean symmetry-aware-minus-flat absolute-error differences are −0.01315 eV for the first 30, −0.00470 eV for the later 40 in-distribution and −0.04224 eV for the later 40 recorded out-of-distribution structures. SI Figure S2 assesses error ranking in the production prediction run.

---

<a id="s1-6"></a>
### S1.6. DFT Sampling and Chemically Guided Selection

DFT sampling comprised initial mixed sampling and one adaptive acquisition round, followed by candidate and exchange-symmetry validation.

<a id="s1-6-1"></a>
#### S1.6.1. Iteration 0: initial sampling (500 structures)

Initial DFT sampling combined compositional diversity, random selection and a physics prior. The 57120 ordered arrangements were characterized by 77-dimensional Phase-D features, the precursor to the final 96D set, sufficient for cold-start sampling. Cluster-based diversity sampling standardized the feature matrix and ran KMeans with k = 250, random_state = 42, and n_init = 10, selecting the structure closest to each cluster centroid, 250 structures. Physics-informed sampling scored each structure by s = 1/(1 + |εd − (−2.25)|) + 1/(1 + |χ − 2.25|), the Pt(111)-derived targets of the classical Sabatier volcano, and selected the top 150 excluding the cluster picks. Uniform random sampling drew 100 structures from the remainder with NumPy's legacy RandomState at seed 42.

Retrospective analysis after DFT shows cluster sampling reached a 22.0% success rate within $|\Delta G_{\mathrm{H}}|$ ≤ 0.2 eV, 55 of 250, random sampling 21.0%, 21 of 100, but the physics branch only 5.3%, 8 of 150. The Pt-derived d-band prior does not transfer to multimetallic M–N₄ systems, which motivated the later retrospective evaluation of a data-calibrated prior. The iteration-0 selection is reproducible to 85.4% ordered match, 427 of 500, with per-strategy rates of 96.8% for cluster, 84.0% for physics, and 100% for random when the upstream selections are fixed. The residual deviation has two diagnosed root causes, the feature space, the correct 77-dimensional Phase-D set versus a mistaken 96-dimensional set, raising the match from 28.4% to 73.2%, and the random-number generator API, the legacy RandomState versus the newer default_rng, raising it further to 85.4%. The remaining difference stems from KMeans centroid sensitivity to the scikit-learn version. The complete iteration-0 structure list is provided in the supplementary data, so all downstream analyses reproduce exactly from that starting point.

<a id="s1-6-2"></a>
#### S1.6.2. Iteration 1: adaptive sampling, 100 selections and 98 valid labels

Iteration 1 excluded labelled candidates from the original space of 57120 ordered arrangements. The iteration-0 ensemble provided the predicted mean $\mu$ and total predictive standard deviation $U$. The acquisition function was $\mathrm{score}=\alpha U_{\mathrm{norm}}+(1-\alpha)(1-|\mu|_{\mathrm{norm}})$, with both components min-max normalized. The archived report gives exploration and exploitation weights of 0.637 and 0.363. This round selected 100 structures, of which 98 ultimately yielded valid DFT labels. It did not use $P_{\mathrm{phys}}$ or the three-term score schedule; the later retrospective experiment separately evaluates the chemical prior.

<a id="s1-6-3"></a>
#### S1.6.3. Full-space prediction and subsequent validation

After iteration 1, the workflow proceeded to full-space prediction of 28560 compositions, DFT validation of 40 candidates (S1.7) and symmetry verification of 96 exchange pairs (S1.3).

<a id="s1-6-4"></a>
#### S1.6.4. Retrospective pool ablation

The retrospective pool ablation calibrates P_phys at each round using only labels already revealed; unselected labels do not enter calibration. The per-group success rates of the adsorption-site d-electron count, π_nd, reported here are descriptive statistics over the full 995 labelled set, the empirical $|\Delta G_{\mathrm{H}}|$ ≤ 0.2 eV rates by d count being 0.61 at d¹, 0.00 at d², 0.02 at d³, 0.61 at d⁵, 0.12 at d⁶, 0.43 at d⁷, 0.35 at d⁸, 0.00 at d⁹, and 0.02 at d¹⁰, normalized by the maximum 0.61, and π_χ is a Gaussian over the adsorption-site electronegativity of near-thermoneutral structures with mean 1.818 and width 0.306. The d-electron count is distinct from the valence-electron count: Co and Ir carry d⁷ and Rh carries d⁸, all nine valence electrons, so the d⁷ and d⁸ peaks of π_nd jointly define the nine-valence-electron family. The prior favors the d-electron and electronegativity region of Co, Rh, and Ir, differing from the Pt-derived prior used for initial sampling. On the same 995 labels used to calibrate the prior, 146 of the 209 structures at or above the 80th-percentile score satisfy $|\Delta G_{\mathrm{H}}|\leq0.2$ eV (69.9%), compared with 253 of 995 overall (25.4%). The threshold includes tied scores, so this group contains more than exactly 20% of the structures. The Pearson correlation between prior score and the binary hit label is 0.53 ($p\approx10^{-73}$). These statistics describe discrimination within the calibration set. Strategy performance is assessed by the paired retrospective comparison below, with recalibration at each round.

A retrospective acquisition comparison on the 995 labeled structures isolated the contribution of the physics-guided term, with labels withheld until acquired. For each of 8 replicates, the 995 structures were permuted with seed 42 + r, the first 200 forming the shared seed set and the remaining 795 the pool. Two strategies then ran five acquisition rounds of 100 structures, from 200 to 700 labeled in total, the physics-guided strategy with the schedule γ_t = 0.3·e^(−t/3) and the pure-UCB strategy with γ = 0, both using K = 5 ensembles and total predictive standard deviation, with the same α_t and β_t schedules. The ground truth is known for every pool structure, so the number of near-thermoneutral candidates each strategy acquires can be counted exactly at thresholds $|\Delta G_{\mathrm{H}}|$ ≤ 0.2 and ≤ 0.1 eV.

Table S5 and Figure S7 summarize the retrospective comparison. The physics-guided strategy ends with 238.0 candidates within 0.2 eV against 228.0 for pure UCB, paired t(7) = 6.67 with p = 0.0003 and 8 of 8 replicates in favor of physics-guided, and 119.25 against 115.50 within 0.1 eV, t(7) = 4.16 with p = 0.0042. The running sum across rounds of the best $|\Delta G_{\mathrm{H}}|$ found so far is 0.00229 with the rule against 0.00508 without, 55% lower, p = 0.016. The advantage is widest mid-run, 192.6 near-thermoneutral structures at round 3 against 161.5, a 19% relative gain, before the two rankings converge as the pool is exhausted. The retrospective comparison uses the archived weight schedule and total predictive uncertainty from five constituent models to assess ranking efficiency within the 995 labeled structures. Labels remain hidden until selection, and final hit counts use the acquired labels.

---

<a id="s1-7"></a>
### S1.7. DFT Validation of Candidate Compositions

Clean- and H-state relaxations tested the screened candidates. The pre-specified set contains 20 all-3d candidates in group A, selected by the dual-model mean_abs ranking before the final-model update, 10 precious-containing candidates in group B, and 10 structures stratified across the predicted range in group C. Group A is contained in the final 137-member all-3d shortlist. Composition matching with fixed $M_1$, $M_2$ and unordered $M_3$, $M_4$ identifies two group-A compositions already in training, Co–Cr–Mn–Zn and Co–Mn–Ni–Zn. Their DFT results remain in the complete list, while generalization statistics cover the other 38 compositions. Co–Ir–Cr–Ti in group B also occurs in the 110-structure held-out set, which has no overlap with training.

For the 38 pre-specified compositions absent from training, predicted and DFT $\Delta G_{\mathrm{H}}$ give Spearman $\rho = 0.545$ ($p = 3.98\times10^{-4}$), RMSE 0.101 eV and mean prediction-minus-DFT residual −0.0055 eV, with a descriptive 95% Student t interval of [−0.039, +0.028] eV across compositions (Figure 6a; SI Figure S5). This interval includes zero. The full 40 give RMSE 0.098 eV and $\rho = 0.549$; these descriptive values include the two training-seen compositions.

Among the training-unseen candidates, group A has 12 of 18 hits at $|\Delta G_{\mathrm{H}}|\leq0.1$ eV, with mean absolute free energy 0.0775 eV, and group B has 8 of 10 hits, with mean 0.0691 eV. The two-sided Welch comparison gives $t(25.3)=0.431$, $p=0.670$, so this selected sample shows no detected difference between the means (Figure 6b). This comparison concerns the means of the selected candidate samples. Group C has 1 of 10 hits and serves to sample the prediction range. The full A20 gives 14 hits and mean 0.0741 eV; its comparison with B10 gives $p=0.787$. Together the pre-specified cohorts yield 21 of 38 training-unseen hits, or 23 of 40 when all DFT results are retained. The 8.5% hit fraction across the full phase space is a model prediction.

The final-model all-3d top-30 was completed with 20 additional DFT calculations in two batches, of which 12 satisfy $|\Delta G_{\mathrm{H}}|\leq0.1$ eV (SI Table S10). Together with the 10 top-30 members already in group A, all 30 ranks are validated and 20 are hits, with mean absolute free energy 0.072 eV (Figure 6c). Rank 16, Co–Mn–Ni–Zn, is already in training; the other 29 yield 19 hits. The composition with the smallest DFT absolute adsorption free energy ranks 27th by prediction. Across both selection protocols, the descriptive total is 35 of 60 hits; 33 of the 58 compositions absent from training are hits. The pooled counts summarize screening outcomes across all validated compositions.

Candidate evaluation targets hydrogen adsorption thermodynamics under the project PBE protocol. The model omits solvation, the electrochemical double layer, pH, reaction kinetics and mass transport. Functional errors, ranking stability across electronic-structure methods, operating stability, synthesis and experimental HER performance are outside this evaluation.

<a id="s1-8"></a>
### S1.8. Electronic-Structure Analysis of Adsorption Centers

Figure 7 compares adsorption electronic structures for three Co-centered compositions and their elemental controls. The site-resolved expression, expanded Co records and electronic-state comparisons are described in S1.12.

Projected occupations are occupation-weighted sums of PROCAR metal-orbital projections from spin-polarized static calculations. They give electron populations in the specified orbital projections. Full per-state values and flank controls are retained in Tables S13 and S14.

Projected band analysis gives a consistent adsorption response in three Co-centered structures, Co–Cu–Fe–Zn, Co–Ni–Cr–Cu and Co–Cr–V–Zn. Their clean-state $d_{z^2}$ occupations range from 1.55 to 1.74 $|e|$, or 21–24% of the projected d population, and decrease by 0.35–0.54 $|e|$ upon H adsorption (Figure 7c; SI Table S13). This partial occupation and depletion are consistent with participation of Co $d_{z^2}$ in adsorption. This analysis measures the adsorption-induced change in total projected orbital occupation. Ten flank compositions with Fe, Ru, Ni, Pd or Pt at $M_1$ compare orbital filling with adsorption response. Fe and Ru have similar clean-state $d_{z^2}$ populations but bind on opposite sides of thermoneutrality, whereas nearly filled Ni/Pd/Pt sites underbind (Figure S10f; SI Table S14). The orbital comparison covers the Co and Fe/Ru/Ni/Pd/Pt compositions listed above.

The Sc comparison uses seven valid Sc-centered and three Co-centered pairs of clean and H-adsorbed electronic-structure calculations (Figure 7d; SI Figure S10). Occupied d-state centroids in the H-adsorbed structures, referenced to each calculation's own Fermi level, average −3.68 eV for Sc and −2.74 eV for Co; the corresponding projected d populations are 1.04 and 7.41 $|e|$. Sc therefore has finite occupied d weight. Mean Bader electron populations on H are 1.636 and 1.058 $|e|$, mean metal–H distances are 1.866 and 1.447 Å, and metal protrusions from the local N₄ plane are 0.918 and 0.107 Å for Sc and Co, respectively. These differences support stronger ionic character and a more extended adsorption geometry for Sc. Metal electron losses on adsorption average 0.031 and 0.036 $|e|$; a Mann–Whitney comparison detects no difference for seven Sc versus three Co structures ($p=0.833$). Charge, occupation and geometry provide complementary descriptions of the adsorption response. Sc–Mn–Cr–Fe is excluded only from the electronic-structure comparison because its H-state static calculation does not reproduce the relaxed magnetic state; its original DFT adsorption result remains among the eight Sc validation compositions. The Sc/Co electronic-structure batch is separate from the original champion Bader analysis in Figure 6d, which gives a Co electron loss of 0.0343 |e|. In this later batch the champion loss is 0.0361 |e|; the mean across all three Co structures rounds to 0.036 |e|.

<a id="s1-9"></a>
### S1.9. SHAP Attribution of Model Features

SHAP values were computed for three models on one common feature set, the 87-dimensional symmetric input. For the XGBoost surrogate, TreeExplainer gives exact Shapley values without background data. For the two neural networks, the symmetry-aware ensemble and the flat MLP without site-role encoders, the 10-member mean prediction was explained with DeepExplainer, each model wrapped in a CPU forward pass that re-applies the training normalization inside the wrapper, the same set of 100 training structures serving as background, and all 995 samples evaluated. Re-explanation of the fixed neural checkpoints reproduces the archived site shares, with maximum absolute additivity residuals below 4 × 10⁻⁷ eV (SI Figure S9). The fitted XGBoost checkpoint is unavailable, so the tree comparator remains an archived summary. Site attribution sums the mean absolute SHAP values over the $M_1$, 16D, $M_2$, 13D, $M_{34}$, 6D, and global, 52D, feature groups; for all three models this aggregation is by index range and does not depend on feature names.

Control design. The flat MLP is the Flat_NLL cell of the strict 2×2 ablation, sharing with the symmetry-aware ensemble the same 87D symmetric input, the same Gaussian NLL loss, the same 10-member ensemble size, the same random seed 42, and the same 80/20 split, so the architectures, including their parameter capacities, differ while the inputs are held fixed. Their attribution difference is descriptive of that comparison. XGBoost was retrained on the same 87D input so that all three attributions share one feature space; the reduction costs no accuracy, five-fold CV MAE 0.1073 eV on 87D against 0.1075 eV on 96D with R² 0.9402 against 0.9403, and leaves the feature ranking unchanged, M₁_valence_electrons first at mean |SHAP| = 0.53. Removing the nine columns that exist only in the 96-dimensional set, atomic_size_mismatch, gcn_deviation, electron_affinity_M1, lattice_strain_M1, neighbor_H_competition, surface_dipole_moment, electronic_configuration_index, electronegativity_to_d_electrons_ratio, and work_function_to_d_band_ratio, raises the $M_1$ share by 0.90 percentage points, consistent with two of them nominally belonging to $M_1$ yet falling into the global bin under name-based grouping. The 96-dimensional model is retained only as a historical feature-set control, no longer as the screening surrogate, and its attribution is retained in Table S12 for provenance of the Figure 7 numbers. XGBoost still differs from the neural networks in explainer, exact TreeSHAP versus sampled DeepSHAP, and in loss, so the tree-versus-network contrast describes these model classes under their respective fitting and explanation methods.

Results. The primary divide in site attribution is between tree models and neural networks, not between the presence and absence of the symmetry constraint, and it is not a feature-set artifact. On the identical 87-dimensional input, XGBoost is active-site-centered, $M_1$ 79.0% and global 15.7%, whereas both neural networks are global-dominated, $M_1$ 40.0% and global 53.6% for the symmetry-aware ensemble, $M_1$ 31.5% and global 60.1% for the flat MLP. Removing the site-role encoders raises the global share, so the shift belongs to the model class rather than to $M_3\leftrightarrow M_4$ permutation invariance. The descriptive architecture difference is +8.5 percentage points on $M_1$ and −6.5 on global; the site-role model places slightly more attribution on the adsorption site, by a margin far smaller than the 39-point $M_1$ gap between the two model classes. The difference is one of attribution concentration, XGBoost placing most of its $M_1$ weight on three features, M₁_valence_electrons at 0.53, M₁_atomic_radius at 0.20, and M₁_d_band_center at 0.16, while the networks spread comparable total weight across the 52 global statistics, about 1.0% each against about 2.5% per $M_1$ feature. The attribution is stable across the activity range, high-activity, n = 122, and low-activity, n = 873, shares differing by less than 3.1 percentage points; that grouped evaluation uses all 995 samples as background. Table S12 summarizes the comparison, which underlies Figure 7 of the main text.

SHAP assigns feature contributions to model predictions; physical bonding effects are examined through electronic-structure calculations. Both neural models retain the same input permutation invariance, and their comparison changes site grouping and parameter capacity together. The evidence for the architectural advantage is the strict 2×2 ablation, p = 0.003 in S1.5, and the 110-structure blind test, p = 0.001 in S1.5, both independent of this attribution analysis.

---

<a id="s1-10"></a>
### S1.10. Software and Data Records

Machine learning and statistical analyses use NumPy, PyTorch, scikit-learn, XGBoost and SHAP. Batch-specific runtime environments are documented in retained environment files and logs. Splits and random seeds are specified in the corresponding scripts and archived records.

Numerical checks use stored predictions, labels, membership lists and models. Neural-network SHAP uses retained weights; XGBoost interpretation uses archived summaries. All 60 candidate-validation records are supplied in this repository; see the [data guide](../DATA_DICTIONARY.md) for file locations.

---

<a id="s1-11"></a>
### S1.11. Evaluation Sets and Statistical Interpretation

Model comparison. All models use the same invariant 87D input. The site-role ensemble reduces paired errors relative to the flat neural baseline with 49.9% fewer parameters; the comparison changes both site grouping and network capacity. It ranks third in the 13-model cross-validation comparison. Of the 110 held-out structures, 40 are recorded as out-of-distribution; strata are specified in S1.5 and Figure S4.

Retrospective comparison. The fixed pool contains 995 existing DFT labels. Eight paired repeats reveal labels over five acquisition rounds, increasing the labeled set from 200 to 700. Five constituent models and total predictive uncertainty assess ranking efficiency within the pool; full-space screening uses ten models and epistemic uncertainty. Prior discrimination over all 995 labels is a calibration statistic, whereas strategy performance is assessed by the paired retrospective comparison.

Validation sets. Two of the 60 DFT entries are training compositions, so prespecified generalization metrics use the other 38 compositions and the A18/B10 comparison. Hit counts for the full validation set and top-30 are 35/60 and 20/30, respectively; training-unseen counts are 33/58 and 19/29. One validation composition also occurs among the 110 held-out structures. Overlapping records are identified as the same composition.

Electronic-structure comparisons. PBE calculations evaluate adsorption thermodynamics and electronic responses in specified compositions. Figure 7 uses three Co compositions; the expanded summary contains 129 paired records for 107 Co compositions (S1.12). Fe/Ru/Ni/Pd/Pt controls test the orbital-filling interpretation. Sc charge and geometry support stronger ionic character. Co–Zn enrichment derives from model predictions.

Approximate symmetry. Among the 124 training exchange pairs, 28.2% have deviations above 50 meV. Targeted recalculation of 56 high-deviation pairs and independent sampling of 40 early-transition-metal pairs have different sampling scopes. Grouping remote orderings uses approximate energy symmetry, and the grouped input assigns the same prediction to both orderings. The Mann–Whitney comparison of 15 near-window and 109 other pairs tests swap-deviation distributions.

---

<!-- BEGIN R7 SUPPORTING -->
<a id="s1-12"></a>
### S1.12. Site-Resolved Expressions and Coordination Responses


This section specifies evaluation of the site-resolved expression and the definitions used for Co–N₄ charge, bonding and electronic-state comparisons. Original screening labels remain fixed; subsequent fixed-geometry calculations analyze electronic responses.

<a id="s1-12-1"></a>
#### S1.12.1. Expression, model versions and evaluation

The main-text components $D_r$ and $D_e$ correspond to the archived expression D01, with $D_e$ denoted $D_{e1}$ in the analysis files. Radii and d-electron counts are taken from the original elemental table. In particular, the tabulated Co d count is 7, distinct from its nine valence electrons. The 100 pm denominator provides numerical scaling. Both components are dimensionless; the intercept and two slopes for each center have units of eV. The center-radius term is constant within each center element and can be absorbed by its intercept.

The grouped cross-validation results refer to fold-specific fits, with Ridge regularization selected inside each outer training partition and errors macro-averaged equally across center elements. These results use composition-grouped protocols; the ensemble ablation in Figure 3 uses shared random five-fold splits. The frozen expression used to evaluate the 110 historical supplementary records was fitted to all 995 training compositions with $\alpha=1$ and archived as All_De1_original. It has 17 intercepts and 34 slopes. This frozen expression is evaluated retrospectively on the 110 historical records. The complete coefficient table and elemental-input conventions are retained in the detailed descriptor record cited below.

<a id="s1-12-2"></a>
#### S1.12.2. Electronic populations, bonding responses and calculation provenance

The expanded orbital summary contains 129 clean/H record pairs representing 107 Co physical IDs. Directions of clean/H changes are summarized within each archived batch; records for the same composition from different batches are retained separately. Orbital axes follow the calculation coordinates. Occupation-weighted band projections, Bader populations and bond-resolved Hamilton populations describe orbital occupation, spatial electron populations and bonding contributions, respectively. Occupation comparisons account for DOS integration and output precision.

For the ligand comparison, $\Delta q_i=q_i^{\mathrm{H}}-q_i^{\mathrm{clean}}$ with $q_i=Z_i^{\mathrm{val}}-N_i^{\mathrm{Bader}}$. The signed sum over Co and its four N ligands measures the net unit response; $\sum_i\lvert\Delta q_i\rvert$ measures its gross atomic redistribution. A small signed sum can therefore coexist with substantial redistribution. The two charge measures describe net charge change and local charge redistribution, respectively.

Bonding comparisons use archived LOBSTER projections of the specified fixed-geometry states. The standard-basis comparison comprises four compositions, each with four Co–N bonds in clean and H states. Negative ICOHP denotes a net bonding contribution; positive $B=\sum_{\mathrm{Co-N}}(\mathrm{ICOHP}_{\mathrm{H}}-\mathrm{ICOHP}_{\mathrm{clean}})$ indicates that the summed Co–N ICOHP moves toward zero after adsorption, corresponding to bond weakening. Orbital responses first subtract clean from H within a composition, then subtract Cr from Fe at fixed Cu or Mn neighbor and one distal Zn. Standard and alternative basis results are compared separately rather than pooled. Closure residuals between orbital contributions and total ICOHP are retained as printed in the original outputs. The Cu–Cr clean branch differs from the original-label reference, and the Cu–Fe alternative-basis comparison is unavailable. $B$ and its orbital terms describe the Co–N bonding response in the specified local basis.

The two Cu-neighbor cross-geometry anchors retain the native Co–Cu–Cr–Zn and Co–Cu–Fe–Zn clean/H coordinate pairs, respectively. The common background geometry for the Mn/Cu comparison is the latter pair. These calculations use fixed nuclei and compute the adsorption electronic-energy response $J$ from energy-without-entropy differences. The comparison groups VASP 6.3.2 and 6.4.3 as variants of one nominal protocol and records the actual version for each batch. Actual inputs, PAW data, convergence and electronic-state checks remain batch-specific. The restart and noncollinear tests below are separately identified rather than pooled with the original endpoints.

<a id="s1-12-3"></a>
#### S1.12.3. Capacity-balance candidate and prospective Ni comparison

The candidate was defined from the four tabulated d-electron counts in the original 87-dimensional input. With $h(n)=n(10-n)$,

$$
Q_{\mathrm{bal}}=
\left[
\frac{h(n_{d,3})+h(n_{d,4})}{2}
-\frac{h(n_{d,1})+h(n_{d,2})}{2}
\right]^2.
$$

This parameter-free quantity is invariant to remote-site exchange. For the two Cr→Fe substitutions with Cu and Mn nearest neighbors, it changed in the same direction as $B$ and oppositely to signed $\Delta G_{\mathrm{H}}$. These two responses informed the candidate definition. The Cu-side prediction was $B_{\mathrm{Ni}}>B_{\mathrm{Cr}}>B_{\mathrm{Fe}}$.

Both fixed-geometry Ni states passed technical-completeness and occupied-subspace projection checks. The H state reproduced the existing static reference at the same nuclear coordinates. The clean state differed in energy without entropy by −31.32533 meV; even after allowing global spin reversal, the whole-cell moment and maximum metal-local-moment differences were 0.1078 and 0.509 $\mu_{\mathrm{B}}$. Electronic-branch compatibility therefore failed, and the formal ordering test was not applied. The returned branch gave $B_{\mathrm{Ni}}=0.26202$ eV, compared with $B_{\mathrm{Cr}}=0.44179$ eV and $B_{\mathrm{Fe}}=0.35268$ eV. The returned-branch ordering opposes the candidate prediction. The incompatible clean electronic states identify this result as a cross-branch comparison.

<a id="s1-12-4"></a>
#### S1.12.4. Fixed geometry, restart sensitivity and noncollinear compatibility

The two Cu-neighbor geometry anchors gave $J=+78.70876$ and +54.55077 meV using energies without entropy. The comparison uses a 10 meV criterion. Clean and H structures are distinct, and the cross-geometries were not relaxed. The Cu magnetic-sign classification uses remote Cr/Fe as its reference; electronic-branch compatibility is evaluated through energy and magnetic-moment comparisons.

At the common geometry, restarting MnFe clean from its original wavefunction with a tighter electronic stopping threshold lowered the energy without entropy by 168.80494 meV and changed the Co moment from −0.531 to +0.703 $\mu_{\mathrm{B}}$. Holding the other seven endpoints fixed changed $J_{\mathrm{Mn}}$ from −126.41696 to +42.38798 meV and $I=J_{\mathrm{Mn}}-J_{\mathrm{Cu}}$ from −180.96773 to −12.16279 meV. This endpoint substitution tests sensitivity to restarting and electronic-state selection, retaining the distinction from the original calculation protocol.

The unconstrained noncollinear baseline without spin–orbit coupling lowered the energy by a further 1.37447 meV, exceeding the prespecified 1 meV compatibility tolerance. Metal-local and whole-cell magnetic vectors and magnetization-density consistency checks passed their operational tolerances. However, XML occupations were printed to only four decimal places, leaving the electron-count and highest-eight-band occupation checks indeterminate. Auxiliary higher-precision PROCAR occupations supported the same numerical thresholds but did not replace the prespecified XML evidence. Full compatibility was not established, and the new clean endpoint was not combined with an old H endpoint to define a new adsorption label.

<a id="s1-12-5"></a>
#### S1.12.5. Data sources

Definitions and results are available in the [expression guide](../../data/analysis/descriptors/expression_definition/section_detailed_EN.md), [full-precision coefficients](../../data/analysis/descriptors/expression_validation/final_coefficients.csv), and [electronic-structure guide](../../data/analysis/electronic_structure/README.md).

---

<!-- ===================== TABLES ===================== -->
<a id="s2"></a>
## S2. Supplementary Tables

<a id="table-s1"></a>
### Table S1. Predicted Free Energies and Epistemic Uncertainties of 137 All-3d Candidates

| Rank | $M_1$ | $M_2$ | $M_3$ | $M_4$ | $\Delta G_{\mathrm{H}}$ symmetry-aware ensemble (eV) | $\sigma_{\mathrm{epi}}$ (eV) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1 | Co | Fe | Cr | V | +0.0002 | 0.0385 |
| 2 | Co | Fe | V | Zn | -0.0012 | 0.0425 |
| 3 | Sc | Co | Mn | V | -0.0014 | 0.0482 |
| 4 | Co | Ni | Fe | Zn | +0.0014 | 0.0270 |
| 5 | Co | Fe | Mn | Zn | +0.0017 | 0.0457 |
| 6 | Sc | Mn | Ti | V | -0.0022 | 0.0415 |
| 7 | Co | Fe | Ni | V | -0.0048 | 0.0451 |
| 8 | Co | Fe | Cr | Cu | +0.0053 | 0.0414 |
| 9 | Co | Cr | Cu | Zn | +0.0054 | 0.0337 |
| 10 | Sc | Mn | Cr | V | +0.0074 | 0.0463 |
| 11 | Co | Mn | Cu | Zn | +0.0117 | 0.0244 |
| 12 | Co | Fe | Cr | Zn | -0.0146 | 0.0577 |
| 13 | Sc | Mn | Fe | V | -0.0153 | 0.0474 |
| 14 | Sc | Co | Cr | Mn | -0.0165 | 0.0501 |
| 15 | Co | Fe | Cr | Ni | +0.0181 | 0.0344 |
| 16 | Co | Mn | Ni | Zn | -0.0190 | 0.0265 |
| 17 | Co | Ni | Mn | Zn | +0.0216 | 0.0344 |
| 18 | Sc | Mn | Cr | Fe | -0.0218 | 0.0408 |
| 19 | Co | Ni | Cu | Zn | -0.0226 | 0.0378 |
| 20 | Co | Fe | Cr | Ti | +0.0235 | 0.0413 |
| 21 | Co | Fe | Cu | V | +0.0273 | 0.0551 |
| 22 | Co | Cr | Ni | Zn | -0.0276 | 0.0428 |
| 23 | Co | Ni | Cr | Zn | +0.0276 | 0.0313 |
| 24 | Co | Cu | Ni | Zn | -0.0285 | 0.0523 |
| 25 | Sc | Mn | Co | Fe | -0.0287 | 0.0401 |
| 26 | Co | Ni | Cr | Cu | +0.0290 | 0.0470 |
| 27 | Co | Cu | Fe | Zn | +0.0302 | 0.0364 |
| 28 | Co | Cr | Fe | Zn | +0.0303 | 0.0416 |
| 29 | Sc | Ni | Mn | V | -0.0311 | 0.0502 |
| 30 | Co | Cr | Cu | Ni | +0.0314 | 0.0560 |
| 31 | Co | Zn | Mn | Ni | +0.0332 | 0.0357 |
| 32 | Co | Zn | Cu | Ni | +0.0333 | 0.0507 |
| 33 | Sc | Mn | Cu | Fe | -0.0334 | 0.0574 |
| 34 | Sc | Mn | Co | V | -0.0341 | 0.0504 |
| 35 | Co | Mn | Fe | Zn | +0.0342 | 0.0306 |
| 36 | Co | Fe | Mn | V | +0.0351 | 0.0580 |
| 37 | Sc | Mn | Cr | Zn | -0.0371 | 0.0506 |
| 38 | Co | Cu | Mn | Zn | +0.0381 | 0.0271 |
| 39 | Co | Zn | Cu | Mn | +0.0389 | 0.0478 |
| 40 | Co | Fe | Cr | Mn | +0.0405 | 0.0546 |
| 41 | Co | Cr | V | Zn | +0.0407 | 0.0427 |
| 42 | Co | Ni | Cu | Fe | +0.0410 | 0.0415 |
| 43 | Co | Fe | Mn | Ni | +0.0427 | 0.0562 |
| 44 | Sc | Mn | Co | Cr | -0.0430 | 0.0401 |
| 45 | Co | Cr | Mn | Zn | +0.0434 | 0.0245 |
| 46 | Co | Cr | Ni | V | +0.0435 | 0.0460 |
| 47 | Sc | Mn | Fe | Ni | -0.0441 | 0.0429 |
| 48 | Co | Zn | Mn | V | +0.0455 | 0.0434 |
| 49 | Co | Ni | V | Zn | +0.0462 | 0.0295 |
| 50 | Sc | Mn | Cr | Cu | -0.0462 | 0.0553 |
| 51 | Sc | Mn | Co | Ni | -0.0469 | 0.0455 |
| 52 | Co | Zn | Cr | Cu | +0.0486 | 0.0492 |
| 53 | Co | Cu | Cr | Zn | +0.0486 | 0.0396 |
| 54 | Co | Zn | Cr | Mn | +0.0499 | 0.0344 |
| 55 | Co | Ni | Cr | V | +0.0501 | 0.0394 |
| 56 | Co | Fe | Ti | V | +0.0504 | 0.0415 |
| 57 | Co | Mn | Cu | Ni | +0.0506 | 0.0436 |
| 58 | Co | Mn | Cr | Zn | +0.0513 | 0.0344 |
| 59 | Sc | Ni | Cr | Mn | -0.0522 | 0.0571 |
| 60 | Co | Cu | Cr | Ni | +0.0535 | 0.0356 |
| 61 | Co | Ni | Cr | Fe | +0.0538 | 0.0388 |
| 62 | Co | Zn | Fe | Mn | +0.0551 | 0.0432 |
| 63 | Co | Zn | Ni | V | +0.0556 | 0.0483 |
| 64 | Co | Zn | Cr | Ni | +0.0558 | 0.0319 |
| 65 | Sc | V | Cr | Mn | -0.0560 | 0.0504 |
| 66 | Co | Ni | Fe | V | +0.0562 | 0.0358 |
| 67 | Sc | Fe | Mn | Zn | -0.0568 | 0.0363 |
| 68 | Co | Cu | Mn | Ni | +0.0570 | 0.0430 |
| 69 | Co | Cr | Mn | Ni | +0.0572 | 0.0258 |
| 70 | Sc | Mn | Fe | Ti | -0.0572 | 0.0490 |
| 71 | Co | Ni | Cu | Mn | +0.0582 | 0.0515 |
| 72 | Sc | Mn | Ni | V | -0.0583 | 0.0551 |
| 73 | Co | Cr | Fe | V | +0.0585 | 0.0312 |
| 74 | Sc | Co | Cu | Mn | -0.0585 | 0.0561 |
| 75 | Co | Mn | V | Zn | +0.0591 | 0.0334 |
| 76 | Co | Mn | Cr | Ni | +0.0606 | 0.0452 |
| 77 | Co | Ni | Cr | Mn | +0.0607 | 0.0346 |
| 78 | Sc | Mn | Cr | Ni | -0.0612 | 0.0451 |
| 79 | Co | Mn | Cr | V | +0.0629 | 0.0484 |
| 80 | Co | Cr | Cu | Mn | +0.0632 | 0.0441 |
| 81 | Co | Cr | Mn | V | +0.0639 | 0.0313 |
| 82 | Sc | Cr | V | Zn | -0.0646 | 0.0526 |
| 83 | Co | Cu | V | Zn | +0.0652 | 0.0431 |
| 84 | Co | Zn | Cu | V | +0.0656 | 0.0530 |
| 85 | Co | Mn | Cr | Fe | +0.0673 | 0.0352 |
| 86 | Co | Zn | Cu | Fe | +0.0680 | 0.0493 |
| 87 | Co | Cu | Cr | Mn | +0.0683 | 0.0404 |
| 88 | Co | Zn | Cr | V | +0.0702 | 0.0388 |
| 89 | Co | Ni | Cu | V | +0.0706 | 0.0581 |
| 90 | Co | Cr | Fe | Mn | +0.0714 | 0.0286 |
| 91 | Co | Cr | Cu | Fe | +0.0728 | 0.0376 |
| 92 | Co | Ni | Mn | V | +0.0731 | 0.0428 |
| 93 | Co | Ni | Fe | Mn | +0.0736 | 0.0319 |
| 94 | Co | Mn | Cr | Cu | +0.0736 | 0.0565 |
| 95 | Sc | Cu | Mn | V | -0.0738 | 0.0458 |
| 96 | Co | Cu | Cr | V | +0.0748 | 0.0425 |
| 97 | Co | Mn | Fe | V | +0.0754 | 0.0394 |
| 98 | Co | Cu | Ni | V | +0.0775 | 0.0573 |
| 99 | Sc | Ni | Mn | Zn | -0.0793 | 0.0556 |
| 100 | Co | Ni | Cr | Ti | +0.0795 | 0.0459 |
| 101 | Co | Cr | Mn | Ti | +0.0798 | 0.0346 |
| 102 | Sc | Fe | Cu | Mn | -0.0801 | 0.0510 |
| 103 | Co | Cr | Fe | Ni | +0.0804 | 0.0353 |
| 104 | Co | Zn | Cr | Ti | +0.0805 | 0.0548 |
| 105 | Sc | V | Mn | Zn | -0.0813 | 0.0529 |
| 106 | Co | Mn | Ni | V | +0.0824 | 0.0515 |
| 107 | Co | Cr | Ti | V | +0.0826 | 0.0394 |
| 108 | Sc | V | Mn | Ti | -0.0830 | 0.0488 |
| 109 | Co | Cu | Fe | Mn | +0.0840 | 0.0431 |
| 110 | Co | Cu | Mn | V | +0.0856 | 0.0470 |
| 111 | Sc | Cr | Mn | Zn | -0.0856 | 0.0386 |
| 112 | Sc | Mn | Co | Ti | -0.0865 | 0.0459 |
| 113 | Sc | Ni | Fe | Mn | -0.0867 | 0.0522 |
| 114 | Co | Zn | Cr | Fe | +0.0869 | 0.0380 |
| 115 | Co | Mn | Fe | Ni | +0.0890 | 0.0413 |
| 116 | Co | Zn | Mn | Sc | +0.0905 | 0.0538 |
| 117 | Co | Zn | Fe | V | +0.0910 | 0.0327 |
| 118 | Co | Zn | Fe | Ni | +0.0913 | 0.0326 |
| 119 | Co | Mn | Cu | Fe | +0.0913 | 0.0451 |
| 120 | Cr | Zn | Ni | V | +0.0914 | 0.0250 |
| 121 | Co | Mn | Ti | V | +0.0921 | 0.0580 |
| 122 | Cr | Zn | Cu | V | +0.0921 | 0.0194 |
| 123 | Co | Zn | Ni | Ti | +0.0923 | 0.0544 |
| 124 | Cr | Zn | Cu | Ti | +0.0929 | 0.0174 |
| 125 | Co | Mn | Cr | Ti | +0.0931 | 0.0482 |
| 126 | Cr | Zn | Ni | Ti | +0.0951 | 0.0241 |
| 127 | Co | Cu | Cr | Ti | +0.0953 | 0.0489 |
| 128 | Cr | Zn | Mn | Ti | +0.0958 | 0.0215 |
| 129 | Cr | Zn | Mn | V | +0.0965 | 0.0176 |
| 130 | Co | Cr | Fe | Ti | +0.0970 | 0.0327 |
| 131 | Co | Zn | Ti | V | +0.0971 | 0.0552 |
| 132 | Cr | Zn | Mn | Ni | +0.0975 | 0.0200 |
| 133 | Sc | V | Cu | Mn | -0.0980 | 0.0517 |
| 134 | Sc | Cu | Cr | Mn | -0.0985 | 0.0531 |
| 135 | Co | Mn | Cu | V | +0.0986 | 0.0575 |
| 136 | Cr | Zn | Cu | Mn | +0.0987 | 0.0184 |
| 137 | Cr | Zn | Co | V | +0.0988 | 0.0294 |

---

<a id="table-s2"></a>
### Table S2. Descriptor Dictionary and Construction of the 87-Dimensional Input

| New block (87D) | # | Raw 96D descriptors |
|:--:|:--:|:--:|
| $M_1$ (unchanged) | 16 | M1_atomic_number<br>M1_atomic_weight<br>M1_atomic_radius<br>M1_electronegativity<br>M1_d_electrons<br>M1_valence_electrons<br>M1_d_band_center<br>M1_d_band_width<br>M1_work_function<br>M1_surface_energy<br>M1_lattice_constant<br>M1_melting_point<br>M1_cohesive_energy<br>M1_H_bond_polarity<br>M1_d_band_filling<br>M1_work_function_deviation |
| $M_2$ (first-shell neighbor) | 13 | M2_atomic_number<br>M2_atomic_weight<br>M2_atomic_radius<br>M2_electronegativity<br>M2_d_electrons<br>M2_valence_electrons<br>M2_d_band_center<br>M2_d_band_width<br>M2_work_function<br>M2_surface_energy<br>M2_lattice_constant<br>M2_melting_point<br>M2_cohesive_energy |
| $M_{34}$ symmetric (mean/absdiff) | 6 | electronegativity_M34_mean<br>electronegativity_M34_absdiff<br>d_electrons_M34_mean<br>d_electrons_M34_absdiff<br>atomic_radius_M34_mean<br>atomic_radius_M34_absdiff |
| Global (unordered set) | 52 | atomic_number_weighted_mean<br>atomic_number_std<br>atomic_number_range<br>atomic_number_m1_deviation<br>atomic_weight_weighted_mean<br>atomic_weight_std<br>atomic_weight_range<br>atomic_weight_m1_deviation<br>atomic_radius_weighted_mean<br>atomic_radius_std<br>atomic_radius_range<br>atomic_radius_m1_deviation<br>electronegativity_weighted_mean<br>electronegativity_std<br>electronegativity_range<br>electronegativity_m1_deviation<br>d_electrons_weighted_mean<br>d_electrons_std<br>d_electrons_range<br>d_electrons_m1_deviation<br>valence_electrons_weighted_mean<br>valence_electrons_std<br>valence_electrons_range<br>valence_electrons_m1_deviation<br>d_band_center_weighted_mean<br>d_band_center_std<br>d_band_center_range<br>d_band_center_m1_deviation<br>d_band_width_weighted_mean<br>d_band_width_std<br>d_band_width_range<br>d_band_width_m1_deviation<br>work_function_weighted_mean<br>work_function_std<br>work_function_range<br>work_function_m1_deviation<br>surface_energy_weighted_mean<br>surface_energy_std<br>surface_energy_range<br>surface_energy_m1_deviation<br>lattice_constant_weighted_mean<br>lattice_constant_std<br>lattice_constant_range<br>lattice_constant_m1_deviation<br>melting_point_weighted_mean<br>melting_point_std<br>melting_point_range<br>melting_point_m1_deviation<br>cohesive_energy_weighted_mean<br>cohesive_energy_std<br>cohesive_energy_range<br>cohesive_energy_m1_deviation |
| Dropped (9 derived columns; S1.9) | 9 | electronegativity_to_d_electrons_ratio<br>work_function_to_d_band_ratio<br>atomic_size_mismatch<br>electron_affinity_M1<br>gcn_deviation<br>neighbor_H_competition<br>lattice_strain_M1<br>surface_dipole_moment<br>electronic_configuration_index |

---

<a id="table-s3"></a>
### Table S3. 2×2 Ablation of Model Architecture and Loss Function

| Configuration | Architecture | Loss | CV MAE (eV) | R² | Parameters |
|:--:|:--:|:--:|:--:|:--:|:--:|
| Flat+MSE | Flat MLP | Homo MSE | 0.1172 ± 0.0110 | 0.9363 | 64706 |
| Flat+NLL | Flat MLP | Hetero NLL | 0.1193 ± 0.0112 | 0.9329 | 64706 |
| Sym+MSE | Symmetric | Homo MSE | 0.1085 ± 0.0107 | 0.9451 | 32418 |
| Sym+NLL | Symmetric | Hetero NLL | 0.1067 ± 0.0089 | 0.9444 | 32418 |

All four cells share the same five-fold splits, so each factor is isolated by fold-paired $t$-tests. Architecture, holding input and loss fixed, gives ΔMAE = −0.0126 eV at NLL, $t(4)=-6.64$, $p=0.003$, and −0.0086 eV at MSE, $t(4)=-3.74$, $p=0.020$. Loss, holding input and architecture fixed, gives −0.0018 eV on the symmetric architecture, $t(4)=-1.18$, $p=0.304$, and +0.0022 eV on the flat MLP, $t(4)=2.10$, $p=0.103$. A two-way ANOVA on the 20 fold-level MAEs with fold as a blocking factor agrees, architecture $F(1,12)=70.7$, $p<10^{-5}$, loss $F(1,12)=0.02$, $p=0.89$, and interaction $F(1,12)=2.44$, $p=0.14$.

---

<a id="table-s4"></a>
### Table S4. Five-Fold Cross-Validation of 13 Models

| Method | CV MAE (eV) | R² |
|:--:|:--:|:--:|
| XGBoost (tuned) | 0.1058 ± 0.0092 | 0.9430 |
| LightGBM | 0.1082 | 0.9387 |
| Symmetry-aware ensemble V2 | 0.1095 ± 0.0101 | 0.9414 |
| GradientBoosting | 0.1104 | 0.9394 |
| RandomForest | 0.1114 | 0.9392 |
| CatBoost | 0.1154 | 0.9401 |
| Flat MLP | 0.1207 ± 0.0108 | 0.9340 |
| GPR | 0.1385 | 0.9235 |
| DecisionTree | 0.1607 | 0.8854 |
| SVR | 0.1677 | 0.8785 |
| AdaBoost | 0.1818 | 0.8876 |
| BayesianRidge | 0.2298 | 0.8231 |
| Lasso | 0.4289 | 0.4343 |

All models were evaluated under the same unified protocol, 87D input and shared five-fold CV with fold-internal standardization. The neural-network entries here are the unified-protocol baselines, so the Symmetry-aware ensemble V2 (0.1095) and Flat MLP (0.1207) differ slightly from the strict 2×2 ablation cells in Table S3 (Sym+NLL 0.1067 and Flat+NLL 0.1193); the Flat MLP entry corresponds to the Flat+NLL cell. LightGBM and GPR entries are the retrained values, 0.1082 eV from 0.2596 eV on a 576-combination grid and 0.1385 eV from 0.2278 eV on the full 995 samples. The five-fold CV MAE of the symmetry-aware ensemble, 0.1095 ± 0.0101 eV, coincides numerically with its blind-test MAE of 0.1095 eV, but the two are independent quantities, the former computed over the five validation folds and the latter over the 110 held-out structures.

---

<a id="table-s5"></a>
### Table S5. Paired Retrospective Comparison of Chemically Guided Selection

(a) Summary statistics across eight paired repeats

| Metric | Pure UCB | Physics-guided | Δ | t | p | Better |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Final cumulative candidates within the window (≤0.1 eV) | 115.50 | 119.25 | +3.75 | 4.16 | 0.0042 | Physics-guided |
| Final cumulative candidates within the window (≤0.2 eV) | 228.00 | 238.00 | +10.00 | 6.67 | 0.0003 | Physics-guided |
| Running sum of best $\lvert\Delta G_{\mathrm{H}}\rvert$ across rounds | 0.00508 | 0.00229 | -0.00279 | -3.17 | 0.016 | Physics-guided |
| Mean 9-valence fraction per batch | 0.324 | 0.337 | +0.013 | 8.20 | 0.0001 | Physics-guided |

(b) Terminal within-window candidate counts per replicate

| Replicate | Pure UCB (≤0.2 eV) | Physics-guided (≤0.2 eV) | Pure UCB (≤0.1 eV) | Physics-guided (≤0.1 eV) |
|:--:|:--:|:--:|:--:|:--:|
| 0 | 229 | 241 | 116 | 121 |
| 1 | 228 | 240 | 116 | 120 |
| 2 | 230 | 239 | 115 | 119 |
| 3 | 225 | 241 | 111 | 120 |
| 4 | 235 | 239 | 115 | 118 |
| 5 | 220 | 234 | 116 | 119 |
| 6 | 234 | 239 | 119 | 120 |
| 7 | 223 | 231 | 116 | 117 |

---

<a id="table-s6"></a>
### Table S6. DFT Reassessment of 56 Pairs with Large Exchange Deviations

| $M_1$ | n | Historical median (meV) | Recalculated median (meV) | >10 meV after recalc | >50 meV after recalc |
|:--:|:--:|:--:|:--:|:--:|:--:|
| Au | 17 | 40.4 | 45.0 | 16 | 8 |
| Co | 5 | 110.1 | 131.3 | 5 | 5 |
| Fe | 26 | 145.3 | 100.8 | 22 | 17 |
| Ir | 1 | 36.6 | 0.1 | 0 | 0 |
| Pd | 3 | 37.6 | 4.9 | 1 | 0 |
| Pt | 3 | 10.5 | 16.5 | 2 | 1 |
| Rh | 1 | 14.0 | 70.2 | 1 | 1 |
| All | 56 | 95.2 | 84.2 | 47 (84%) | 32 (57%) |

---

<a id="table-s7"></a>
### Table S7. Independent Validation of 40 Early-Transition-Metal Exchange Pairs

| $M_1$ | n | Median (meV) | Symmetric (<10 meV) |
|:--:|:--:|:--:|:--:|
| Cr | 8 | 1.6 | 7/8 |
| Mn | 8 | 3.8 | 6/8 |
| Sc | 8 | 10.6 | 4/8 |
| Ti | 8 | 0.8 | 7/8 |
| V | 8 | 6.4 | 5/8 |
| All | 40 | 2.9 | 29/40 |

---

<a id="table-s8"></a>
### Table S8. Effects of Descriptor Count and Regression Form on Predictive Performance

| # descriptors | R² (linear) | R² (quadratic) | MAE (linear, eV) |
|:--:|:--:|:--:|:--:|
| 1 | 0.299 | 0.319 | 0.485 |
| 2 | 0.298 | 0.375 | 0.486 |
| 3 | 0.409 | 0.723 | 0.453 |
| 5 | 0.514 | 0.870 | 0.396 |
| 8 | 0.560 | 0.908 | 0.372 |
| 12 | 0.661 | 0.915 | 0.316 |

Signed-free-energy predictions on 110 held-out DFT labels give R² = 1 − SSE/SST = 0.8927, MAE 0.1554 eV and RMSE 0.212 eV. Squared Pearson correlation is a different metric.

The k-sweep rows above use the top-k descriptors by SHAP importance, five-fold CV on degree-2 polynomial features with linear regression. The five-descriptor model in Figure 7 fixes the descriptor set to the $M_1$ valence-electron count, atomic radius, d-band center, lattice constant, and electronegativity, fit with Ridge regression, α = 0.1; its five-fold CV R² is 0.89, and the blind-test line above belongs to it.

---

<a id="table-s9"></a>
### Table S9. DFT Validation of 40 Pre-Specified Candidate Compositions

| Sample | $M_1$ | $M_2$ | $M_3$ | $M_4$ | Group | Pred $\Delta G_{\mathrm{H}}$ (eV) | DFT $\Delta G_{\mathrm{H}}$ (eV) | $\lvert\Delta G_{\mathrm{H}}\rvert$ (eV) | Within window (≤0.1 eV) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1 | Co | Cr | Cu | Zn | A (low-cost) | +0.005 | 0.075 | 0.075 | ✓ |
| 2 | Co | Mn | Ni | Zn | A (low-cost) | -0.019 | -0.047 | 0.047 | ✓ |
| 3 | Co | Fe | Mn | Zn | A (low-cost) | +0.002 | 0.022 | 0.022 | ✓ |
| 4 | Co | Cr | Ni | Zn | A (low-cost) | -0.028 | -0.123 | 0.123 | – |
| 5 | Co | Fe | V | Zn | A (low-cost) | -0.001 | 0.103 | 0.103 | – |
| 6 | Co | Mn | Fe | Zn | A (low-cost) | +0.034 | 0.054 | 0.054 | ✓ |
| 7 | Co | Ni | Mn | Zn | A (low-cost) | +0.022 | 0.005 | 0.005 | ✓ |
| 8 | Co | Zn | Mn | Ni | A (low-cost) | +0.033 | 0.039 | 0.039 | ✓ |
| 9 | Co | Cr | V | Zn | A (low-cost) | +0.041 | 0.247 | 0.247 | – |
| 10 | Co | Zn | Cr | Cu | A (low-cost) | +0.049 | -0.045 | 0.045 | ✓ |
| 11 | Co | Cu | Cr | Zn | A (low-cost) | +0.049 | -0.077 | 0.077 | ✓ |
| 12 | Co | Mn | Cu | Zn | A (low-cost) | +0.012 | 0.027 | 0.027 | ✓ |
| 13 | Co | Cr | Mn | Zn | A (low-cost) | +0.043 | 0.041 | 0.041 | ✓ |
| 14 | Co | Zn | Mn | V | A (low-cost) | +0.046 | 0.134 | 0.134 | – |
| 15 | Co | Cr | Fe | Zn | A (low-cost) | +0.030 | 0.021 | 0.021 | ✓ |
| 16 | Co | Ni | Fe | Zn | A (low-cost) | +0.001 | 0.063 | 0.063 | ✓ |
| 17 | Co | Zn | Fe | Mn | A (low-cost) | +0.055 | 0.151 | 0.151 | – |
| 18 | Co | Mn | Cr | Zn | A (low-cost) | +0.051 | 0.062 | 0.062 | ✓ |
| 19 | Co | Cu | Fe | Zn | A (low-cost) | +0.030 | -0.0015 | 0.0015 | ✓ |
| 20 | Co | Zn | Cr | V | A (low-cost) | +0.070 | 0.144 | 0.144 | – |
| 21 | Co | Ru | Mn | Zn | B (precious) | -0.007 | -0.131 | 0.131 | – |
| 22 | Co | Cr | Ru | V | B (precious) | -0.020 | 0.012 | 0.012 | ✓ |
| 23 | Co | Ir | Cr | Pt | B (precious) | -0.004 | -0.072 | 0.072 | ✓ |
| 24 | Co | Zn | Ag | Pt | B (precious) | +0.004 | 0.005 | 0.005 | ✓ |
| 25 | Rh | Sc | Cr | Cu | B (precious) | +0.009 | 0.068 | 0.068 | ✓ |
| 26 | Co | Ir | Ag | Mn | B (precious) | -0.000 | -0.064 | 0.064 | ✓ |
| 27 | Co | Ir | Cr | Ti | B (precious) | +0.002 | -0.093 | 0.093 | ✓ |
| 28 | Co | Rh | Cr | Ru | B (precious) | +0.003 | 0.043 | 0.043 | ✓ |
| 29 | Co | Cr | Cu | Ru | B (precious) | -0.008 | 0.113 | 0.113 | – |
| 30 | Rh | Cu | Fe | Sc | B (precious) | +0.046 | -0.090 | 0.090 | ✓ |
| 31 | Rh | V | Pd | Ti | C (calibration) | -0.038 | 0.112 | 0.112 | – |
| 32 | Cr | Au | Rh | Sc | C (calibration) | +0.109 | 0.023 | 0.023 | ✓ |
| 33 | Co | Cr | Cu | Pt | C (calibration) | +0.028 | -0.101 | 0.101 | – |
| 34 | V | Co | Au | Cu | C (calibration) | -0.227 | -0.232 | 0.232 | – |
| 35 | V | Cu | Au | Pt | C (calibration) | -0.243 | -0.308 | 0.308 | – |
| 36 | Cr | Cu | Ag | Rh | C (calibration) | +0.137 | 0.101 | 0.101 | – |
| 37 | Ag | Au | Ni | Pt | C (calibration) | +0.396 | 0.667 | 0.667 | – |
| 38 | Ag | V | Au | Pd | C (calibration) | +0.424 | 0.561 | 0.561 | – |
| 39 | Zn | Pd | Co | Sc | C (calibration) | +0.467 | 0.491 | 0.491 | – |
| 40 | Fe | Sc | Ag | Rh | C (calibration) | +0.557 | 0.339 | 0.339 | – |

<a id="table-s10"></a>
### Table S10. DFT Validation of 20 Additional Compositions in the All-3d Top-30

| Rank | $M_1$ | $M_2$ | $M_3$ | $M_4$ | Pred $\Delta G_{\mathrm{H}}$ (eV) | DFT $\Delta G_{\mathrm{H}}$ (eV) | $\lvert\Delta G_{\mathrm{H}}\rvert$ (eV) | Within window (≤0.1 eV) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 1 | Co | Fe | Cr | V | +0.000 | +0.098 | 0.098 | ✓ |
| 3 | Sc | Co | Mn | V | -0.001 | -0.039 | 0.039 | ✓ |
| 6 | Sc | Mn | Ti | V | -0.002 | -0.173 | 0.173 | – |
| 7 | Co | Fe | Ni | V | -0.005 | +0.101 | 0.101 | – |
| 8 | Co | Fe | Cr | Cu | +0.005 | -0.138 | 0.138 | – |
| 10 | Sc | Mn | Cr | V | +0.007 | -0.121 | 0.121 | – |
| 12 | Co | Fe | Cr | Zn | -0.015 | +0.066 | 0.066 | ✓ |
| 13 | Sc | Mn | Fe | V | -0.015 | +0.027 | 0.027 | ✓ |
| 14 | Sc | Co | Cr | Mn | -0.017 | +0.050 | 0.050 | ✓ |
| 15 | Co | Fe | Cr | Ni | +0.018 | +0.034 | 0.034 | ✓ |
| 18 | Sc | Mn | Cr | Fe | -0.022 | +0.131 | 0.131 | – |
| 19 | Co | Ni | Cu | Zn | -0.023 | -0.060 | 0.060 | ✓ |
| 20 | Co | Fe | Cr | Ti | +0.024 | +0.110 | 0.110 | – |
| 21 | Co | Fe | Cu | V | +0.027 | +0.102 | 0.102 | – |
| 23 | Co | Ni | Cr | Zn | +0.028 | -0.058 | 0.058 | ✓ |
| 24 | Co | Cu | Ni | Zn | -0.029 | -0.100 | 0.100 | – |
| 25 | Sc | Mn | Co | Fe | -0.029 | +0.027 | 0.027 | ✓ |
| 26 | Co | Ni | Cr | Cu | +0.029 | -0.067 | 0.067 | ✓ |
| 29 | Sc | Ni | Mn | V | -0.031 | -0.096 | 0.096 | ✓ |
| 30 | Co | Cr | Cu | Ni | +0.031 | -0.084 | 0.084 | ✓ |

Rank is the position in the V2 137-structure all-3d low-cost shortlist (Table S1), ordered by predicted $|\Delta G_{\mathrm{H}}|$. Ranks 1–20 and 21–30 were computed in two batches after the final model was fixed; together with the 10 top-30 members already inside group A, ranks 2, 4, 5, 9, 11, 16, 17, 22, 27, and 28, all 30 members of the low-cost top-30 are now validated explicitly, 20 of 30 with $|\Delta G_{\mathrm{H}}|$ ≤ 0.1 eV, 67%, at mean $|\Delta G_{\mathrm{H}}|$ = 0.072 eV, and the best-performing composition Co–Cu–Fe–Zn, rank 27, remains the best of all 60 validated structures, DFT −0.0015 eV.

<a id="table-s11"></a>
### Table S11. Composition Overlap and Training-Unseen Validation Results

| Set | n | Hits ≤0.1 eV | RMSE (eV) | Spearman ρ |
|---|---|---|---|---|
| All DFT | 60 | 35 | 0.097250 | 0.406891 |
| Training-unseen DFT | 58 | 33 | 0.098845 | 0.402196 |
| Pre-specified | 40 | 23 | 0.098288 | 0.548780 |
| Pre-specified, training-unseen | 38 | 21 | 0.100741 | 0.545464 |
| Completion | 20 | 12 | 0.095140 | 0.010526 |
| All-3d top-30 | 30 | 20 | 0.084157 | 0.055395 |
| Top-30, training-unseen | 29 | 19 | 0.085442 | 0.025616 |

Co–Cr–Mn–Zn and Co–Mn–Ni–Zn in Table S9 are already in training; the latter is rank 16 in the final top-30. Co–Ir–Cr–Ti is shared with the 110-structure held-out set, which is disjoint from training. All original DFT rows are retained. The primary A/B comparison excludes the two training-seen A members, yielding A18 versus B10, means 0.077529/0.069066 eV and Welch p=0.669786; the full A20/B10 gives p=0.787167.

<a id="table-s12"></a>
### Table S12. Comparison of SHAP Site Attributions across Models

(a) Three-way comparison on the identical 87-dimensional feature set (all 995 samples; both neural networks share the same 100 background samples)

| Model | Features | $M_1$ | $M_2$ | $M_{34}$ | global |
|-------|----------|-----|-----|------|--------|
| XGBoost (TreeExplainer, exact) | 87D | 79.0% | 3.1% | 2.2% | 15.7% |
| Flat MLP (Flat_NLL ablation cell) | 87D | 31.5% | 5.5% | 2.9% | 60.1% |
| Symmetry-aware ensemble | 87D | 40.0% | 3.6% | 2.8% | 53.6% |
| Net architectural effect (symmetry − flat) | 87D | +8.5 | −1.9 | −0.1 | −6.5 |
| *XGBoost, historical feature-set control* | *96D* | *78.1%* | *3.2%* | *2.2%* | *16.5%* |

(b) Activity-resolved attribution of the symmetry-aware ensemble (background = all 995 samples; not interchangeable with (a))

| Group | $M_1$ | $M_2$ | $M_{34}$ | global |
|-------|-----|-----|------|--------|
| High activity ($\lvert\Delta G_{\mathrm{H}}\rvert$ ≤ 0.1 eV, n=122) | 34.8% | 4.9% | 2.5% | 57.8% |
| Low activity (n=873) | 37.8% | 3.6% | 1.9% | 56.7% |

In (a) all three models are explained on the same 87-dimensional symmetric input with the same index-range site partition, so the percentages are directly comparable. The flat MLP and the symmetry-aware ensemble additionally share the NLL loss, ensemble size, and random seed, with architecture as the only difference; the fourth row is therefore the net effect attributable to the architectural inductive bias. Both neural networks are global-dominated, and removing site-role encoders raises the global share, indicating that global dominance arises from the model class rather than from $M_3\leftrightarrow M_4$ permutation invariance or from the choice of feature set. The italicized last row is the 96-dimensional XGBoost, retained only as a historical feature-set control and no longer as the screening surrogate over the 28560-structure space; it is retained for provenance of the Figure 7 feature-importance numbers. Its $M_1$ share is 0.90 percentage points lower than the 87-dimensional control because two of its nine extra columns, electron_affinity_M1 and lattice_strain_M1, nominally belong to $M_1$ but fall into the global bin under name-based grouping; restricting XGBoost to 87D costs no accuracy, S1.9. In (b) the two groups agree within 3.1 percentage points, indicating a consistent dependence structure across activity windows. Values from `data/analysis/modeling/shap_three_way_site_attribution.csv`, `shap_xgb_87d_by_site.csv`, and `shap_grouped_site_attribution.csv`.

<a id="table-s13"></a>
### Table S13. Projected $d_{z^2}$ Occupancies of Three Co-Centered Compositions

| Structure | State | d total (e) | $d_{z^2}$ occupancy (e) | $d_{z^2}$ share of d | $\Delta n(d_{z^2})$ clean → H (e) |
|:--:|:--:|:--:|:--:|:--:|:--:|
| Co–Cu–Fe–Zn (best-performing) | clean | 7.38 | 1.744 | 23.6% | |
| Co–Cu–Fe–Zn (best-performing) | H | 7.42 | 1.211 | 16.3% | −0.533 |
| Co–Ni–Cr–Cu (median) | clean | 7.38 | 1.551 | 21.0% | |
| Co–Ni–Cr–Cu (median) | H | 7.41 | 1.206 | 16.3% | −0.345 |
| Co–Cr–V–Zn (boundary) | clean | 7.34 | 1.739 | 23.7% | |
| Co–Cr–V–Zn (boundary) | H | 7.41 | 1.198 | 16.2% | −0.541 |

Occupancies are occupation-weighted sums over bands of band occupation times $d_{z^2}$ projection from the PROCAR files of static runs at the Γ point with ISPIN = 2 and LORBIT = 11. The d total confirms the d⁷ configuration of nine-valence-electron Co, 7.34–7.42 |e| in all six configurations. Hydrogen adsorption depletes $d_{z^2}$ by 0.35–0.54 |e| consistently, characterizing the adsorption-induced orbital occupation response.

<a id="table-s14"></a>
### Table S14. Projected $d_{z^2}$ Occupancies of Fe, Ru, Ni, Pd and Pt Centers

| Structure | $M_1$ | valence | $\Delta G_{\mathrm{H}}$ (eV) | State | d total (e) | $d_{z^2}$ occupancy (e) | $d_{z^2}$ share of d | $\Delta n(d_{z^2})$ clean → H (e) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| FeScRhV | Fe | 8 | +0.087 | clean | 6.27 | 0.994 | 15.9% | |
| FeScRhV | Fe | 8 | +0.087 | H | 6.10 | 1.063 | 17.4% | +0.069 |
| FeRhRuSc | Fe | 8 | +0.150 | clean | 6.30 | 0.990 | 15.7% | |
| FeRhRuSc | Fe | 8 | +0.150 | H | 6.43 | 1.063 | 16.5% | +0.073 |
| RuCuAuRh | Ru | 8 | −0.534 | clean | 6.01 | 1.149 | 19.1% | |
| RuCuAuRh | Ru | 8 | −0.534 | H | 6.13 | 1.024 | 16.7% | −0.125 |
| RuMnCoZn | Ru | 8 | −0.587 | clean | 6.02 | 1.125 | 18.7% | |
| RuMnCoZn | Ru | 8 | −0.587 | H | 6.08 | 1.014 | 16.7% | −0.111 |
| NiCoCuRh | Ni | 10 | +1.147 | clean | 8.35 | 1.824 | 21.8% | |
| NiCoCuRh | Ni | 10 | +1.147 | H | 8.26 | 1.492 | 18.1% | −0.332 |
| NiFeCuPd | Ni | 10 | +1.287 | clean | 8.34 | 1.819 | 21.8% | |
| NiFeCuPd | Ni | 10 | +1.287 | H | 8.28 | 1.491 | 18.0% | −0.328 |
| PdRhAgIr | Pd | 10 | +1.219 | clean | 8.53 | 1.796 | 21.1% | |
| PdRhAgIr | Pd | 10 | +1.219 | H | 8.39 | 1.541 | 18.4% | −0.255 |
| PdNiCoZn | Pd | 10 | +1.317 | clean | 8.52 | 1.808 | 21.2% | |
| PdNiCoZn | Pd | 10 | +1.317 | H | 8.39 | 1.530 | 18.2% | −0.278 |
| PtCuRhRu | Pt | 10 | +0.891 | clean | 8.16 | 1.686 | 20.7% | |
| PtCuRhRu | Pt | 10 | +0.891 | H | 8.08 | 1.426 | 17.7% | −0.260 |
| PtAuCoRh | Pt | 10 | +0.979 | clean | 8.16 | 1.690 | 20.7% | |
| PtAuCoRh | Pt | 10 | +0.979 | H | 8.08 | 1.416 | 17.5% | −0.274 |

Same protocol as Table S13 (static runs, Γ point, ISPIN = 2, LORBIT = 11). The d total confirms the d⁶ configuration of eight-valence Fe and Ru and the d⁸ configuration of ten-valence Ni, Pd, and Pt. Fe and Ru carry nearly equal $d_{z^2}$ yet bind in opposite directions, and Ni/Pd/Pt carry a nearly filled $d_{z^2}$ yet all underbind, showing different binding responses at similar $d_{z^2}$ occupations.

---

<a id="table-s15"></a>
### Table S15. Composition-Selection Criteria for Multimetallic M–N₄/G Catalysts

The near-thermoneutral target is $|\Delta G_{\mathrm{H}}|\leq0.1$ eV.

| Design choice | Main evidence | Implication for screening |
|---|---|---|
| Prioritize Co, Rh or Ir at $M_1$ | Nine-valence-electron metals show 3.78-fold enrichment among predicted near-thermoneutral candidates | Prioritize Co for all-3d combinations; orbital-response evidence comes from three Co compositions |
| Include Sc among $M_1$ candidates | Five of eight DFT validation compositions meet the target window | Retain candidates beyond nine-valence-electron metals; charge and geometry support stronger ionic character |
| Combine multiple elemental properties of $M_1$ | The five-descriptor model gives prediction $R^2=0.8927$ over 110 held-out compositions | Combine valence-electron count with other elemental properties to identify overall adsorption free-energy trends |
| Compare Zn-containing and other auxiliary-metal combinations | Co-centered Zn-containing compositions show a higher predicted fraction of qualifying compositions; Zn-free combinations also yield compositions meeting the DFT criterion | Prioritize Zn-containing combinations while keeping other compositions in the search space |

---
