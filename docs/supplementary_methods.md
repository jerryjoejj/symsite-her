# NC supplementary methods

<a id="s1-1"></a>
### S1.1. Material Structure and Metal Compositions

In $(M_1,M_2,M_3,M_4)$–N₄/G, $M_1$ is the H adsorption center, $M_2$ the neighboring site, and $M_3$ and $M_4$ the remote sites. The set of candidate metals includes Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu and Zn from the 3d series. The 4d elements are Ru, Rh, Pd and Ag; the 5d elements are Ir, Pt and Au. Initial structural parameters are specified in the main-text Computational Details; geometries before and after hydrogen adsorption are shown in Figure S1.

<a id="s1-2"></a>
### S1.2. Composition Space and Physical IDs

Arranging 17 metals over four sites without repetition gives 57120 ordered structures. Fixing $M_1$ and $M_2$ and sorting the remote element symbols lexicographically defines physical ID $(M_1,M_2,\{M_3,M_4\})$, yielding $17\times16\times C(15,2)=28560$ compositions. Relaxed structures have space group P1 at 0.01 Å tolerance; remote-site grouping uses approximate energetic exchange symmetry.

The 124 training exchange pairs have a median deviation of 5.6 meV. Fractions below 10 meV, between 10 and 50 meV, and above 50 meV are 54.8%, 16.9% and 28.2%, respectively. Medians for Ir, Pt, Au and Fe centers are 0.81, 0.99, 24.6 and 84.9 meV. Co, Rh and Ir centers collectively have a median of 1.81 meV, with 70.8% below 10 meV. The 15 near-thermoneutral pairs and remaining 109 pairs have medians of 3.4 and 6.7 meV, respectively (Mann–Whitney $p=0.99$).

<a id="s1-3"></a>
### S1.3. DFT Verification of Remote-Site Exchange Symmetry

The first calculation batch reexamined 56 of the 124 training exchange pairs with deviations above 10 meV, comprising 112 structures and 224 single-point calculations. Initial deviations fell within 10–50, 50–100, 100–200 and above 200 meV for 21, eight, 22 and five pairs, respectively. Calculations retained the original INCAR settings, with EDIFF = $10^{-5}$ eV and EDIFFG = −0.02 eV/Å. Deviations remained above 10 meV for 47/56 pairs. The Fe-center median decreased from 145 to 101 meV, with 22/26 pairs still above the threshold (Table S6; Figure S6).

The second calculation batch used a random sample of eight pairs each with Sc, Ti, V, Cr and Mn centers, totaling 40 pairs, 80 structures and 160 single-point calculations. Calculations used the same settings; 29/40 pairs fell below 10 meV, with a median of 2.9 meV. Four of eight Sc-centered pairs fell below the threshold. Auxiliary Fe or Au occurred in six of the 11 pairs above the threshold and eight of the 29 below it. Results by center element are given in Table S7.

<a id="s1-4"></a>
### S1.4. Site Features and Exchange Invariance

Removing nine derived columns from 96 descriptors gives the 87-dimensional input listed in Table S2. The $M_1$ block contains 13 base elemental attributes plus H-bond polarity, d-band filling and work-function deviation; $M_2$ contains the 13 base attributes. The $M_{34}$ block contains remote-site means and absolute differences of electronegativity, d-electron count and atomic radius. The global block contains weighted mean, standard deviation, range and $M_1$ deviation for each base attribute, totaling 52 features. The weighted mean assigns a weight of 0.5 to $M_1$.

Let $\tau$ denote $M_3\leftrightarrow M_4$ exchange and $\mathbf{x}(s)$ the 87-dimensional input for structure $s$. Remote means, absolute differences and global statistics are invariant under exchange, so $\mathbf{x}(\tau s)=\mathbf{x}(s)$ and any model $f_\theta$ satisfies

$$f_\theta(\mathbf{x}(s)) = f_\theta(\mathbf{x}(\tau s)).$$

All model comparisons use this input. Feature standardization uses training-partition statistics alone, as specified in the main text.

<a id="s1-5"></a>
### S1.5. Ensemble Models, Training and Baseline Comparisons

Network architectures and optimizer settings are specified in the main text. Training used weight decay $10^{-5}$, a maximum of 300 epochs and early stopping after 30 epochs without validation-loss improvement. Constituent models used different random seeds and bootstrap samples; complete seed lists are provided in the training scripts. Five-fold splits used shuffled KFold with seed 42. LightGBM used a hyperparameter search over 576 combinations, and GPR cross-validation used all 995 compositions. Results for 13 models and the architecture–loss ablation are listed in Tables S4 and S3, respectively.

The 110 held-out compositions comprise the initial 30, followed by 40 in-distribution and 40 out-of-distribution compositions. Mean absolute-error differences, calculated as symmetry-aware minus flat model errors, are −0.01315, −0.00470 and −0.04224 eV for these groups, respectively (Figure S4). Error ranking by epistemic standard deviation is shown in Figure S2.

Machine learning and statistical analyses used NumPy, PyTorch, scikit-learn, XGBoost and SHAP. Software versions, splits and random seeds are specified in environment files and analysis scripts in the code repository.

<a id="s1-6"></a>
### S1.6. DFT Sampling and Chemically Guided Selection

<a id="s1-6-1"></a>
#### S1.6.1. Initial and adaptive DFT sampling

The initial 500 structures were selected from 57120 ordered arrangements. Standardized 77-dimensional Phase-D features were clustered using KMeans with $k=250$, random_state = 42 and n_init = 10. Selecting the structure nearest each centroid gave 250 structures. The next 150 were selected from the remainder using $s=1/(1+|\varepsilon_d-(-2.25)|)+1/(1+|\chi-2.25|)$. Here, $\varepsilon_d$ and $\chi$ are the d-band center and electronegativity used in the Pt-derived prior. NumPy RandomState with seed 42 then selected 100 structures randomly from the remaining candidates. Qualifying counts at $|\Delta G_{\mathrm{H}}|\leq0.2$ eV were 55/250, 8/150 and 21/100 for the three strategies, respectively. Repeating the initial selection reproduced 427/500 ordered structures, with differences associated with the sensitivity of KMeans to the software version; subsequent analyses used the complete saved selection list.

Iteration 1 used the main-text two-term score with exploration and exploitation weights of 0.637 and 0.363. The initial ensemble predicted free energy and total predictive standard deviation for remaining candidates, selecting 100 structures that yielded 98 valid DFT labels. The three-term chemical-prior score was used in the separate retrospective comparison below.

<a id="s1-6-2"></a>
#### S1.6.2. Chemical-prior calibration and paired retrospective comparison

All 995 compositions were used for prior calibration in Figure 4a,b. Qualifying fractions at $|\Delta G_{\mathrm{H}}|\leq0.2$ eV for d¹, d², d³, d⁵, d⁶, d⁷, d⁸, d⁹ and d¹⁰ groups were 0.61, 0.00, 0.02, 0.61, 0.12, 0.43, 0.35, 0.00 and 0.02. Dividing by the largest value, 0.61, gives $\pi_{n_d}$; the Gaussian $\pi_\chi$ has center 1.818 and width 0.306. Co and Ir have d⁷ counts and Rh d⁸; all three have nine valence electrons.

In each of eight paired retrospective repeats, the 995 compositions were randomly permuted with seed $42+r$, assigning the first 200 as shared initial data and the remaining 795 as candidates. Both strategies used five constituent models and total predictive standard deviation, selecting 100 compositions per round for five rounds. Priors were recalibrated only from acquired labels. Chemical-prior guidance used $\gamma_t=0.3e^{-t/3}$ and pure UCB used $\gamma_t=0$, with identical remaining weight schedules (Figure S7a). Final numbers of qualifying compositions included the initial samples at thresholds of 0.2 and 0.1 eV. The minimum absolute free energy obtained up to each round was summed across rounds to compare how rapidly the strategies identified favorable candidates. Results for each round and paired statistics are given in Table S5 and Figure S7.

<a id="s1-7"></a>
### S1.7. DFT Validation of Candidate Compositions

Prespecified groups A, B and C contained 20 all-3d, ten precious-metal-containing and ten stratified candidates, respectively. Group A followed the dual-model mean_abs ranking before the final model update; all members belonged to the final 137 all-3d candidates. Physical ID matching identified Co–Cr–Mn–Zn and Co–Mn–Ni–Zn as training compositions, leaving 38 for generalization evaluation. Group-B Co–Ir–Cr–Ti also occurred among the 110 held-out compositions; set memberships are listed in Tables S9–S11.

Predicted and DFT free energies for the 38 compositions absent from the training set were compared using Spearman correlation and RMSE. The mean prediction-minus-DFT residual was −0.0055 eV, with a 95% confidence interval of [−0.039, +0.028] eV calculated across compositions using the Student t distribution. A two-sided Welch test of mean absolute free energies for A18/B10 gave $t(25.3)=0.431$ and $p=0.670$; the complete A20/B10 comparison gave $p=0.787$. Group-wise qualifying counts and errors are shown in Figure S5 and Table S11.

DFT calculations for twenty additional compositions completed validation of the final model's all-3d top-30 candidates, ten of which were already in group A. Co–Mn–Ni–Zn was a training composition; 19 of the other 29 satisfied $|\Delta G_{\mathrm{H}}|\leq0.1$ eV, compared with 20 of all 30. The two selection protocols yielded 35 qualifying compositions among 60, or 33 among the 58 compositions absent from the training set (Tables S10 and S11).

<a id="s1-8"></a>
### S1.8. Electronic-Structure Analysis of Adsorption Centers

Projected metal-orbital populations were calculated by weighting PROCAR projections from spin-polarized static calculations by band occupations. State-resolved results for three Co-centered compositions and ten Fe-, Ru-, Ni-, Pd- or Pt-centered controls are listed in Tables S13 and S14.

The Sc/Co comparison included seven Sc and three Co compositions, each with pristine and adsorption structures. Occupied d-state centers were referenced to the Fermi level of each structure, and metal protrusion was measured relative to the local N₄ plane. Mean occupied d-state centers in adsorption structures were −3.68 and −2.74 eV for Sc and Co, respectively. Mean adsorption-induced metal electron losses were 0.031 and 0.036 $|e|$, respectively (Mann–Whitney $p=0.833$). Charge, orbital-population and geometry results are shown in Figure S10. Sc–Mn–Cr–Fe was excluded from electronic-structure comparison because its adsorption-structure static calculation did not reproduce the relaxed magnetic state; its original DFT adsorption free energy was retained.

Figure 6d and the subsequent Sc/Co comparison used different calculation batches. The Co electron losses for Co–Cu–Fe–Zn were 0.0343 and 0.0361 $|e|$, respectively; the later batch averaged 0.036 $|e|$ across three Co compositions. Extended Co coordination-response analysis is described in S1.10.

<a id="s1-9"></a>
### S1.9. SHAP Attribution of Model Features

SHAP comparisons used the common 87-dimensional input. Both neural networks shared NLL loss, ten constituent models, seed 42 and an 80/20 split. DeepExplainer analyzed mean predictions for all 995 compositions using the same 100 training structures as background. Calculations retained training normalization; repeating the SHAP calculation with fixed model weights gave a maximum additivity residual below $4\times10^{-7}$ eV. XGBoost attribution comparisons used SHAP summaries generated with TreeExplainer. Mean absolute SHAP values were summed by column index within the $M_1$, $M_2$, $M_{34}$ and global blocks.

In the feature-set comparison, XGBoost five-fold MAEs were 0.1073 and 0.1075 eV for 87- and 96-dimensional inputs, with $R^2=0.9402$ and 0.9403, respectively. Removed columns are listed in Table S2 and site attributions in Table S12. The neural-network comparison changes both site grouping and parameter count; tree and neural models additionally use different losses and explainers.

The free-energy-stratified analysis used all 995 compositions as background, comparing 122 compositions satisfying $|\Delta G_{\mathrm{H}}|\leq0.1$ eV with the remaining 873. Site-attribution shares differed by less than 3.1 percentage points (Figure S9).


<a id="s1-10"></a>
### S1.10. Site-Resolved Expressions and Coordination Responses

<a id="s1-10-1"></a>
#### S1.10.1. Empirical expression and evaluation

The main-text $D_r$ and $D_e$ correspond to expression D01, with $D_e$ denoted $D_{e1}$ in analysis files. Radii and d-electron counts follow the elemental-attribute table; the Co d-electron count is seven. Scaling by 100 pm makes both components dimensionless; the intercept and two coefficients for each adsorption-center element have units of eV. The center radius is constant for a given adsorption-center element and can be absorbed into the intercept.

Composition-grouped cross-validation fitted each fold separately, selecting Ridge regularization within each outer training partition and averaging center-specific MAEs with equal weights. The 110 historical supplementary records were evaluated with a fixed expression fitted to all 995 training compositions at $\alpha=1$. This version, All_De1_original, contains 17 intercepts and 34 coefficients; complete coefficients and input conventions are provided in the coefficient file `models/descriptors/frozen_expressions.json`.

<a id="s1-10-2"></a>
#### S1.10.2. Charge and bonding responses

Extended orbital analysis comprised 129 pristine/adsorption pairs for 107 Co physical IDs. Different batches for the same composition were retained separately, with adsorption-induced changes evaluated within each batch. Orbital axes follow calculation coordinates; projected orbital populations, Bader populations and ICOHP describe orbital, charge and bonding responses, respectively.

Ligand charge changes are defined as $\Delta q_i=q_i^{\mathrm{H}}-q_i^{\mathrm{clean}}$, where $q_i=Z_i^{\mathrm{val}}-N_i^{\mathrm{Bader}}$. The signed sum over Co and four N ligands measures net charge change, while $\sum_i\lvert\Delta q_i\rvert$ measures total atomic redistribution.

LOBSTER comparisons used fixed-geometry states of four compositions, each containing four Co–N bonds. Negative ICOHP denotes net bonding; $B=\sum_{\mathrm{Co-N}}(\mathrm{ICOHP}_{\mathrm{H}}-\mathrm{ICOHP}_{\mathrm{clean}})>0$ denotes adsorption-induced Co–N weakening. Orbital responses were calculated by subtracting pristine values from adsorption values, then Cr results from Fe results with the neighboring Cu or Mn and one remote Zn unchanged. Standard and alternative bases were compared separately, retaining residual differences between the summed orbital contributions and total ICOHP. The Cu–Cr pristine structure has a different electronic state from the reference used to obtain the original DFT label.

Two sets of reference geometries comprised the pristine and adsorption structures of Co–Cu–Cr–Zn and Co–Cu–Fe–Zn; the Mn/Cu comparison used the latter set. Calculations fixed nuclear coordinates and obtained $J$ from total-energy differences excluding entropy. Calculation batches used VASP 6.3.2 or 6.4.3, with inputs, PAW data and electronic-state checks recorded for each batch.

<a id="s1-10-3"></a>
#### S1.10.3. Capacity-balance candidate and Ni comparison

The candidate capacity-balance descriptor uses the tabulated d-electron counts at four sites. With $h(n)=n(10-n)$,

$$
Q_{\mathrm{bal}}=
\left[
\frac{h(n_{d,3})+h(n_{d,4})}{2}
-\frac{h(n_{d,1})+h(n_{d,2})}{2}
\right]^2.
$$

This parameter-free quantity is invariant under remote-site exchange. In the Cu- and Mn-neighbor Cr→Fe comparisons used to formulate it, $Q_{\mathrm{bal}}$ changed in the same direction as $B$ and in the opposite direction to signed $\Delta G_{\mathrm{H}}$. The resulting Cu-neighbor prediction was $B_{\mathrm{Ni}}>B_{\mathrm{Cr}}>B_{\mathrm{Fe}}$.

The static calculation for the Co–Cu–Ni–Zn adsorption structure reproduced the reference result at identical atomic positions, whereas the pristine energy without entropy changed by −31.32533 meV. After allowing global spin reversal, the difference in total cell moment and the maximum difference in local metal moments remained 0.1078 and 0.509 $\mu_{\mathrm{B}}$, respectively. The electronic-state compatibility criteria were not met. This branch gave $B_{\mathrm{Ni}}=0.26202$ eV, below $B_{\mathrm{Cr}}=0.44179$ eV and $B_{\mathrm{Fe}}=0.35268$ eV, reversing the prediction (Figure S14a).

<a id="s1-10-4"></a>
#### S1.10.4. Fixed geometry and electronic-state sensitivity

The two Cu-neighbor geometries gave $J=+78.70876$ and +54.55077 meV. The prespecified comparison threshold was 10 meV; calculations at both reference geometries used fixed atomic positions, with separate coordinates for the pristine and adsorption structures. Magnetic-sign classification for Cu compositions used remote Cr/Fe as the reference, with electronic-branch compatibility checked separately.

At the common geometry, restarting the MnFe pristine structure from its original wavefunction with tighter electronic convergence lowered the energy without entropy by 168.80494 meV. The Co moment changed from −0.531 to +0.703 $\mu_{\mathrm{B}}$. Holding the other seven endpoints fixed changed $J_{\mathrm{Mn}}$ from −126.41696 to +42.38798 meV and $I=J_{\mathrm{Mn}}-J_{\mathrm{Cu}}$ from −180.96773 to −12.16279 meV. These changes measure sensitivity to restarting and electronic-state selection (Figure S14b).

Unconstrained noncollinear calculations without spin–orbit coupling further lowered the energy by 1.37447 meV, exceeding the prespecified 1 meV compatibility tolerance. Checks of local metal and total cell magnetic moments and magnetization density satisfied the specified criteria. XML occupations retained only four decimal places, preventing conclusive checks of the electron count and occupations of the highest eight bands. Higher-precision occupation data from PROCAR met the numerical thresholds, but full compatibility was not established.
