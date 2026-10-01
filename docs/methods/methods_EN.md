# Methods

Version: v1
Date: 2026-08-11
Status: Methods revised to follow the established R1–R7 writing style; EN/CN synced
Last modified: 2026-09-27

---

## English Version

### Computational Details

Density functional calculations used the projector augmented-wave method in VASP [32] and the Perdew–Burke–Ernzerhof functional within the generalized gradient approximation [33]. Dispersion interactions were treated with DFT-D3 using Becke–Johnson damping [34,35]. Archived relaxation batches used spin polarization and Γ-point sampling, with electronic and force convergence criteria of $10^{-5}$ eV and 0.02 eV/Å, respectively. Atomic positions were relaxed at a fixed cell, using the default plane-wave cutoff, reported as 400 eV in verified outputs.

The clean model contained 48 C atoms, 16 N atoms and four distinct metals, totaling 68 atoms. Each metal occupied an N₄ coordination site in the $(M_1,M_2,M_3,M_4)$–N₄/G platform. Initial lattice parameters were $a=b=14.76$ Å and $\gamma=120^\circ$, with a cell length of 20 Å normal to the substrate. H was initialized atop $M_1$ to form the 69-atom adsorption model; clean and H-adsorbed structures were relaxed separately.

Hydrogen adsorption free energies were evaluated using the computational hydrogen electrode model of Nørskov and coworkers [36],

$$\Delta G_{\mathrm{H}} = E_{\mathrm{H}} - E_{\mathrm{clean}} - \tfrac{1}{2}E_{\mathrm{H}_2} + \Delta\text{ZPE} - T\Delta S$$

Here, $E_{\mathrm{H}}$, $E_{\mathrm{clean}}$ and $E_{\mathrm{H}_2}$ denote the DFT energies of the H-adsorbed state, clean state and hydrogen reference, respectively. The terms $\Delta\text{ZPE}$ and $T\Delta S$ account for zero-point energy and entropy corrections. Adsorption free energies retained the archived conversion $\Delta G_{\mathrm{H}}=E_{\mathrm{H}}-E_{\mathrm{clean}}+3.481$ eV, with the constant combining the hydrogen reference and thermodynamic correction. Candidates were assessed using the corrected $|\Delta G_{\mathrm{H}}|$, and charge transfer in the best candidate was examined by Bader analysis [37].

### Composition Space and Exchange Symmetry

Selecting four distinct elements from 17 transition metals generated $C(17,4) \times 4! = 57120$ ordered arrangements. The energetic effect of $M_3\leftrightarrow M_4$ exchange was examined in 124 training-set pairs calculated with matched DFT settings. These pairs came from iterations 0 and 1 and covered 14 metals; their adsorption centers excluded early transition metals. The swap deviation was defined as

$$\Delta E_{34} = |\Delta G_{\mathrm{H}}(M_3,M_4) - \Delta G_{\mathrm{H}}(M_4,M_3)|$$

Swap deviations were grouped into values below 10 meV, 10–50 meV and above 50 meV. Additional calculations covered 56 targeted pairs with deviations above 10 meV and 40 independently sampled early-transition-metal pairs (SI Section S1.3). The near-thermoneutral group comprised 15 pairs with at least one arrangement satisfying $|\Delta G_{\mathrm{H}}|\leq0.1$ eV; the remaining 109 pairs formed the comparison group. Space groups were assigned using spglib [38] with a tolerance of 0.01 Å. The approximate remote-site energy permutation relation grouped arrangements into 28560 compositions, each identified by a physical ID with fixed $M_1$, $M_2$ and unordered $\{M_3,M_4\}$.

### DFT Dataset Construction

Dataset construction combined diversity sampling and composition deduplication to limit redundant information [39]. The initial 500 structures were selected through a mixed sampling strategy. KMeans-centroid diversity sampling in the 77-dimensional Phase-D feature space contributed 50%, Pt-like d-band and electronegativity sampling contributed 30%, and uniform random sampling contributed 20%. This feature space determined initial selections; model training used the unified descriptors defined below.

One adaptive acquisition round used the initial ensemble to score uncomputed structures in the original space of 57120 ordered arrangements. Its two-term function was $\mathrm{score}=\alpha U_{\mathrm{norm}}+(1-\alpha)(1-|\mu|_{\mathrm{norm}})$, where $\mu$ is the predicted hydrogen adsorption free energy and $U$ is the total predictive standard deviation. The quantities $U$ and $|\mu|$ were min–max normalized over the remaining candidates, with $\alpha=0.637$. The round selected 100 structures, of which 98 yielded valid DFT results. The initial 500 records, these 98 results and 608 historical records provided 1206 calculation records. Within each physical ID, the minimum clean-state and H-state energies were selected separately to construct the adsorption free energy. Deduplication yielded 995 training compositions, covering 3.48% of the composition space.

### Symmetric Descriptors and Feature Standardization

Removing nine derived columns from the 96-descriptor set produced the common 87-dimensional input, comprising 16 $M_1$, 13 $M_2$, six $M_{34}$ and 52 global features. Removed columns represented property ratios, size mismatch, strain and dipole proxies (SI Section S1.4; Table S2). Remote sites were represented by the means and absolute differences of electronegativity, d-electron count and atomic radius,

$$g(\mathbf{x}_{M_3},\mathbf{x}_{M_4}) = \left[\mathrm{mean}(\mathbf{x}_{M_3},\mathbf{x}_{M_4}),\; \left|\mathbf{x}_{M_3}-\mathbf{x}_{M_4}\right|\right] \in \mathbb{R}^{6}$$

Here, $\mathbf{x}_M$ contains these three attributes of metal $M$, while global statistics summarize the unordered set of four metals. This input makes the symmetry-aware ensemble, flat MLP and tree models invariant under remote-site exchange,

$$f_\theta(M_1,M_2,M_3,M_4) = f_\theta(M_1,M_2,M_4,M_3) \quad \forall\,\theta$$

Features were standardized using the z-score transformation,

$$\tilde{x}_j = \frac{x_j - \bar{x}_j}{s_j}$$

The mean $\bar{x}_j$ and standard deviation $s_j$ were calculated from the training partition alone to prevent information leakage [40]. In five-fold cross-validation, each fold contained 796 training and 199 validation compositions, both standardized with that fold's training statistics. The final model used statistics from all 995 training compositions for held-out and full-space predictions.

### Ensemble Model and Training

The symmetry-aware ensemble comprised 10 neural-network constituent models, each encoding $M_1$, $M_2$ and $M_{34}$ features into separate 32-dimensional representations. These representations were combined with 52 global features (Figure 3a). The fusion network contained two hidden layers $[128, 64]$, ReLU activations and a dropout probability of 0.1, giving

$$\mathbf{z} = \mathrm{trunk}\left(\left[\mathbf{e}_1(\mathbf{x}^{(M_1)});\; \mathbf{e}_2(\mathbf{x}^{(M_2)});\; \mathbf{e}_{34}(g(\mathbf{x}^{(M_3)},\mathbf{x}^{(M_4)}));\; \mathbf{x}^{(\mathrm{global})}\right]\right)$$

The heteroscedastic output layer $(\mu, \log\sigma^2) = \mathrm{head}(\mathbf{z})$ predicted the mean and log-variance of hydrogen adsorption free energy and was trained with Gaussian negative log-likelihood loss [41]. Constituent models used different random seeds and bootstrap samples, with Adam optimization at a learning rate of $10^{-3}$ and a batch size of 64. Training stopped after 30 consecutive epochs without improvement in validation loss.

The ensemble prediction and uncertainty components were calculated from the $K=10$ constituent models,

$$\hat{y}(\mathbf{x}) = \frac{1}{K}\sum_{k=1}^{K}\mu_k(\mathbf{x}), \qquad \sigma_{\mathrm{ale}}^2(\mathbf{x}) = \frac{1}{K}\sum_{k=1}^{K}\sigma_k^2(\mathbf{x}), \qquad \sigma_{\mathrm{epi}}^2(\mathbf{x}) = \frac{1}{K}\sum_{k=1}^{K}\left[\mu_k(\mathbf{x})-\hat{y}(\mathbf{x})\right]^2$$

Here, $\sigma_{\mathrm{ale}}^2$ is the mean predicted variance, and $\sigma_{\mathrm{epi}}^2$ is the population variance of the constituent-model means. Ensemble spread was evaluated against held-out errors to assess error-ranking performance [42]. Full-space screening used the epistemic standard deviation $\sigma_{\mathrm{epi}}$, whereas acquisition scores used the total predictive standard deviation $U=\sqrt{\sigma_{\mathrm{ale}}^2+\sigma_{\mathrm{epi}}^2}$.

### Model Comparisons and Performance Evaluation

Model selection used five-fold cross-validation with shuffled KFold splits and random seed 42. The flat MLP shared the 87-dimensional input and training hyperparameters with the symmetry-aware ensemble. Its hidden layers $[256, 128, 64]$ contained 64706 parameters per constituent model, compared with 32418 for the symmetry-aware model. This comparison evaluated site grouping together with the corresponding network structures. XGBoost [43] was tuned on the same features and five-fold splits, using 300 trees, depth 7 and learning rate 0.04. Row and column subsampling fractions were both 0.8, and the minimum child weight was 5. Point-prediction benchmarks also included ten traditional machine-learning models (SI Table S4).

A 2$\times$2 comparison combined symmetry-aware or flat architectures with NLL or MSE loss. All four configurations shared the five-fold splits and training hyperparameters, with 10 constituent models per fold. MSE models used a single output neuron and the loss

$$L_\text{MSE} = \frac{1}{N} \sum_i (y_i - \mu_i)^2$$

Here, $y_i$ is the DFT hydrogen adsorption free energy, $\mu_i$ is its prediction, and $N$ is the sample size.

Held-out evaluation was separated from model selection, distinguishing prediction on unseen compositions from extrapolation to new chemical domains [44]. Evaluation outside the training set used 110 structures with no physical ID overlap with the 995 training compositions. The initial 30 included 12 highly ranked, high-confidence candidates, six promising but uncertain candidates, six boundary candidates and six blind-spot candidates. The subsequent 80 comprised 40 in-distribution and 40 out-of-distribution structures, which defined the stratified comparison (SI Figure S4). Absolute prediction errors were paired by structure when comparing models.

### Chemical Prior and Retrospective Screening

The chemical prior used the d-electron count and electronegativity of the adsorption center. Compositions with $|\Delta G_{\mathrm{H}}|\leq0.2$ eV were classified as meeting the criterion within the current training data. The fraction meeting this criterion in each d-electron group was divided by the largest group fraction to obtain $\pi_{n_d}$. The mean and standard deviation of adsorption-center electronegativity among qualifying compositions defined the center and width of the Gaussian component $\pi_\chi$. The components were combined as

$$P_{\mathrm{phys}} = 0.6\,\pi_{n_d} + 0.4\,\pi_\chi$$

The prior distributions in Figure 4a,b were calibrated using all 995 compositions. Success fractions were compared between the full set and compositions at or above the 80th-percentile score, including ties at the threshold. During retrospective screening, the prior was recalibrated each round using DFT results from already selected compositions.

The retrospective comparison evaluated discovery under a shared selection budget [45], using a fixed set of 995 DFT compositions and the score $\mathrm{score}=\alpha_tU_{\mathrm{norm}}+\beta_tE+\gamma_tP_{\mathrm{phys}}$, where $E=1-|\mu|_{\mathrm{norm}}$. The chemical-prior strategy used $\gamma_t=0.3e^{-t/3}$, and the pure-UCB strategy used $\gamma_t=0$, with shared $\alpha_t$ and $\beta_t$ schedules (SI Section S1.6; Figure S7a). Each repeat shared random seeds, an initial set of 200 compositions and the remaining candidates. Both strategies used five constituent models and total predictive standard deviation. Each round selected 100 compositions, followed by model and prior updates using accumulated DFT results; five rounds yielded 700 compositions. Candidate DFT results entered updates only after selection. Eight paired repeats counted final qualifying compositions, including the initial samples, at the 0.2 and 0.1 eV thresholds.

### DFT Validation of Candidate Materials

The final model predicted hydrogen adsorption free energies and epistemic standard deviations for all 28560 compositions. Screening successively retained compositions with $|\Delta G_{\mathrm{H}}|\leq0.1$ eV, epistemic standard deviations no greater than the full-space median, and all four metals within Sc–Zn. Candidates were ranked by predicted absolute free energy (Figure 5; SI Table S1).

The prespecified validation set contained 40 compositions, including 20 all-3d candidates selected for group A using the dual-model mean_abs score. Their predicted absolute free energies were below 0.1 eV. All had Co adsorption centers and contained Zn, and all belonged to the final set of 137 all-3d compositions. Group B selected ten candidates with the lowest predicted absolute free energies among compositions containing Pt, Pd, Ir or Rh. Group C sampled the prediction set after excluding reliable candidates, drawing three, four and three structures from [0, 0.1), [0.1, 0.3) and [0.3, 1.0) eV, respectively. Each stratum used random seed 42. All 40 compositions underwent separate clean- and H-state relaxations from initial geometries. After fixing the final model, DFT calculations covered the 20 additional compositions in its all-3d top-30 that were absent from group A, yielding 60 validation results.

Validation compositions were matched to the training set by physical ID. Co–Cr–Mn–Zn and Co–Mn–Ni–Zn had training records, so evaluation outside training used the remaining 38 prespecified compositions. All 60 results are retained in SI Tables S10 and S11. Co–Ir–Cr–Ti also belonged to the 110-structure held-out set; validation memberships and overlaps are documented in SI Section S1.7.

### Elemental-Attribute Model and SHAP Analysis

The five-attribute adsorption-center model used valence-electron count, atomic radius, d-band center, lattice constant and electronegativity, with the prediction

$$\hat{y} = \beta_0 + \sum_{j=1}^{5}\beta_j d_j + \sum_{1\le j\le k\le 5}\beta_{jk}\, d_j d_k$$

Here, $d_j$ denotes one of the five elemental attributes. Scikit-learn PolynomialFeatures generated 20 degree-two terms, fitted by Ridge regression with $\alpha=0.1$. The model was trained on 995 compositions and evaluated on the 110 held-out compositions (Figure 7b).

SHAP analysis [46] compared XGBoost, the symmetry-aware ensemble and the flat MLP on the same 87-dimensional input. XGBoost was retrained on this input and analyzed with TreeExplainer [47] without background data; its comparison used archived SHAP summaries. DeepExplainer analyzed the mean predictions of the ten constituent models for each neural network. Evaluation used CPU forward passes retaining training normalization, with the same 100 training structures as background and all 995 compositions explained. Mean absolute SHAP values were summed by column index within the 16-feature $M_1$, 13-feature $M_2$, six-feature $M_{34}$ and 52-feature global blocks (SI Section S1.9; Figure S9).

The adsorption-free-energy-stratified SHAP analysis used all 995 compositions as background. It compared 122 compositions satisfying $|\Delta G_{\mathrm{H}}|\leq0.1$ eV with the remaining 873 compositions, using Wilcoxon rank-sum tests for individual-feature SHAP distributions.

### Site-Resolved Expressions and Electronic Structure Analysis

The site-resolved empirical expression fitted composition differences using elemental attributes selected with SISSO [48] and explicit site assignments. Cross-validation grouped records by composition, calculated the MAE separately for each adsorption-center element, and averaged these values with equal weights. The fit to all 995 compositions and historical supplementary evaluation are documented separately in SI Section S1.12.

Electronic structure analysis compared archived clean/H pairs through occupation-weighted orbital projections, Bader charges and LOBSTER bonding projections [49,50]. Charge and bonding responses were defined as adsorbed minus clean values; fixed-geometry composition comparisons used total energies excluding the entropy term. Each comparison retained its calculation-batch and projection-basis assignments. Collinear magnetic states were aligned allowing global spin reversal, with electronic-state compatibility assessed through the energy, magnetic-moment and occupation criteria in SI Section S1.12. Reference geometries, state-selection sensitivity and the prospective candidate check are detailed in SI Section S1.12 and Figures S12–S14. Adsorption-free-energy evaluation retained the original DFT data, with subsequent electronic-state records archived separately.

### Statistical Analysis

Cross-validation MAEs were compared by fold-paired t-tests between architectures at fixed loss and between losses at fixed architecture. The 20 fold-level MAEs also entered a two-way ANOVA containing architecture, loss and their interaction, with fold as a blocking factor. The 110-structure held-out comparison used paired t-tests on structure-level absolute errors. Retrospective screening compared terminal qualifying counts across eight paired repeats, reporting means and sample standard deviations.

Mean absolute free energies were compared by Welch t-test between 18 training-unseen group-A compositions and ten group-B compositions; the full A20/B10 comparison was reported separately. Descriptive 95% Student t intervals were calculated across the selected compositions. Spearman tests evaluated predicted versus DFT free energies and epistemic standard deviation versus absolute prediction error. A Mann–Whitney test compared swap deviations between the 15 near-thermoneutral pairs and the remaining 109 pairs.

Enrichment of Co, Rh and Ir centers among 2428 predicted qualifying compositions was assessed by a binomial proportion test against the full-space baseline of 5040/28560. Fisher's exact test compared predicted qualifying fractions for Co-centered compositions with and without Zn (Figure 5d). Pearson analysis assessed the relationship between prior scores and binary qualification labels. The t, Spearman, Mann–Whitney and Fisher tests were two-sided; the binomial test assessed enrichment of nine-valence-electron centers. The three primary hypotheses concerning architecture, acquisition strategy and all-3d versus precious-metal groups were tested without multiple-testing correction. Uncorrected p values for the SHAP group comparisons were reported as exploratory.

---
