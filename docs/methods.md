# NC manuscript methods

### Computational Details

DFT calculations were performed in VASP using the projector augmented-wave method [38], the Perdew–Burke–Ernzerhof functional [39] and DFT-D3 dispersion corrections with Becke–Johnson damping [40,41]. Structural relaxations used spin polarization and Γ-point sampling, with electronic and force convergence thresholds of $10^{-5}$ eV and 0.02 eV/Å, respectively. The plane-wave energy cutoff was set to 400 eV; atomic positions were relaxed with the cell fixed.

The pristine $(M_1,M_2,M_3,M_4)$–N₄/G structure contained 48 C atoms, 16 N atoms and four distinct metals occupying N₄ sites, totaling 68 atoms. Initial lattice parameters were $a=b=14.76$ Å and $\gamma=120^\circ$, with a cell length of 20 Å normal to the substrate. H was initially placed at the top site of $M_1$ to form the 69-atom adsorption structure; both structures were relaxed separately.

Hydrogen adsorption free energies were calculated using the computational hydrogen electrode model of Nørskov and coworkers [42],

$$\Delta G_{\mathrm{H}} = E_{\mathrm{H}} - E_{\mathrm{clean}} - \tfrac{1}{2}E_{\mathrm{H}_2} + \Delta\text{ZPE} - T\Delta S$$

Here, $E_{\mathrm{H}}$, $E_{\mathrm{clean}}$ and $E_{\mathrm{H}_2}$ are the DFT energies of the adsorption structure, pristine structure and hydrogen molecule. The terms $\Delta\text{ZPE}$ and $T\Delta S$ are zero-point energy and entropy corrections. Charge transfer was evaluated by Bader analysis [43].

### Composition Space and Exchange Symmetry

Selecting four distinct elements from 17 transition metals yielded $C(17,4) \times 4! = 57120$ ordered arrangements. Remote-site exchange was tested using 124 pairs from iterations 0 and 1, covering 14 metals but excluding early-transition-metal adsorption centers. Paired calculations used identical DFT settings, with exchange deviation defined as

$$\Delta E_{34} = |\Delta G_{\mathrm{H}}(M_3,M_4) - \Delta G_{\mathrm{H}}(M_4,M_3)|$$

Deviations were grouped as below 10 meV, 10–50 meV and above 50 meV. Further calculations reexamined 56 pairs with deviations above 10 meV and tested 40 independently sampled early-transition-metal pairs (SI Section S1.3). The near-thermoneutral group comprised 15 pairs with at least one arrangement satisfying $|\Delta G_{\mathrm{H}}|\leq0.1$ eV; the other 109 formed the comparison group. Space groups were assigned using spglib [44] with a tolerance of 0.01 Å. Physical IDs fixed $M_1$ and $M_2$ while treating $\{M_3,M_4\}$ as unordered, grouping the space into 28560 compositions.

### DFT Dataset Construction

Dataset construction combined diversity sampling and composition deduplication [45]. The initial 500 structures were selected in the 77-dimensional Phase-D feature space. KMeans-centroid sampling, Pt-like d-band-center and electronegativity sampling, and uniform random sampling contributed 50%, 30% and 20%, respectively (SI Section S1.6).

One adaptive acquisition round scored the remaining ordered arrangements using the initial ensemble and $\mathrm{score}=\alpha U_{\mathrm{norm}}+(1-\alpha)(1-|\mu|_{\mathrm{norm}})$. Here, $\mu$ is the predicted free energy and $U$ the total predictive standard deviation. The quantities $|\mu|$ and $U$ were min–max normalized over the remaining candidates, with $\alpha=0.637$. Of 100 selections, 98 yielded valid DFT results, totaling 1206 records with the initial 500 and 608 historical records. Within each physical ID, the minimum pristine-structure and adsorption-structure energies were selected separately to calculate free energy, yielding 995 training compositions.

### Symmetric Descriptors and Feature Standardization

Removing nine derived columns from 96 descriptors yielded the common 87-dimensional input. It comprised 16 $M_1$, 13 $M_2$, six $M_{34}$ and 52 global features (SI Section S1.4; Table S2). Remote features used the means and absolute differences of electronegativity, d-electron count and atomic radius,

$$g(\mathbf{x}_{M_3},\mathbf{x}_{M_4}) = \left[\mathrm{mean}(\mathbf{x}_{M_3},\mathbf{x}_{M_4}),\; \left|\mathbf{x}_{M_3}-\mathbf{x}_{M_4}\right|\right] \in \mathbb{R}^{6}$$

Here, $\mathbf{x}_M$ contains these three attributes. Global statistics were calculated over the unordered set of four metals, making all models satisfy

$$f_\theta(M_1,M_2,M_3,M_4) = f_\theta(M_1,M_2,M_4,M_3) \quad \forall\,\theta$$

Features were standardized using the z-score transformation,

$$\tilde{x}_j = \frac{x_j - \bar{x}_j}{s_j}$$

The mean $\bar{x}_j$ and standard deviation $s_j$ were calculated from the training partition alone [46]. Each cross-validation fold contained 796 training and 199 validation compositions. Final-model standardization used all 995 training compositions for held-out and full-space predictions.

### Ensemble Model and Training

The symmetry-aware ensemble contained 10 constituent models, each encoding $M_1$, $M_2$ and $M_{34}$ into separate 32-dimensional representations. These were concatenated with 52 global features and passed through a $[128, 64]$ fusion network with ReLU activation and dropout probability 0.1 (Figure 3a),

$$\mathbf{z} = \mathrm{trunk}\left(\left[\mathbf{e}_1(\mathbf{x}^{(M_1)});\; \mathbf{e}_2(\mathbf{x}^{(M_2)});\; \mathbf{e}_{34}(g(\mathbf{x}^{(M_3)},\mathbf{x}^{(M_4)}));\; \mathbf{x}^{(\mathrm{global})}\right]\right)$$

The output layer $(\mu, \log\sigma^2) = \mathrm{head}(\mathbf{z})$ predicted the free-energy mean and log-variance using Gaussian negative log-likelihood loss [32]. Constituent models used different random seeds and bootstrap samples. The Adam optimizer was used with a learning rate of $10^{-3}$ and batch size 64; training stopped after 30 epochs without improvement in validation loss.

The ensemble prediction and uncertainties from $K=10$ constituent models were [32]

$$\hat{y}(\mathbf{x}) = \frac{1}{K}\sum_{k=1}^{K}\mu_k(\mathbf{x}), \qquad \sigma_{\mathrm{ale}}^2(\mathbf{x}) = \frac{1}{K}\sum_{k=1}^{K}\sigma_k^2(\mathbf{x}), \qquad \sigma_{\mathrm{epi}}^2(\mathbf{x}) = \frac{1}{K}\sum_{k=1}^{K}\left[\mu_k(\mathbf{x})-\hat{y}(\mathbf{x})\right]^2$$

Here, $\sigma_{\mathrm{ale}}^2$ and $\sigma_{\mathrm{epi}}^2$ denote aleatoric and epistemic variances. Error-ranking ability was assessed through the relationship between epistemic standard deviation and held-out prediction errors [47]. Full-space screening used $\sigma_{\mathrm{epi}}$, whereas acquisition used the total predictive standard deviation $U=\sqrt{\sigma_{\mathrm{ale}}^2+\sigma_{\mathrm{epi}}^2}$.

### Model Comparisons and Performance Evaluation

Five-fold cross-validation used shuffled KFold splits with random seed 42. The flat MLP shared the 87-dimensional input and training hyperparameters, with hidden layers $[256, 128, 64]$. Each constituent model contained 64706 parameters, compared with 32418 for the symmetry-aware model. XGBoost [48] was tuned on the same splits, using 300 trees, depth 7, learning rate 0.04, row and column subsampling 0.8, and a minimum child weight of 5. Ten traditional machine-learning models were also compared (SI Table S4).

The 2$\times$2 ablation combined both architectures with NLL or MSE loss using shared five-fold splits and training hyperparameters, with 10 constituent models per fold. MSE models used a single output neuron and the loss

$$L_\text{MSE} = \frac{1}{N} \sum_i (y_i - \mu_i)^2$$

Here, $y_i$ is the DFT free energy, $\mu_i$ its prediction, and $N$ the sample size.

Held-out evaluation used 110 compositions whose physical IDs were absent from the training set, independently of model selection [49]. The first 30 comprised 12 highly ranked, high-confidence, six promising but uncertain, six boundary and six blind-spot candidates. The subsequent 80 comprised 40 in-distribution and 40 out-of-distribution compositions (SI Figure S4). Model errors were paired by composition.

### Chemical Prior and Retrospective Screening

The chemical prior classified compositions with $|\Delta G_{\mathrm{H}}|\leq0.2$ eV as qualifying. Qualifying fractions for adsorption-center d-electron groups were normalized by the largest group fraction to obtain $\pi_{n_d}$. The mean and standard deviation of adsorption-center electronegativity among qualifying compositions defined the Gaussian component $\pi_\chi$. The components were combined as

$$P_{\mathrm{phys}} = 0.6\,\pi_{n_d} + 0.4\,\pi_\chi$$

For Figure 4a,b, the prior was calibrated using all 995 compositions. Qualifying fractions were compared between compositions at or above the 80th-percentile score, including ties, and the full set.

Retrospective comparison used the 995 existing DFT compositions under a shared selection budget [50]. Scores were $\mathrm{score}=\alpha_tU_{\mathrm{norm}}+\beta_tE+\gamma_tP_{\mathrm{phys}}$, where $E=1-|\mu|_{\mathrm{norm}}$. The chemical-prior and pure-UCB strategies used $\gamma_t=0.3e^{-t/3}$ and $\gamma_t=0$, respectively, sharing $\alpha_t$, $\beta_t$, random seeds and 200 initial compositions. Both used five constituent models and total predictive standard deviation (SI Section S1.6; Figure S7a). Each round selected 100 compositions, updating models and priors only with acquired DFT results; five rounds yielded 700 compositions. Eight paired repeats counted terminal qualifying compositions at 0.2 and 0.1 eV thresholds, including the initial samples.

### DFT Validation of Candidate Materials

Full-space screening successively selected compositions with $|\Delta G_{\mathrm{H}}|\leq0.1$ eV, epistemic standard deviations at or below the full-space median, and all four metals within Sc–Zn. Compositions were ranked by predicted absolute free energy (Figure 5; SI Table S1).

The prespecified validation set contained 40 compositions. Group A comprised 20 all-3d candidates selected using the dual-model mean_abs score, all with predicted absolute free energies below 0.1 eV. Group B comprised the ten candidates containing Pt, Pd, Ir or Rh with the lowest predicted absolute free energies. After excluding reliable candidates, group C sampled three, four and three compositions from [0, 0.1), [0.1, 0.3) and [0.3, 1.0) eV, respectively. Each stratum used random seed 42. Pristine and adsorption structures were relaxed separately from initial geometries. Once the final model was fixed, 20 additional compositions completed its all-3d top-30, yielding 60 validation results.

Physical ID matching identified two training compositions among the prespecified 40, leaving 38 for generalization evaluation. Set overlaps and complete results are given in SI Section S1.7 and Tables S9–S11.

### Elemental-Attribute Model and SHAP Analysis

The five-attribute adsorption-center model used valence-electron count, atomic radius, d-band center, lattice constant and electronegativity,

$$\hat{y} = \beta_0 + \sum_{j=1}^{5}\beta_j d_j + \sum_{1\le j\le k\le 5}\beta_{jk}\, d_j d_k$$

Here, $d_j$ denotes the corresponding attribute. Scikit-learn PolynomialFeatures generated 20 degree-two terms for Ridge regression [33] with $\alpha=0.1$, trained on 995 compositions and evaluated on 110 held-out compositions.

SHAP analysis [34] compared feature contributions in XGBoost, the symmetry-aware ensemble and the flat MLP using the 87-dimensional input. XGBoost used TreeExplainer [51] without background data; neural networks used DeepExplainer on the mean prediction of ten constituent models. CPU evaluation retained training normalization and explained all 995 compositions using the same 100 training structures as background. Mean absolute SHAP values were summed by column index within each feature block (SI Section S1.9; Figure S9). A separate free-energy-stratified comparison used all 995 compositions as background. Feature-wise SHAP distributions for 122 compositions satisfying $|\Delta G_{\mathrm{H}}|\leq0.1$ eV and the remaining 873 were compared by Wilcoxon rank-sum tests.

### Site-Resolved Expressions and Electronic Structure Analysis

The site-resolved empirical expression used SISSO-selected elemental attributes [37] and explicit site assignments. Cross-validation grouped records by composition and averaged adsorption-center-specific MAEs with equal weights. Coefficients fitted to all 995 compositions and the historical supplementary evaluation are documented in SI Section S1.10.

Electronic structure analysis compared occupation-weighted orbital projections, Bader charges and LOBSTER bonding projections [35,36] between pristine and adsorption structures. Charge and bonding responses were calculated by subtracting pristine values from adsorption values; fixed-geometry comparisons used total energies excluding entropy. Collinear magnetic comparisons allowed global spin reversal. Calculation batches, projection bases, reference geometries and electronic-state compatibility criteria are detailed in SI Section S1.10 and Figures S12–S14.

### Statistical Analysis

Fold-paired t-tests compared architectures at fixed loss and losses at fixed architecture. The 20 fold-level MAEs were also analyzed by two-way ANOVA with architecture, loss and their interaction, using fold as a blocking factor. Held-out comparisons paired absolute errors across 110 compositions; retrospective comparisons paired the final numbers of qualifying compositions across eight repeats, reporting means ± sample standard deviations.

Welch t-tests compared mean absolute free energies between the 18 group-A compositions absent from the training set and the ten group-B compositions, and between the complete A20/B10 sets. The 95% confidence intervals based on the Student t distribution were calculated across selected compositions. Spearman correlations were calculated between predictions and DFT free energies and between epistemic standard deviations and absolute prediction errors. A Mann–Whitney test compared exchange deviations between 15 near-thermoneutral and 109 other pairs.

A binomial proportion test assessed Co, Rh and Ir enrichment among 2428 predicted qualifying compositions against the full-space baseline of 5040/28560. Fisher's exact test compared predicted qualifying fractions for Co-centered compositions with and without Zn; Pearson correlation was calculated between prior scores and binary qualification labels. The t, Spearman, Mann–Whitney and Fisher tests were two-sided.
