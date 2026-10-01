# Low-cost composition descriptors with site roles: definition, interpretation and validation

This document describes the inputs, equations, coefficients, worked example and evaluation of the site-resolved empirical expression D01. Its model identifier is `All_De1_original`.

## 1. Main finding and research question

The expression uses two composition components calculated directly from element properties: a size component \(D_r\) and an electronic component \(D_{e1}\). They distinguish the adsorption center, neighboring metal and remote metals while remaining invariant to exchange of the two remote sites. On the multimetallic M–N₄/G platform studied here, this representation describes composition trends in hydrogen adsorption free energy more accurately than an unordered auxiliary-metal representation with the same parameter count. Composition-trend and candidate-screening results are reported below.

Grouping structures only by the adsorption-center element or its valence-electron count combines different auxiliary-metal environments for the same center. The two-component expression also uses the auxiliary metals' radii and d-electron counts and retains their neighboring or remote roles. It therefore gives different predictions for different environments around the same center and distinguishes some structures with the same elements in different site assignments.

## 2. Structures, inputs and descriptor definitions

### 2.1 Sites and element properties

The platform contains four distinct metals at specified sites in \((M_1,M_2,M_3,M_4)\)-N₄/G. The sites are numbered by their roles as adsorption center, neighbor and remote metals.

| Symbol | Definition |
|---|---|
| \(M_1\) | Hydrogen adsorption center |
| \(M_2\) | Neighboring auxiliary site |
| \(M_3,M_4\) | Two remote auxiliary sites |
| \(r_i\) | Tabulated atomic radius of \(M_i\), in pm |
| \(n_i\) | Tabulated d-electron count of \(M_i\) |

Calculations use the element properties in the [model definition](../../../../models/descriptors/frozen_expressions.json). For example, the tabulated d-electron count of Co is 7.

### 2.2 Size component

With the mean remote radius \(\bar r_{34}=(r_3+r_4)/2\),

\[
D_r=
\frac{r_2+\bar r_{34}+|r_3-r_4|-2r_1}
{100\,\mathrm{pm}}
=
\frac{(r_2-r_1)+(\bar r_{34}-r_1)+|r_3-r_4|}
{100\,\mathrm{pm}}.
\]

The expression has three terms. \(r_2-r_1\) describes the radius difference between the neighbor and adsorption center; \(\bar r_{34}-r_1\) compares the mean remote radius with the center; and \(|r_3-r_4|\) retains the radius difference within the remote pair. Together, these terms describe the size environment using tabulated properties.

The fixed denominator of 100 pm makes \(D_r\) dimensionless. The first two terms can have either sign, whereas the remote-radius difference is nonnegative, so \(D_r\) is a signed size component.

For a fixed adsorption-center element, \(r_1\) is constant. In the linear model below, the contribution from \(-2r_1\) can be absorbed into that element's intercept:

\[
b'_{M_1}=b_{M_1}-a_{M_1}\frac{2r_1}{100\,\mathrm{pm}}.
\]

Thus, the intercept absorbs the center-radius term within each adsorption-center branch.

### 2.3 Electronic component

With the mean remote d-electron count \(\bar n_{34}=(n_3+n_4)/2\),

\[
D_{e1}=(n_2-n_1)(\bar n_{34}-n_2).
\]

The first factor compares the tabulated d-electron counts of the neighbor and center. The second compares the mean remote count with the neighbor. Their product connects all three site roles and differs from the mean or total d-electron count of the four metals.

\(D_{e1}>0\) when the two differences have the same sign and \(D_{e1}<0\) when they have opposite signs. If \(n_2=n_1\) or \(\bar n_{34}=n_2\), the component is zero. Different element combinations can therefore have the same component value.

Both components are exactly invariant to \(M_3/M_4\) exchange. DFT adsorption-energy differences under remote exchange are reported in the symmetry result tables.

## 3. Predictions from the two components

### 3.1 Components and center-specific coefficients

The model predicts signed hydrogen adsorption free energy:

\[
\widehat{\Delta G_{\mathrm H}}
=b_{M_1}+a_{M_1}D_r+c_{M_1}D_{e1}.
\]

The 17 adsorption-center elements share the same two component definitions. Each has an intercept and two slopes, giving 51 coefficients in total. The adsorption-center element \(M_1\) selects the coefficient set.

\(D_r\) and \(D_{e1}\) are dimensionless; \(b_{M_1}\), \(a_{M_1}\) and \(c_{M_1}\) have units of eV. Slopes can be positive or negative. Candidates are ranked by the absolute predicted free energy.

### 3.2 Cross-validation and the final expression

In composition-grouped cross-validation, a separate model is trained on each outer training set, with Ridge regularization selected within that set. Outer test labels do not enter this selection. The resulting fold-specific models evaluate the predictive performance of the expression.

For evaluation on the additional records, the final expression was fitted to all 995 training compositions after selecting regularization during development. Its coefficients were then frozen. The registered model is All_De1_original, with final \(\alpha=1\). Its coefficients are listed in the appendix.

The fold-specific models and the final model fitted to all 995 compositions have separate records of training membership, regularization strength, coefficients and evaluation results.

### 3.3 Inputs from the 87D feature table

All inputs required for the two components are included in the [87D feature table](../../../processed/features/training_87d.csv):

| Required quantity | Feature column |
|---|---|
| \(r_1,r_2\) | M1_atomic_radius, M2_atomic_radius |
| \(\bar r_{34},|r_3-r_4|\) | atomic_radius_M34_mean, atomic_radius_M34_absdiff |
| \(n_1,n_2\) | M1_d_electrons, M2_d_electrons |
| \(\bar n_{34}\) | d_electrons_M34_mean |

Seven property columns produce two continuous components. The adsorption-center identity then selects the coefficients for prediction. Application inputs come from the element-property table.

## 4. Worked example

Consider Co as the adsorption center and Cr, Cu and Ni as auxiliary metals. The following values come from the element-property table.

| Element | Atomic radius/pm | d-electron count |
|---|---:|---:|
| Co | 125 | 7 |
| Cr | 128 | 5 |
| Cu | 128 | 10 |
| Ni | 124 | 8 |

For the site assignment \((\mathrm{Co},\mathrm{Cr},\mathrm{Cu},\mathrm{Ni})\), Cr occupies the neighboring site and Cu/Ni occupy the remote sites:

\[
D_r=\frac{128+(128+124)/2+|128-124|-2\times125}{100}
=0.08,
\]

\[
D_{e1}=(5-7)\left[\frac{10+8}{2}-5\right]=-8.
\]

The Co branch of the frozen final expression is

\[
\frac{\widehat{\Delta G_{\mathrm H}}}{\mathrm{eV}}
\approx-0.008225703+0.100006371D_r-0.002310111D_{e1}.
\]

Using the full-precision coefficients gives approximately 0.018256 eV. Exchanging only the remote Cu and Ni leaves both components and the prediction unchanged. Moving Ni to the neighboring site changes the components and prediction:

| Site assignment in \(M_1,M_2,M_3,M_4\) order | \(D_r\) | \(D_{e1}\) | Frozen-expression prediction/eV |
|---|---:|---:|---:|
| Co, Cr, Cu, Ni | 0.08 | −8.0 | 0.018256 |
| Co, Cr, Ni, Cu | 0.08 | −8.0 | 0.018256 |
| Co, Ni, Cr, Cu | 0.02 | −0.5 | −0.005071 |

This example illustrates remote-exchange invariance and the effect of the neighboring-site assignment on the prediction.

## 5. Site-role comparison

The original representation uses the actual neighboring metal to calculate the two components. The unordered control assigns each auxiliary metal to the neighboring role in turn, calculates the full components and averages over the three assignments. Both models use 17 sets of intercepts and slopes, giving 51 coefficients.

The 995 training compositions are evaluated using two sets of composition-grouped five-fold splits. Records containing the same set of four metals belong to the same fold. Each outer training set uses inner cross-validation to select the Ridge strength. Macro-averaged MAE is calculated within each center and then averaged equally over the 17 adsorption-center elements.

| Representation | First grouped split macro-MAE/eV | Second grouped split macro-MAE/eV |
|---|---:|---:|
| Original site-role components | 0.118106 | 0.118411 |
| Unordered components | 0.132774 | 0.131039 |

Per-composition predictions, 20 fold-specific models and per-center metrics are provided in the [site-role results](../site_role_controls/). Both split assignments are in the [grouped split file](../validation_protocol/splits.json).

## 6. Final-expression evaluation

The final expression fitted to all 995 training compositions has an MAE of 0.125739 eV on 110 additional records, compared with 0.148560 eV for the center-median reference. Among the top 22 records ranked by absolute predicted free energy, the expression selects 10 with DFT \(|\Delta G_{\mathrm H}|\leq0.1\) eV. The center-median reference has an expected count of 10.2 when tied candidates are selected with equal probability. See the [summary metrics](../expression_validation/existing_F0_metrics.csv) and [per-record predictions](../frozen_expression_validation/predictions.csv).

For the Cr/Fe/Ni comparison with a fixed Co center, Cu neighbor and one remote Zn, DFT selects Fe and the empirical expression selects Ni. Values are provided in the [local composition comparison](../../electronic_structure/model_link/S2I_evidence_gate/analysis/composition_predictions.csv) and [ranking results](../../electronic_structure/model_link/S2I_evidence_gate/analysis/triad_choices.csv).

## 7. Calculation steps

1. Assign the adsorption center, neighbor and remote metals according to the M–N₄/G site framework.
2. Read atomic radii and d-electron counts from the model's 17-element table and calculate \(D_r,D_{e1}\).
3. Select the relevant \(M_1\) in the [full coefficient table](../expression_validation/final_coefficients.csv). The slopes use the stated input units and can be substituted directly into the equation.
4. Calculate the signed free-energy prediction. Rank compositions using \(|\widehat{\Delta G_{\mathrm H}}|\).

The [model file](../../../../models/descriptors/frozen_expressions.json) provides the expression definitions, element properties and training-input distributions for each center.

## Appendix A. Coefficients for the final expression fitted to all 995 compositions

The table corresponds to All_De1_original with \(\alpha=1\), using
\(\widehat{\Delta G_{\mathrm H}}=b_{M_1}+a_{M_1}D_r+c_{M_1}D_{e1}\).
All three coefficient columns are in eV. Values are shown to 9 decimal places; calculations use the full precision in the [CSV](../expression_validation/final_coefficients.csv) or [frozen model definition](../../../../models/descriptors/frozen_expressions.json).

| Adsorption center \(M_1\) | \(b_{M_1}\) | \(a_{M_1}\) | \(c_{M_1}\) |
|---|---:|---:|---:|
| Ag | 0.421538078 | -0.283607401 | -0.000614991 |
| Au | 0.353228761 | -1.290614130 | -0.011370093 |
| Co | -0.008225703 | 0.100006371 | -0.002310111 |
| Cr | 0.121809572 | -0.025366433 | -0.000826565 |
| Cu | 1.343397155 | 0.212691705 | -0.003210970 |
| Fe | 0.288703440 | 1.475988053 | 0.012386673 |
| Ir | -0.479271332 | 0.392801835 | -0.006942212 |
| Mn | 0.392875936 | -0.030656615 | 0.000017115 |
| Ni | 1.076086463 | 0.678168062 | -0.003085211 |
| Pd | 1.490691718 | 0.741336090 | -0.001690296 |
| Pt | 1.176253807 | 0.788469487 | -0.001538520 |
| Rh | -0.259407091 | 0.734580526 | 0.000275214 |
| Ru | -0.638326018 | -0.081931175 | -0.000139714 |
| Sc | -0.985644706 | -1.608713172 | -0.002041941 |
| Ti | -0.675075801 | -0.064249524 | -0.000311424 |
| V | -0.277563357 | -0.047662795 | 0.001259614 |
| Zn | 0.503694750 | -0.186905611 | -0.005583169 |
