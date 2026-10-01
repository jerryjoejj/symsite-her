"""HERALD - HER Active Learning Discovery.

Deep-ensemble screening of multimetallic single-atom HER catalysts.

Modules:
    models: Deep ensembles with epistemic and aleatoric uncertainty.
    acquisition: Adaptive exploration and optimization weights.
    dft: Parse saved energies and convert to adsorption free energies.
    data: Load composition spaces and merge training records.
    sampling: Select initial samples with a mixed strategy.

Energy convention:
    delta_G_H = E(H_site) - E(clean) + 3.481 eV, including the H2 reference.
    Near-zero delta_G_H defines the activity target.
"""

__version__ = "1.0.0"

# Free-energy correction (eV): -1/2 E(H2) + delta_ZPE - T*delta_S.
# The 3.481 eV correction includes the H2 reference; 0.24 eV alone omits it.
DELTA_G_H_CORRECTION = 3.481
