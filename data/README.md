# Result data

This directory contains result tables, composition features, element properties and analysis outputs.

| Directory | Contents |
|---|---|
| [processed/training](processed/training/) | Training labels, feature tables and earlier labeled sets |
| [processed/cohorts](processed/cohorts/) | Sampling, prediction and validation membership |
| [processed/symmetry_batches](processed/symmetry_batches/) | Composition membership and exchange comparisons |
| [processed](processed/) | Manuscript result tables and electronic/geometry summaries |
| [analysis/modeling](analysis/modeling/README.md) | Model comparisons, attribution and screening |
| [analysis/descriptors](analysis/descriptors/README.md) | Empirical expressions, coefficients and evaluation results |
| [analysis/electronic_structure](analysis/electronic_structure/README.md) | Charge, orbital, bonding and electronic-state responses |
| [analysis/charge_density](analysis/charge_density/README.md) | Remote-site charge-response comparisons |
| [analysis/scandium_comparison](analysis/scandium_comparison/README.md) | Sc/Co charge, geometry and orbital results |
| [elements](elements/) and [space](space/) | Element properties and composition-space features |

Tables whose names contain `structures` list compositions and selection fields.

[orbital_pdos.csv](processed/electronic_structure/orbital_pdos.csv) contains 12,006 rows for six electronic states, with energy, Fermi level and spin-resolved orbital curves. Empty H 1s columns denote clean structures. Bond distances and geometry summaries are stored under `processed/structure/`; Sc/Co electronic results are under `processed/electronic_structure/`.

See the [data dictionary](../docs/DATA_DICTIONARY.md), [table index](../provenance/table_index.csv) and [usage guide](../docs/REPRODUCIBILITY.md).
