# Model training

The default run uses the supplied production ensemble and refits empirical models. Neural training writes to new output directories:

```bash
python code/train.py --task production --output outputs/training_production
python code/train.py --task architecture --output outputs/training_architecture
python code/train.py --task acquisition --output outputs/training_acquisition
```

In the capsule, run from `/code`, omit the `code/` prefix and use `/results/` output paths.

| Procedure | Published settings |
|---|---|
| Production | 995 compositions; 10 bootstrap members; seeds 42–51; 300 epochs; batch 64; Adam learning rate 0.001; weight decay 1e-5; dropout 0.1; Gaussian NLL |
| Architecture/loss | Shuffled five-fold split, seed 42; 10 members per fold; maximum 200 epochs; validation patience 30; four configurations |
| Acquisition | Eight paired repeats; split seeds 42–49; initial 200; five batches of 100; five members per round; 200 epochs; model seed 42; four CPU threads |

Feature and target standardization use each training partition. Production and acquisition fitting use all configured epochs because no validation partition is passed. Architecture comparison supplies the validation fold and restores the best validation state. The symmetric MSE implementation keeps two network heads and optimizes the mean output with MSE.

Acquisition training recalibrates the prior each round and exports newly selected composition IDs, round metrics and paired statistics. `training_run.json` records configurations and duration. Published round observations remain in `data/processed/acquisition/ablation_acquisition_curves.csv`.

## Installation check

```bash
python code/train.py --task production --smoke --output outputs/training_smoke
```

This trains one member for two epochs and checks finite predictions and checkpoint writing. Full training uses the table above. Hardware and library changes can change neural optimization trajectories; the supplied checkpoint defines the manuscript screening predictions.
