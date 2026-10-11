#!/usr/bin/env python3
"""Retrospective comparison of physics-guided acquisition and pure UCB.

Both strategies select from 995 compositions with known DFT labels. Labels
are revealed only after selection. Each paired repeat shares the initial
set, candidate pool, V2 model, batch size and diversity-selection procedure.
The acquisition weight gamma distinguishes the strategies:
    Pure UCB: gamma=0, score=alpha*U+beta*E.
    Physics-guided: gamma=compute_gamma(t), with an added gamma*P_phys term.
Independent random splits are used instead of replaying iteration 0.

Per-round outputs include the best absolute free energy, cumulative hits
at 0.1/0.2 eV and batch d-electron distributions. Paired t tests compare
final hit counts and convergence AUC across repeats.
CPU execution uses one process per repeat and four threads per process.
"""

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from multiprocessing import get_context

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" if (ROOT / "src").exists() else ROOT / "code/src"))

TRAIN_FEATURES = ROOT / "data/processed/training/consolidated_ground_state_features_v2.csv"
PROPS_PATH = ROOT / "data/elements/element_properties.json"
OUT = ROOT / "outputs/retraining"

# Experiment settings
N_REPLICATES = 8         # Number of paired random-split repeats
SEED_SIZE = 200          # Initial labeled-set size
BATCH_SIZE = 100         # Compositions selected per round
N_ROUNDS = 5             # Acquisition rounds (200 -> 700 labeled compositions)
N_MEMBERS = 5            # Ensemble members for estimating epistemic uncertainty
EPOCHS = 200
PATIENCE = 25
THREADS = 4
BASE_SEED = 42
EXCELLENT_THR = 0.2      # Activity threshold, matching the PhysicsPrior default
HIT_THRS = (0.1, 0.2)    # Hit thresholds


def _load_pool():
    """Load the 995 labeled compositions and element properties; extract M1, labels and features."""
    from herald.features.calculator import FeatureCalculator, load_element_properties
    df = pd.read_csv(TRAIN_FEATURES)
    props = load_element_properties(PROPS_PATH)
    calc = FeatureCalculator(props)
    data = calc.compute_symmetric_input_from_features(df)   # dict of (N,·)
    y = df["delta_G_H"].to_numpy(dtype=float)
    M1 = df["M1"].to_numpy()
    # Concatenated 87D features for diversity selection
    feat_flat = np.concatenate(
        [data["x_M1"], data["x_M2"], data["x_M34_sym"], data["g_global"]], axis=1
    )
    return data, y, M1, feat_flat, props


def _subset(data, idx):
    return {k: v[idx] for k, v in data.items()}


def _train_ensemble(data_tr, y_tr):
    from herald.models.symmetry_aware_ensemble_v2 import (
        SymmetryAwareEnsembleV2, SymmetryEnsembleConfigV2)
    import torch
    torch.set_num_threads(THREADS)
    cfg = SymmetryEnsembleConfigV2(
        n_members=N_MEMBERS, epochs=EPOCHS, early_stop_patience=PATIENCE,
        device="cpu", seed=BASE_SEED,
    )
    ens = SymmetryAwareEnsembleV2(cfg)
    ens.fit(data_tr, y_tr, verbose=False)
    return ens


def _run_arm(gamma_mode, data, y, M1, feat_flat, props, seed_idx, pool_idx, rep):
    """Run one acquisition strategy.

    gamma_mode is 'ucb' (gamma=0) or 'physics' (gamma=compute_gamma).
    Return per-round metrics as a list of dictionaries. The caller supplies
    the same seed_idx and pool_idx for both strategies.
    """
    from herald.acquisition.physics_guided import (
        PhysicsPrior, physics_guided_score, compute_gamma, greedy_diverse_selection)

    df_all = pd.DataFrame({"M1": M1, "delta_G_H": y})
    labeled = list(seed_idx)
    remaining = list(pool_idx)
    records = []

    for t in range(N_ROUNDS):
        # 1) Train on the currently labeled set.
        ens = _train_ensemble(_subset(data, labeled), y[labeled])

        # 2) Predict the remaining pool.
        rem = np.array(remaining)
        pred = ens.predict(_subset(data, rem))

        # 3) Calibrate the chemical prior using the currently labeled set.
        prior = PhysicsPrior.calibrate(df_all.iloc[labeled], props,
                                       excellent_thr=EXCELLENT_THR)

        # 4) Use the same alpha/beta schedule; only gamma differs.
        alpha = max(0.7 * (0.8 ** t), 0.3)
        gamma = 0.0 if gamma_mode == "ucb" else compute_gamma(t, gamma_0=0.3, tau_decay=3.0)

        scores = physics_guided_score(
            pred_mean=pred.mean, uncertainty=pred.total_std,
            M1_list=M1[rem], prior=prior, props=props,
            alpha=alpha, gamma=gamma, target=0.0,
        )

        # 5) Apply the same diversity-based batch selection.
        k = min(BATCH_SIZE, len(rem))
        sel_local = greedy_diverse_selection(
            scores, feat_flat[rem], n_select=k,
            diversity_weight=0.5, candidate_pool=min(500, len(rem)))
        sel_global = rem[sel_local]

        # Save composition identities for this newly trained acquisition run.
        selection = pd.read_csv(TRAIN_FEATURES).iloc[sel_global][["M1", "M2", "M3", "M4", "delta_G_H"]].copy()
        selection.insert(0, "training_row_0based", sel_global)
        selection.to_csv(OUT / f"selected_repeat{rep}_{gamma_mode}_round{t+1}.csv", index=False)

        # 6) Reveal labels and record batch metrics.
        batch_y = y[sel_global]
        # Identify Co/Rh/Ir through valence_electrons == 9.
        val_batch = np.array([int(props[m]["valence_electrons"]) for m in M1[sel_global]])
        labeled.extend(sel_global.tolist())
        remaining = [i for i in remaining if i not in set(sel_global.tolist())]

        lab_y = y[labeled]
        rec = {
            "rep": rep, "arm": gamma_mode, "round": t + 1,
            "n_labeled": len(labeled), "gamma": round(gamma, 4), "alpha": round(alpha, 3),
            "batch_best_abs": float(np.abs(batch_y).min()),
            "cum_best_abs": float(np.abs(lab_y).min()),
            "batch_mean_abs": float(np.abs(batch_y).mean()),
            "val9_frac_batch": float((val_batch == 9).mean()),  # Fraction of nine-valence-electron centers (Co/Rh/Ir)
        }
        for thr in HIT_THRS:
            rec[f"batch_hits_{thr}"] = int((np.abs(batch_y) <= thr).sum())
            rec[f"cum_hits_{thr}"] = int((np.abs(lab_y) <= thr).sum())
        records.append(rec)

    return records


def _run_replicate(args):
    """Run a paired repeat with the same split and initial set; only gamma differs."""
    rep, n_total = args
    rng = np.random.default_rng(BASE_SEED + rep)
    perm = rng.permutation(n_total)
    seed_idx = perm[:SEED_SIZE]
    pool_idx = perm[SEED_SIZE:]

    data, y, M1, feat_flat, props = _load_pool()
    out = []
    for mode in ("ucb", "physics"):
        out.extend(_run_arm(mode, data, y, M1, feat_flat, props,
                            seed_idx, pool_idx, rep))
    print(f"  [rep {rep}] finished (seed={SEED_SIZE}, pool={len(pool_idx)})", flush=True)
    return out


def _paired_stats(df):
    """Run paired tests on final hit counts and convergence AUC (lower is better)."""
    from scipy import stats
    rows = []
    reps = sorted(df["rep"].unique())

    def _pivot(metric, agg):
        u, p = [], []
        for r in reps:
            sub = df[df["rep"] == r]
            uu = sub[sub.arm == "ucb"]
            pp = sub[sub.arm == "physics"]
            u.append(agg(uu)); p.append(agg(pp))
        return np.array(u), np.array(p)

    # Final cumulative hits (higher is better)
    for thr in HIT_THRS:
        u, p = _pivot(f"cum_hits_{thr}",
                      lambda s: s.sort_values("round")[f"cum_hits_{thr}"].iloc[-1])
        t, pv = stats.ttest_rel(p, u)
        rows.append({"metric": f"final_cum_hits_{thr}", "ucb_mean": u.mean(),
                     "physics_mean": p.mean(), "delta": p.mean() - u.mean(),
                     "t": t, "p_value": pv, "better": "physics" if p.mean() > u.mean() else "ucb"})

    # Convergence AUC: sum of cum_best_abs over rounds (lower is better).
    u, p = _pivot("cum_best_abs",
                  lambda s: s.sort_values("round")["cum_best_abs"].sum())
    t, pv = stats.ttest_rel(p, u)
    rows.append({"metric": "convergence_auc_cum_best_abs", "ucb_mean": u.mean(),
                 "physics_mean": p.mean(), "delta": p.mean() - u.mean(),
                 "t": t, "p_value": pv, "better": "physics" if p.mean() < u.mean() else "ucb"})

    # Enrichment of selected nine-valence-electron centers
    u, p = _pivot("val9_frac_batch", lambda s: s["val9_frac_batch"].mean())
    t, pv = stats.ttest_rel(p, u)
    rows.append({"metric": "mean_val9_frac_batch", "ucb_mean": u.mean(),
                 "physics_mean": p.mean(), "delta": p.mean() - u.mean(),
                 "t": t, "p_value": pv, "better": "physics" if p.mean() > u.mean() else "ucb"})
    return pd.DataFrame(rows)
