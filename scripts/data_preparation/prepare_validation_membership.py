"""Freeze composition membership and train-unseen validation statistics.

All 60 DFT rows remain intact. Membership is defined on the same physical ID
used to deduplicate the training set, independently of DFT job identity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed"
OUT = DATA / "dft_validation"


def physical_ids(frame: pd.DataFrame) -> pd.Series:
    """Return canonical composition IDs preserving M1 and M2 roles."""
    return frame.apply(
        lambda r: "-".join((r.M1, r.M2, *sorted((r.M3, r.M4)))), axis=1
    )


def summarize(frame: pd.DataFrame) -> dict:
    """Summarize fixed predictions and descriptive intervals by composition."""
    error = frame.pred_dG - frame.delta_G_H
    absolute = frame.delta_G_H.abs()
    rho = stats.spearmanr(frame.pred_dG, frame.delta_G_H)
    return {
        "n": len(frame), "hits": int((absolute <= 0.1).sum()),
        "hit_fraction": float((absolute <= 0.1).mean()),
        "mean_abs_DFT_eV": float(absolute.mean()),
        "median_abs_DFT_eV": float(absolute.median()),
        "mean_abs_DFT_ci95_eV": list(stats.t.interval(
            0.95, len(frame)-1, loc=absolute.mean(), scale=stats.sem(absolute))),
        "RMSE_eV": float(np.sqrt(np.mean(error**2))),
        "MAE_eV": float(error.abs().mean()),
        "mean_residual_eV": float(error.mean()),
        "mean_residual_ci95_eV": list(stats.t.interval(
            0.95, len(frame)-1, loc=error.mean(), scale=stats.sem(error))),
        "Spearman_rho": float(rho.statistic), "Spearman_p": float(rho.pvalue),
    }


def main() -> None:
    """Write membership CSV, statistics and provenance without editing labels."""
    files = {
        "training": "training/consolidated_ground_state.csv",
        "blind": "architecture/arch_advantage_blindtest.csv",
        "all": "dft_validation/verification_results_all60.csv",
        "pre": "dft_validation/verification_results_v3.csv",
        "cheap": "screening/SI_table_cheap_candidates.csv",
    }
    data = {k: pd.read_csv(DATA / v) for k, v in files.items()}
    for frame in data.values():
        frame["physical_id"] = physical_ids(frame)
        assert frame.physical_id.is_unique
    train_ids = set(data["training"].physical_id)
    blind_ids = set(data["blind"].physical_id)
    pre_ids = set(data["pre"].physical_id)
    assert not train_ids & blind_ids
    all60 = data["all"].copy()
    all60["seen_training"] = all60.physical_id.isin(train_ids)
    all60["in_blind110"] = all60.physical_id.isin(blind_ids)
    all60["pre_specified"] = all60.physical_id.isin(pre_ids)
    all60["cohort"] = np.where(all60.pre_specified, all60.group.str[0], "Completion")
    ranks = data["cheap"].set_index("physical_id").Rank
    all60["cheap_rank"] = all60.physical_id.map(ranks).astype("Int64")
    all60["in_top30"] = all60.cheap_rank.le(30).fillna(False)
    all60["DFT_hit_0p1"] = all60.delta_G_H.abs().le(0.1)
    all60["training_source"] = all60.physical_id.map(
        data["training"].set_index("physical_id").sources).fillna("")
    expected = {"Co-Cr-Mn-Zn", "Co-Mn-Ni-Zn"}
    assert set(all60.loc[all60.seen_training, "physical_id"]) == expected
    assert set(all60.loc[all60.in_blind110, "physical_id"]) == {"Co-Ir-Cr-Ti"}
    pre = all60[all60.pre_specified]
    top = all60[all60.in_top30]
    subsets = {
        "all60": all60, "unseen58": all60[~all60.seen_training],
        "pre40": pre, "pre_unseen38": pre[~pre.seen_training],
        "completion20": all60[~all60.pre_specified],
        "top30": top, "top_unseen29": top[~top.seen_training],
    }
    summary = {k: summarize(v) for k, v in subsets.items()}
    for key, n, hits in [("all60",60,35),("unseen58",58,33),
                         ("pre40",40,23),("pre_unseen38",38,21),
                         ("completion20",20,12),("top30",30,20),
                         ("top_unseen29",29,19)]:
        assert (summary[key]["n"], summary[key]["hits"]) == (n, hits)
    comparisons = {}
    for name, frame in [("all_pre", pre), ("unseen_pre", pre[~pre.seen_training])]:
        groups = {c: summarize(g) for c, g in frame.groupby("cohort")}
        a = frame.loc[frame.cohort == "A", "delta_G_H"].abs()
        b = frame.loc[frame.cohort == "B", "delta_G_H"].abs()
        test = stats.ttest_ind(a, b, equal_var=False)
        comparisons[name] = {"groups": groups, "Welch_A_B": {
            "t": float(test.statistic), "df": float(test.df), "p": float(test.pvalue)}}
    csv = OUT / "validation_membership.csv"
    all60.to_csv(csv, index=False)
    report = {
        "physical_ID": "M1, M2, sorted(M3, M4)",
        "threshold_eV": 0.1,
        "primary_generalization_scope": "pre_unseen38; A18 versus B10",
        "descriptive_scope": "All 60 DFT entries and complete top-30 retained",
        "interval": "Student t 95% interval across selected compositions; descriptive",
        "tests": "Two-sided Spearman and Welch A/B; no multiple-testing adjustment",
        "subsets": summary, "comparisons": comparisons,
        "input_md5": {v: hashlib.md5((DATA/v).read_bytes()).hexdigest() for v in files.values()},
        "membership_md5": hashlib.md5(csv.read_bytes()).hexdigest(),
    }
    (OUT / "validation_membership_stats.json").write_text(json.dumps(report, indent=2)+"\n")
    print("Membership fixed: 60/58, 40/38, 30/29; all expected hit counts verified.")


if __name__ == "__main__":
    main()
