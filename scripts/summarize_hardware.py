"""Summarise hardware.csv (written by validate_hardware.py) over seeds.

    python scripts/summarize_hardware.py results/exp2_breast_cancer_shallow

For each backend mode and ordering: mean and standard deviation over seeds of the
accuracy on the backend, of the exact accuracy on the same test samples, and of
their difference; plus a Welch test of heuristic vs default on each mode.
Duplicate evaluations of the same model on the same mode keep the latest one.
"""
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

d = Path(sys.argv[1])
h = pd.read_csv(d / "hardware.csv").drop_duplicates(["run_id", "mode"], keep="last")
h["noise_minus_exact"] = h.accuracy - h.accuracy_exact
pd.set_option("display.width", 200)
summ = (h.groupby(["mode", "ordering", "n_layers"])
         .agg(models=("seed", "nunique"), acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
              exact_mean=("accuracy_exact", "mean"), diff_mean=("noise_minus_exact", "mean"),
              diff_std=("noise_minus_exact", "std"), expval_err=("mean_abs_expval_error", "mean"))
         .reset_index())
print((summ.round(4)).to_string(index=False))
print()
for (mode, L), g in h.groupby(["mode", "n_layers"]):
    a, b = g[g.ordering == "heuristic"].accuracy, g[g.ordering == "default"].accuracy
    if len(a) > 1 and len(b) > 1:
        t = stats.ttest_ind(a, b, equal_var=False, alternative="greater")
        print(f"{mode} L={L}: heuristic - default = {100 * (a.mean() - b.mean()):+.1f} pp "
              f"(one-sided Welch p = {t.pvalue:.4g}, {len(a)} vs {len(b)} models)")
