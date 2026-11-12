"""Summaries, Welch tests and plots for results/<exp>/runs.csv and gradients.csv.

    python -m qorder.analysis results/exp2_real_shallow
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

GROUP_COLS_CANDIDATES = ["dataset", "m_kind", "m_n_layers", "m_observable", "m_observable_type",
                         "m_entangling_layer", "m_reupload", "m_shots"]


def _group_cols(df):
    return [c for c in GROUP_COLS_CANDIDATES if c in df.columns and df[c].nunique(dropna=False) >= 1]


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (same order as the input; NaN stays NaN)."""
    p = np.asarray(pvalues, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def summarize(df: pd.DataFrame, metric="test_acc") -> pd.DataFrame:
    """Mean / std / n of `metric` per configuration and ordering."""
    g = _group_cols(df) + ["ordering"]
    out = df.groupby(g, dropna=False)[metric].agg(["mean", "std", "count"]).reset_index()
    return out


def welch_vs_heuristic(df: pd.DataFrame, metric="test_acc", reference="heuristic") -> pd.DataFrame:
    """Welch t-test of the heuristic against every other ordering, per configuration.

    gain = mean(heuristic) - mean(other). Also reports the gain over the best
    alternative ordering in that configuration (the 'heuristic gain' metric).
    """
    rows = []
    for keys, sub in df.groupby(_group_cols(df), dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        h = sub.loc[sub.ordering == reference, metric].to_numpy()
        if len(h) == 0:
            continue
        others = {o: s[metric].to_numpy() for o, s in sub.groupby("ordering") if o != reference}
        best = max(others, key=lambda o: others[o].mean()) if others else None
        for o, v in others.items():
            t, p = stats.ttest_ind(h, v, equal_var=False) if len(h) > 1 and len(v) > 1 else (np.nan, np.nan)
            rows.append({**dict(zip(_group_cols(df), keys)), "vs": o,
                         "heuristic_mean": h.mean(), "other_mean": v.mean(),
                         "gain": h.mean() - v.mean(), "t": t, "p_value": p,
                         "is_best_alternative": o == best})
    out = pd.DataFrame(rows)
    if not out.empty:
        keys = [c for c in _group_cols(df) if c in out]
        out["p_holm"] = out.groupby(keys, dropna=False)["p_value"].transform(lambda s: pd.Series(holm(s.values), index=s.index))
    return out


def heuristic_gain_by(df: pd.DataFrame, by="m_n_layers", metric="test_acc") -> pd.DataFrame:
    """Gain of the heuristic over the best alternative, as a function of `by` (H2)."""
    w = welch_vs_heuristic(df, metric)
    w = w[w.is_best_alternative]
    return w[[c for c in w.columns if c in (_group_cols(df) + ["vs", "gain", "p_value"])]]


def across_datasets(result_dirs, metric="test_acc", reference="heuristic",
                    others=("default", "random", "reversed", "anti_heuristic", "heuristic_corr", "heuristic_mrmr",
                            "kbest", "default_center", "heuristic_center", "cone_reversed", "cone_random"),
                    observable="local"):
    """H1 across datasets (Demsar, 2006): one mean per (dataset, depth, ordering).

    Returns (table, tests): the table of means and, for each depth and each other
    ordering, the wins/ties/losses of the reference and a Wilcoxon signed-rank test
    over the per-dataset means at that depth (Holm-adjusted within each depth for
    the one-sided comparisons).
    """
    frames = []
    for d in result_dirs:
        f = Path(d) / "runs.csv"
        if f.exists():
            df = pd.read_csv(f)
            if "m_observable" in df:
                df = df[df.m_observable.fillna(observable) == observable]
            df = df.assign(experiment=Path(d).name)
            frames.append(df)
    if not frames:
        raise ValueError("no runs.csv found")
    df = pd.concat(frames, ignore_index=True)
    layers = "m_n_layers" if "m_n_layers" in df else None
    idx = ["experiment"] + ([layers] if layers else [])
    table = df.pivot_table(index=idx, columns="ordering", values=metric, aggfunc="mean")
    # one Wilcoxon test per depth, one mean per dataset (Demsar, 2006); one-sided for the
    # orderings the heuristic is expected to beat, two-sided for its variants
    expected_worse = {"default", "random", "reversed", "anti_heuristic", "default_center",
                      "cone_reversed", "cone_random"}
    depths = sorted(table.index.get_level_values(layers).unique()) if layers else [None]
    rows = []
    for L in depths:
        sub = table.xs(L, level=layers) if layers else table
        for o in others:
            if o not in sub or reference not in sub:
                continue
            pair = sub[[reference, o]].dropna()
            diff = pair[reference] - pair[o]
            alt = "greater" if o in expected_worse else "two-sided"
            try:
                p = stats.wilcoxon(pair[reference], pair[o], alternative=alt, zero_method="zsplit").pvalue \
                    if len(pair) >= 6 else np.nan
            except ValueError:
                p = np.nan
            rows.append({"depth": L, "vs": o, "datasets": len(pair), "alternative": alt,
                         "mean_gain": diff.mean(), "median_gain": diff.median(),
                         "wins": int((diff > 0.005).sum()), "ties(±0.5pp)": int((diff.abs() <= 0.005).sum()),
                         "losses": int((diff < -0.005).sum()), "wilcoxon_p": p})
    tests = pd.DataFrame(rows)
    if not tests.empty:
        one = tests.alternative == "greater"
        tests["wilcoxon_p_holm"] = np.nan
        for L in depths:
            m = one & ((tests.depth == L) if layers else True)
            tests.loc[m, "wilcoxon_p_holm"] = holm(tests.loc[m, "wilcoxon_p"].values)
    return table, tests


def depth_needed(runs: pd.DataFrame, gradients: pd.DataFrame | None = None, tol=0.01,
                 metric="test_acc", n_qubits=8, exclude=("anti_heuristic",)):
    """Reformulated H3: smallest depth at which each ordering reaches the target accuracy.

    target = mean accuracy at the deepest depth tested (where the light cone already
    covers every qubit), averaged over the orderings not in `exclude`, minus `tol`.
    If a gradients table is given, adds the gradient variance (local cost, n_qubits) at that depth.
    """
    df = runs
    if "m_observable" in df:
        df = df[df.m_observable.fillna("local") == "local"]
    means = df.groupby(["ordering", "m_n_layers"])[metric].mean().unstack()
    deepest = means[means.columns.max()]
    target = deepest[[o for o in deepest.index if o not in exclude]].mean() - tol
    rows = []
    for o, r in means.iterrows():
        ok = r[r >= target]
        L = int(ok.index.min()) if len(ok) else None
        row = {"ordering": o, "tol": tol, "target_acc": target, "depth_needed": L,
               "acc_at_depth": float(r[L]) if L is not None else np.nan}
        if gradients is not None and L is not None:
            g = gradients[(gradients.observable == "local") & (gradients.n_qubits == n_qubits)
                          & (gradients.n_layers == L)]
            row["grad_var_at_depth"] = float(g.var_first.iloc[0]) if len(g) else np.nan
            for c in ("var_first_ci_low", "var_first_ci_high"):
                if c in g:
                    row[c] = float(g[c].iloc[0]) if len(g) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["depth_needed", "ordering"], na_position="last")


def depth_sensitivity(runs: pd.DataFrame, tols=(0.005, 0.01, 0.015, 0.02, 0.03), **kw) -> pd.DataFrame:
    """Depth needed per ordering (rows) for each tolerance (columns), to report how the conclusion moves."""
    t = pd.concat([depth_needed(runs, tol=x, **kw) for x in tols])
    return t.pivot_table(index="ordering", columns="tol", values="depth_needed")


def plot_orderings(df, metric="test_acc", path=None):
    import matplotlib.pyplot as plt
    cfgs = list(df.groupby(_group_cols(df), dropna=False))
    fig, axes = plt.subplots(1, len(cfgs), figsize=(4 * len(cfgs), 3.5), squeeze=False)
    for ax, (keys, sub) in zip(axes[0], cfgs):
        order = sorted(sub.ordering.unique(), key=lambda o: (o != "heuristic", o))
        data = [sub.loc[sub.ordering == o, metric] for o in order]
        try:
            ax.boxplot(data, tick_labels=order)
        except TypeError:  # matplotlib < 3.9
            ax.boxplot(data, labels=order)
        ax.set_title(str(keys), fontsize=7)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        if "logreg_test_acc" in sub and metric == "test_acc":
            ax.axhline(sub.logreg_test_acc.iloc[0], ls="--", lw=0.8, color="gray")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig


def plot_gradients(gdf, path=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for obs, sub in gdf.groupby("observable"):
        sub = sub.sort_values("n_layers")
        ax.semilogy(sub.n_layers, sub.var_first, "o-", label=f"{obs} cost")
    ax.set_xlabel("layers L")
    ax.set_ylabel(r"Var[$\partial C / \partial \theta_{0,0}$]")
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", nargs="+")
    ap.add_argument("--across", action="store_true", help="H1 across several experiment folders")
    ap.add_argument("--depth", default=None, help="H3 (reformulated): depth needed; give the gradients folder")
    a = ap.parse_args(argv)
    if a.across:
        table, tests = across_datasets(a.result_dir)
        pd.set_option("display.width", 200)
        print((table * 100).round(1).to_string())
        print()
        print(tests.round(4).to_string(index=False))
        return
    if a.depth:
        g = pd.read_csv(Path(a.depth) / "gradients.csv")
        r = pd.read_csv(Path(a.result_dir[0]) / "runs.csv")
        print(depth_needed(r, g).round(4).to_string(index=False))
        print("\nDepth needed vs tolerance (columns = tolerance in accuracy units):")
        print(depth_sensitivity(r).to_string())
        return
    d = Path(a.result_dir[0])
    if (d / "runs.csv").exists():
        df = pd.read_csv(d / "runs.csv")
        s = summarize(df)
        w = welch_vs_heuristic(df)
        s.to_csv(d / "summary.csv", index=False)
        w.to_csv(d / "welch.csv", index=False)
        print(s.to_string(index=False))
        print()
        print(w.to_string(index=False))
        plot_orderings(df, path=d / "orderings.png")
    if (d / "gradients.csv").exists():
        g = pd.read_csv(d / "gradients.csv")
        plot_gradients(g, path=d / "gradients.png")
        print(g.to_string(index=False))


if __name__ == "__main__":
    main()
