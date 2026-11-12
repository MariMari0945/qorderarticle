"""Experiment runner.

    python -m qorder.run configs/exp1_adversarial_drc.yaml
    python -m qorder.run configs/exp2_real_shallow.yaml --quick     # smoke test

Results go to results/<name>/runs.csv (one row per trained model, appended as
it goes, so an interrupted Colab session resumes where it stopped) and, for
shallow models, results/<name>/models/<run_id>.json (for hardware validation).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from . import datasets, ordering, topology


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _grid(cfg):
    g = cfg.get("grid") or {}
    keys = list(g)
    for values in itertools.product(*[g[k] for k in keys]):
        yield dict(zip(keys, values))


def _run_id(d):
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def classical_baseline(X, y, Xt, yt):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(X), y)
    p = clf.predict_proba(sc.transform(Xt))[:, 1]
    return {"logreg_test_acc": float(np.mean((p > 0.5) == yt)), "logreg_auc": float(roc_auc_score(yt, p))}


def run_training(cfg, out_dir: Path, quick=False):
    """Train every (grid point, ordering, seed).

    With `resplit: true` in the config, seed s also draws its own train/test split
    (the dataset loader gets seed=s), so the seeds measure the variability of the
    data split and of the optimisation together. Otherwise the split is fixed.
    """
    seeds = cfg.get("seeds", [0])
    if quick:
        seeds = seeds[:2]
    if cfg.get("resplit"):
        for s in seeds:
            _train(cfg, out_dir, quick, [s], split_seed=s)
    else:
        _train(cfg, out_dir, quick, seeds, split_seed=None)
    return pd.read_csv(out_dir / "runs.csv")


def _train(cfg, out_dir: Path, quick, seeds, split_seed=None):
    def build_model(c):
        if c["kind"] == "shallow_reference":
            from .reference import ShallowReferenceClassifier
            return ShallowReferenceClassifier(**{k: v for k, v in c.items() if k != "kind"})
        if c["kind"] == "shallow_shots":
            from .reference import ShallowShotsClassifier
            return ShallowShotsClassifier(**{k: v for k, v in c.items() if k != "kind"})
        from .models import build_model as _bm  # needs pennylane/jax
        return _bm(c)

    ds_kw = {k: v for k, v in cfg["dataset"].items() if k != "name"}
    if split_seed is not None:
        ds_kw["seed"] = split_seed
    X, y, Xt, yt = datasets.load(cfg["dataset"]["name"], **ds_kw)
    print(f"dataset {cfg['dataset']['name']}: train {X.shape}, test {Xt.shape}, "
          f"class balance {y.mean():.2f}" + (f", split seed {split_seed}" if split_seed is not None else ""))
    base = classical_baseline(X, y, Xt, yt)
    print("classical baseline:", base)
    mi = ordering.mutual_information(X, y, random_state=0)
    corr = ordering.abs_correlation(X, y)
    scores = {"mi": mi, "corr": corr}
    if any(o.endswith("mrmr") for o in cfg["orderings"]):
        scores["mrmr"] = ordering.mrmr_scores(X, y)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    info_name = "dataset_info.json" if split_seed is None else f"dataset_info_split{split_seed}.json"
    with open(out_dir / info_name, "w") as f:
        json.dump({"mutual_information": mi.tolist(), "abs_correlation": corr.tolist(), **base, "n_train": len(X), "n_test": len(Xt)}, f, indent=2)

    csv = out_dir / "runs.csv"
    done = set(pd.read_csv(csv)["run_id"]) if csv.exists() else set()
    orderings = cfg["orderings"]

    for overrides in _grid(cfg):
        mcfg = {**cfg["model"], **overrides}
        if quick:
            mcfg["max_steps"] = min(mcfg.get("max_steps", 1000), 60)
            mcfg["convergence_interval"] = 0
        kind = mcfg["kind"]
        if kind in ("shallow", "shallow_reference", "shallow_shots"):
            n_qubits = mcfg.get("n_qubits", X.shape[1])
            if n_qubits != X.shape[1]:
                raise ValueError(f"shallow model needs one feature per qubit: {X.shape[1]} features, {n_qubits} qubits")
            slots = list(range(n_qubits))
        else:
            n_qubits = int(np.ceil(X.shape[1] / 3))
            slots = [s // 3 for s in range(X.shape[1])]
        graph = topology.get_topology(cfg.get("topology", "linear"), n_qubits)

        mq = mcfg.get("measured_qubit", 0)
        for name, seed in itertools.product(orderings, seeds):
            run_cfg, run_slots = mcfg, slots
            if name in ("kbest", "kbest_corr"):
                # Baseline: a smaller circuit with only the k most informative features, k = cone size,
                # so every qubit is inside the light cone. Separates "ordering" from "feature selection".
                from .reference import light_cone_size
                k = light_cone_size(mcfg["n_layers"], X.shape[1], 0)
                ranked = ordering.rank_features(scores["mi"] if name == "kbest" else scores["corr"])
                perm = ranked[:k]
                run_cfg = {**mcfg, "n_qubits": int(k), "measured_qubit": 0}
                run_slots = list(range(k))
            elif name in ("cone_reversed", "cone_random"):
                # Baseline: the same L+1 features as the heuristic inside the light cone, but in a different
                # order within it (reversed, or a random non-identity permutation). Separates the effect of the
                # position inside the cone from that of selecting which features enter it.
                from .reference import light_cone_qubits
                perm = ordering.make_ordering("heuristic", X, y, slots, graph, measured_qubit=mq,
                                              seed=1000 + seed, scores=scores)
                cone = set(light_cone_qubits(mcfg["n_layers"], n_qubits, mq))
                inside = [s for s in range(len(slots)) if slots[s] in cone]
                feats = perm[inside].copy()
                if name == "cone_reversed":
                    new = feats[::-1]
                else:
                    r = np.random.default_rng(2000 + seed)
                    new = r.permutation(feats)
                    while len(feats) > 1 and np.array_equal(new, feats):
                        new = r.permutation(feats)
                perm = perm.copy()
                perm[inside] = new
            elif name.endswith("_center"):
                # Baseline: measure a qubit in the middle of the chain instead of qubit 0. The light cone
                # then grows in both directions (2L+1 qubits) without reordering anything.
                if kind not in ("shallow", "shallow_reference", "shallow_shots"):
                    raise ValueError("'_center' orderings need the shallow circuit")
                center = (X.shape[1] - 1) // 2
                run_cfg = {**mcfg, "measured_qubit": center}
                perm = ordering.make_ordering(name[: -len("_center")], X, y, slots, graph, measured_qubit=center,
                                              seed=1000 + seed, scores=scores)
            else:
                perm = ordering.make_ordering(name, X, y, slots, graph, measured_qubit=mq,
                                              seed=1000 + seed, scores=scores)
            key = {"exp": cfg["name"], "model": run_cfg, "ordering": name, "seed": seed,
                   "perm": perm.tolist(), "quick": quick}
            if split_seed is not None:
                key["split_seed"] = split_seed
            rid = _run_id(key)
            if rid in done:
                continue
            model = build_model({**run_cfg, "random_state": seed})
            t0 = time.time()
            model.fit(X[:, perm], y)
            p = model.predict_proba(Xt[:, perm])
            mq_run = run_cfg.get("measured_qubit", 0)
            row = {
                "run_id": rid, "exp": cfg["name"], "dataset": cfg["dataset"]["name"],
                **{f"m_{k}": v for k, v in run_cfg.items()},
                "ordering": name, "seed": seed, "perm": json.dumps(perm.tolist()),
                "train_acc": model.score(X[:, perm], y),
                "test_acc": float(np.mean((p > 0.5) == yt)),
                "test_loss": model.loss(Xt[:, perm], yt),
                "test_f1": float(f1_score(yt, p > 0.5)),
                "test_auc": float(roc_auc_score(yt, p)),
                "steps": model.info_["steps"], "steps_to_converge": model.info_["steps_to_converge"],
                "converged": model.info_["converged"], "train_time_s": round(time.time() - t0, 1),
                "mi_on_measured_qubit": float(sum(mi[perm[s]] for s in range(len(perm)) if run_slots[s] == mq_run)),
                **base, "quick": quick,
            }
            row.setdefault("m_measured_qubit", mq_run)
            if "mean_abs_expval" in model.info_:
                row["train_mean_abs_expval"] = model.info_["mean_abs_expval"]
            row["split_seed"] = split_seed if split_seed is not None else -1
            # rewrite instead of append, so rows with new columns (e.g. m_measured_qubit) stay aligned
            new = pd.DataFrame([row])
            if csv.exists():
                new = pd.concat([pd.read_csv(csv), new], ignore_index=True)
            tmp = csv.with_suffix(".tmp")
            new.to_csv(tmp, index=False)
            os.replace(tmp, csv)  # atomic: an interrupted session never leaves a half-written file
            if kind in ("shallow", "shallow_reference", "shallow_shots"):
                exp = model.export()
                exp["perm"] = perm.tolist()
                with open(out_dir / "models" / f"{rid}.json", "w") as f:
                    json.dump(exp, f)
            print(f"{name:14s} seed={seed} {overrides} test_acc={row['test_acc']:.3f} "
                  f"steps={row['steps']} ({row['train_time_s']}s)", flush=True)


def run_gradients(cfg, out_dir: Path, quick=False):
    from .gradients import sweep
    g = dict(cfg["gradients"])
    if quick:
        g["n_init"] = min(g.get("n_init", 200), 20)
    rows = sweep(**g)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "gradients.csv", index=False)
    print(df.to_string(index=False))
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--quick", action="store_true", help="few steps/seeds, to check the pipeline")
    ap.add_argument("--results", default="results")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    out_dir = Path(a.results) / (cfg["name"] + ("_quick" if a.quick else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
    if cfg.get("type", "training") == "gradients":
        run_gradients(cfg, out_dir, a.quick)
    else:
        run_training(cfg, out_dir, a.quick)
    print(f"\nresults in {out_dir}{os.sep}")


if __name__ == "__main__":
    main()
