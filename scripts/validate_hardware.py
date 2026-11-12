"""Evaluate trained shallow models on finite shots, a noisy fake backend or IBM hardware.

Examples
    # free: exact sampling and a noisy local copy of an IBM device
    python scripts/validate_hardware.py results/exp2_breast_cancer_shallow --mode ideal
    python scripts/validate_hardware.py results/exp2_breast_cancer_shallow --mode fake:auto

    # QPU (spends the monthly allowance - run --dry-run first)
    python scripts/validate_hardware.py results/exp2_breast_cancer_shallow --mode ibm:least_busy \
        --orderings heuristic default --layers 2 --seeds 0 --shots 1000 --max-samples 200 --budget 120

The script picks one trained model per (ordering, layers, seed) from runs.csv,
rebuilds the test split from the saved config and appends to hardware.csv.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qorder import datasets  # noqa: E402
from qorder.hardware import estimate_qpu_seconds, evaluate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("result_dir")
ap.add_argument("--mode", default="ideal", help="ideal | fake:auto | fake:FakeFez | ibm:least_busy | ibm:<name>")
ap.add_argument("--orderings", nargs="+", default=["heuristic", "default"])
ap.add_argument("--layers", nargs="+", type=int, default=None)
ap.add_argument("--observable", default="local")
ap.add_argument("--seeds", nargs="+", type=int, default=[0])
ap.add_argument("--shots", type=int, default=1000)
ap.add_argument("--max-samples", type=int, default=200)
ap.add_argument("--budget", type=float, default=None, help="max QPU seconds for the whole call (ibm only)")
ap.add_argument("--dry-run", action="store_true", help="only print what would run and the QPU estimate")
a = ap.parse_args()

d = Path(a.result_dir)
cfg = yaml.safe_load(open(d / "config.yaml"))
runs = pd.read_csv(d / "runs.csv")
sel = runs[runs.ordering.isin(a.orderings) & runs.seed.isin(a.seeds)]
if "m_observable" in sel:
    sel = sel[sel.m_observable == a.observable]
if a.layers:
    sel = sel[sel.m_n_layers.isin(a.layers)]
sel = sel[[(d / "models" / f"{r}.json").exists() for r in sel.run_id]]
if sel.empty:
    sys.exit("no exported shallow models match the selection")

_, _, Xt, yt = datasets.load(cfg["dataset"]["name"], **{k: v for k, v in cfg["dataset"].items() if k != "name"})
n_eval = min(a.max_samples, len(Xt))
total = estimate_qpu_seconds(n_eval, a.shots, n_jobs=len(sel))
print(f"{len(sel)} models x {n_eval} test samples x {a.shots} shots "
      f"-> estimated QPU time {total:.1f} s (only counts on IBM hardware)")
print(sel[["ordering", "m_n_layers", "seed", "test_acc"]].to_string(index=False))
if a.dry_run:
    sys.exit(0)
if a.mode.startswith("ibm:") and a.budget is not None and total > a.budget:
    sys.exit(f"estimate {total:.1f} s exceeds --budget {a.budget} s; reduce models, samples or shots")

out_csv = d / "hardware.csv"
for _, r in sel.iterrows():
    export = json.load(open(d / "models" / f"{r.run_id}.json"))
    res = evaluate(export, Xt, yt, mode=a.mode, shots=a.shots, max_samples=a.max_samples)
    row = {"run_id": r.run_id, "ordering": r.ordering, "n_layers": r.m_n_layers, "seed": r.seed,
           "test_acc_simulator_full": r.test_acc, **res}
    pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
    print(json.dumps(row))
