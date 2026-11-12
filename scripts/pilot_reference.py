"""Offline sanity pilot with the NumPy reference simulator (no PennyLane needed).

Not a result for the paper: few seeds, few steps. It only checks that the
pipeline behaves as the light-cone argument predicts.

    python scripts/pilot_reference.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qorder import datasets, ordering, topology  # noqa: E402
from qorder.reference import predict_proba_reference, train_reference  # noqa: E402

N = 8
graph = topology.linear_chain(N)
slots = list(range(N))
suites = {
    "adversarial (4 real + 4 noise)": datasets.adversarial(dimension=4, n_noise=4),
    "breast_cancer (first 8 cols)": datasets.breast_cancer(method="first"),
}
for title, (X, y, Xt, yt) in suites.items():
    print(f"\n== {title} ==")
    print("MI:", np.round(ordering.mutual_information(X, y), 3))
    for L in (1, 2):
        for name in ("heuristic", "default", "reversed", "swapped", "anti_heuristic"):
            perm = ordering.make_ordering(name, X, y, slots, graph)
            sc = datasets.angle_scaler(X[:, perm])
            A, At = sc.transform(X[:, perm]), sc.transform(Xt[:, perm])
            accs = []
            for seed in range(3):
                m = train_reference(A, y, L, steps=150, lr=0.1, seed=seed)
                accs.append(np.mean((predict_proba_reference(m, At, L) > 0.5) == yt))
            print(f"L={L} {name:15s} perm={perm.tolist()} acc={np.mean(accs):.3f} ± {np.std(accs):.3f}",
                  flush=True)
