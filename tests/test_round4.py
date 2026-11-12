import json

import numpy as np

from qorder import run
from qorder.reference import ShallowShotsClassifier, light_cone_qubits


def _cfg(orderings, L=2):
    return {"name": "t_r4", "dataset": {"name": "digits", "classes": [3, 8], "n_components": 8},
            "model": {"kind": "shallow_reference", "n_qubits": 8, "observable": "local",
                      "learning_rate": 0.1, "max_steps": 3},
            "grid": {"n_layers": [L]}, "orderings": orderings, "seeds": [0, 1]}


def test_cone_orderings_permute_only_inside_the_cone(tmp_path):
    L = 2
    df = run.run_training(_cfg(["heuristic", "cone_reversed", "cone_random"], L), tmp_path, quick=True)
    cone = light_cone_qubits(L, 8, 0)
    for s in (0, 1):
        g = df[df.seed == s].set_index("ordering").perm.map(json.loads)
        h, r, x = np.array(g["heuristic"]), np.array(g["cone_reversed"]), np.array(g["cone_random"])
        out = [q for q in range(8) if q not in cone]
        assert (h[out] == r[out]).all() and (h[out] == x[out]).all()          # outside the cone unchanged
        assert sorted(h[cone]) == sorted(r[cone]) == sorted(x[cone])          # same features inside
        assert (r[cone] == h[cone][::-1]).all()
        assert not (x[cone] == h[cone]).all()                                 # a real permutation


def test_shots_classifier_exact_and_noisy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 8))
    y = (X[:, 0] > 0).astype(int)
    exact = ShallowShotsClassifier(n_layers=1, shots=None, max_steps=60, convergence_interval=0).fit(X, y)
    noisy = ShallowShotsClassifier(n_layers=1, shots=1, max_steps=60, convergence_interval=0).fit(X, y)
    assert exact.info_["steps"] == 60 and noisy.info_["steps"] == 60
    assert not np.allclose(exact.model_["theta"], noisy.model_["theta"])
    assert exact.score(X, y) > 0.6
    again = ShallowShotsClassifier(n_layers=1, shots=1, max_steps=60, convergence_interval=0).fit(X, y)
    assert np.allclose(again.model_["theta"], noisy.model_["theta"])   # reproducible
