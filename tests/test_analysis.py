import numpy as np
import pandas as pd

from qorder import analysis


def test_holm_matches_definition():
    p = np.array([0.01, 0.04, 0.03, np.nan])
    h = analysis.holm(p)
    assert np.allclose(h[:3], [0.03, 0.06, 0.06]) and np.isnan(h[3])


def test_depth_needed():
    rows = []
    for L, acc_h, acc_d in [(1, .90, .85), (2, .91, .87), (3, .91, .90), (5, .91, .91)]:
        for s in range(3):
            rows += [{"ordering": "heuristic", "m_n_layers": L, "m_observable": "local", "test_acc": acc_h},
                     {"ordering": "default", "m_n_layers": L, "m_observable": "local", "test_acc": acc_d}]
    g = pd.DataFrame([{"observable": "local", "n_qubits": 8, "n_layers": L, "var_first": v}
                      for L, v in [(1, .14), (2, .12), (3, .09), (5, .06)]])
    out = analysis.depth_needed(pd.DataFrame(rows), g, tol=0.01).set_index("ordering")
    assert out.loc["heuristic", "depth_needed"] == 1 and out.loc["default", "depth_needed"] == 3
    assert out.loc["heuristic", "grad_var_at_depth"] > out.loc["default", "grad_var_at_depth"]


def test_depth_needed_excludes_anti_and_sensitivity():
    rows = []
    for L, h, d, a in [(1, .90, .80, .70), (3, .90, .88, .80), (7, .90, .90, .70)]:
        rows += [{"ordering": "heuristic", "m_n_layers": L, "m_observable": "local", "test_acc": h},
                 {"ordering": "default", "m_n_layers": L, "m_observable": "local", "test_acc": d},
                 {"ordering": "anti_heuristic", "m_n_layers": L, "m_observable": "local", "test_acc": a}]
    df = pd.DataFrame(rows)
    out = analysis.depth_needed(df, tol=0.01).set_index("ordering")
    assert np.isclose(out.loc["heuristic", "target_acc"], 0.89)  # anti excluded from the target
    sens = analysis.depth_sensitivity(df, tols=(0.01, 0.03))
    assert sens.loc["default", 0.01] == 7 and sens.loc["default", 0.03] == 3


def test_across_datasets_per_depth(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(7):
        rows = []
        for L in (1, 2):
            for s in range(3):
                rows += [{"ordering": "heuristic", "m_n_layers": L, "test_acc": .9 + rng.normal(0, .005)},
                         {"ordering": "default", "m_n_layers": L, "test_acc": .8 + rng.normal(0, .005)}]
        d = tmp_path / f"exp5_d{i}"
        d.mkdir()
        pd.DataFrame(rows).to_csv(d / "runs.csv", index=False)
    table, tests = analysis.across_datasets(sorted(tmp_path.iterdir()), others=("default",))
    assert set(tests.depth) == {1, 2}
    assert (tests.datasets == 7).all() and (tests.wins == 7).all()
    assert (tests.wilcoxon_p < 0.01).all() and (tests.alternative == "greater").all()
