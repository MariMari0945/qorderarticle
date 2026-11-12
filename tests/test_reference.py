import numpy as np

from qorder import reference


def test_light_cone_grows_one_qubit_per_layer():
    rng = np.random.default_rng(0)
    n = 8
    for L in (1, 2, 3, 5):
        for reupload in (False, True):
            th = rng.uniform(0, 2 * np.pi, (L + 1, n))
            X = rng.uniform(-1, 1, (6, n))
            base = reference.expval(th, X, L, reupload)
            influence = [np.abs(reference.expval(th, X + 0.7 * np.eye(n)[q], L, reupload) - base).max() > 1e-10
                         for q in range(n)]
            assert influence == [q < reference.light_cone_size(L, n) for q in range(n)]


def test_global_cost_sees_all_qubits():
    rng = np.random.default_rng(1)
    n, L = 6, 1
    th = rng.uniform(0, 2 * np.pi, (L + 1, n))
    X = rng.uniform(-1, 1, (4, n))
    base = reference.expval(th, X, L, observable="global")
    for q in range(n):
        assert np.abs(reference.expval(th, X + 0.7 * np.eye(n)[q], L, observable="global") - base).max() > 1e-10


def test_adjoint_gradient_matches_finite_differences():
    rng = np.random.default_rng(2)
    n, L = 5, 2
    for obs in ("local", "global"):
        for reupload in (False, True):
            th = rng.uniform(0, 2 * np.pi, (L + 1, n))
            X = rng.uniform(-1, 1, (3, n))
            E, G = reference.expval_and_grad(th, X, L, reupload, obs)
            eps = 1e-6
            for l in range(L + 1):
                for q in range(n):
                    t2 = th.copy(); t2[l, q] += eps
                    fd = (reference.expval(t2, X, L, reupload, obs) - E) / eps
                    assert np.allclose(fd, G[l, q], atol=1e-5)


def test_known_single_qubit_value():
    # n=1, L=0: <Z> = cos(theta)
    th = np.array([[0.3]])
    assert np.isclose(reference.expval(th, np.zeros((1, 1)), 0)[0], np.cos(0.3))


def test_light_cone_from_center_is_twice_as_wide():
    rng = np.random.default_rng(3)
    n = 8
    for L in (1, 2, 3):
        th = rng.uniform(0, 2 * np.pi, (L + 1, n))
        X = rng.uniform(-1, 1, (5, n))
        base = reference.expval(th, X, L, measured_qubit=3)
        infl = [q for q in range(n)
                if np.abs(reference.expval(th, X + 0.7 * np.eye(n)[q], L, measured_qubit=3) - base).max() > 1e-10]
        assert infl == reference.light_cone_qubits(L, n, 3)
        assert len(infl) == min(2 * L + 1, n)
