"""PennyLane and Qiskit implementations must match the NumPy reference.

Skipped automatically where those libraries are not installed.
"""
import numpy as np
import pytest

from qorder import reference

CASES = [(8, 1, False, "local"), (8, 3, True, "local"), (6, 2, False, "global")]


@pytest.mark.parametrize("n,L,reupload,obs", CASES)
def test_pennylane_matches_reference(n, L, reupload, obs):
    pytest.importorskip("pennylane")
    pytest.importorskip("jax")
    import jax.numpy as jnp
    import pennylane as qml

    from qorder.models import observable_op, shallow_circuit_ops

    rng = np.random.default_rng(0)
    th = rng.uniform(0, 2 * np.pi, (L + 1, n))
    X = rng.uniform(-1.5, 1.5, (5, n))
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, interface="jax")
    def circ(t, x):
        shallow_circuit_ops(t, x, n, L, reupload)
        return qml.expval(observable_op(n, obs))

    got = np.asarray(circ(jnp.asarray(th), jnp.asarray(X)))
    assert np.allclose(got, reference.expval(th, X, L, reupload, obs), atol=1e-8)


@pytest.mark.parametrize("n,L,reupload,obs", CASES)
def test_qiskit_matches_reference(n, L, reupload, obs):
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import SparsePauliOp, Statevector

    from qorder.hardware import shallow_qiskit_circuit

    rng = np.random.default_rng(0)
    th = rng.uniform(0, 2 * np.pi, (L + 1, n))
    X = rng.uniform(-1.5, 1.5, (3, n))
    qc, x = shallow_qiskit_circuit(th, n, L, reupload, obs)
    qc = qc.remove_final_measurements(inplace=False)
    # qiskit is little-endian: qubit 0 is the rightmost Pauli label
    label = ("I" * (n - 1) + "Z") if obs == "local" else "Z" * n
    op = SparsePauliOp(label)
    ref = reference.expval(th, X, L, reupload, obs)
    for i in range(len(X)):
        bound = qc.assign_parameters(dict(zip(x, X[i])))
        assert np.isclose(Statevector(bound).expectation_value(op).real, ref[i], atol=1e-8)


def test_shallow_classifier_trains():
    pytest.importorskip("pennylane")
    from qorder import datasets
    from qorder.models import ShallowClassifier

    X, y, Xt, yt = datasets.breast_cancer()
    m = ShallowClassifier(n_layers=1, max_steps=150, convergence_interval=0).fit(X, y)
    assert m.score(Xt, yt) > 0.75


def test_drc_runs_on_adversarial():
    pytest.importorskip("pennylane")
    from qorder import datasets
    from qorder.models import DataReuploadingClassifier

    X, y, Xt, yt = datasets.adversarial(n_train=64, n_test=64)
    m = DataReuploadingClassifier(n_layers=2, max_steps=20, convergence_interval=0).fit(X, y)
    assert 0.0 <= m.score(Xt, yt) <= 1.0


@pytest.mark.parametrize("mq", [3, 5])
def test_measured_qubit_matches_reference(mq):
    pytest.importorskip("pennylane")
    pytest.importorskip("qiskit")
    import jax.numpy as jnp
    import pennylane as qml
    from qiskit.quantum_info import SparsePauliOp, Statevector

    from qorder.hardware import shallow_qiskit_circuit
    from qorder.models import observable_op, shallow_circuit_ops

    n, L = 8, 2
    rng = np.random.default_rng(1)
    th = rng.uniform(0, 2 * np.pi, (L + 1, n))
    X = rng.uniform(-1.5, 1.5, (3, n))
    ref = reference.expval(th, X, L, measured_qubit=mq)
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, interface="jax")
    def circ(t, x):
        shallow_circuit_ops(t, x, n, L, False)
        return qml.expval(observable_op(n, "local", mq))

    assert np.allclose(np.asarray(circ(jnp.asarray(th), jnp.asarray(X))), ref, atol=1e-8)
    qc, x = shallow_qiskit_circuit(th, n, L, measured_qubit=mq)
    qc = qc.remove_final_measurements(inplace=False)
    label = "".join("Z" if q == mq else "I" for q in reversed(range(n)))
    for i in range(len(X)):
        sv = Statevector(qc.assign_parameters(dict(zip(x, X[i]))))
        assert np.isclose(sv.expectation_value(SparsePauliOp(label)).real, ref[i], atol=1e-8)
