"""Gradient-variance measurements (H3, and the barren-plateau sanity check).

For each configuration, sample `n_init` random parameter vectors
theta ~ U[0, 2pi), compute d<O>/dtheta on a batch of inputs, and report the
variance across initialisations. Two summaries are kept:

* var_first  - Var[d<O>/dtheta_{0,0}] (first-layer parameter on the measured
               qubit, the usual choice in barren-plateau papers);
* var_mean   - Var averaged over all parameters (parameters outside the
               light cone have zero gradient and pull this down - that is
               the point of reporting both).

backend='reference' uses the NumPy simulator and runs anywhere;
backend='pennylane' uses JAX autodiff (faster for many configurations).
"""
from __future__ import annotations

import numpy as np

from . import reference


def _grads_reference(theta, X, n_layers, reupload, observable):
    _, G = reference.expval_and_grad(theta, X, n_layers, reupload, observable)
    return G.mean(axis=-1)  # average over inputs -> (L+1, n)


def _grads_pennylane_factory(n_qubits, n_layers, reupload, observable):
    import jax
    import jax.numpy as jnp
    import pennylane as qml

    from .models import observable_op, shallow_circuit_ops

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="jax", diff_method="backprop")
    def circuit(theta, X):
        shallow_circuit_ops(theta, X, n_qubits, n_layers, reupload)
        return qml.expval(observable_op(n_qubits, observable))

    g = jax.jit(jax.grad(lambda t, X: jnp.mean(circuit(t, X))))
    return lambda theta, X: np.asarray(g(jnp.asarray(theta), jnp.asarray(X)))


def gradient_variance(n_qubits=8, n_layers=2, observable="local", reupload=False,
                      X=None, n_init=200, n_inputs=16, seed=0, backend="reference", n_boot=1000):
    """Variance of the cost gradient over random initialisations.

    X: input angles (already scaled), shape (m, n_qubits); a random subset of
    `n_inputs` rows is used per initialisation. If None, inputs are U[-pi/2, pi/2].
    """
    rng = np.random.default_rng(seed)
    if X is None:
        X = rng.uniform(-np.pi / 2, np.pi / 2, size=(max(n_inputs, 64), n_qubits))
    X = np.asarray(X, float)
    if backend == "pennylane":
        grad_fn = _grads_pennylane_factory(n_qubits, n_layers, reupload, observable)
    else:
        grad_fn = lambda t, Xb: _grads_reference(t, Xb, n_layers, reupload, observable)  # noqa: E731
    grads = []
    for _ in range(n_init):
        theta = rng.uniform(0, 2 * np.pi, size=(n_layers + 1, n_qubits))
        Xb = X[rng.choice(len(X), size=min(n_inputs, len(X)), replace=False)]
        grads.append(grad_fn(theta, Xb))
    G = np.stack(grads)  # (n_init, L+1, n)
    var = G.var(axis=0)
    # bootstrap 95% CI of Var[dC/dtheta_00] over the random initialisations
    g00 = G[:, 0, 0]
    boot = np.array([rng.choice(g00, size=len(g00), replace=True).var() for _ in range(n_boot)])
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    cone = reference.light_cone_size(n_layers, n_qubits)
    return {
        "n_qubits": n_qubits, "n_layers": n_layers, "observable": observable, "reupload": reupload,
        "var_first": float(var[0, 0]),
        "var_first_ci_low": float(ci_low), "var_first_ci_high": float(ci_high),
        "var_mean": float(var.mean()),
        "var_mean_in_cone": float(var[:, :cone].mean()) if observable == "local" else float(var.mean()),
        "light_cone": cone, "n_init": n_init,
    }


def sweep(depths=(1, 2, 3, 5, 7), observables=("local", "global"), n_qubits_list=(8,), **kw):
    """Gradient variance for every (n_qubits, depth, observable)."""
    rows = []
    for n in n_qubits_list:
        for L in depths:
            for obs in observables:
                rows.append(gradient_variance(n_qubits=n, n_layers=L, observable=obs, **kw))
    return rows
