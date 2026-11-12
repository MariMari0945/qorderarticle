"""Quantum classifiers (PennyLane + JAX).

ShallowClassifier
    The 8-qubit circuit of the second experimental front: RY encoding, RY
    variational layers, CZ on every bond of a linear chain, final RY layer,
    local (<Z_0>) or global (<Z_0 ... Z_{n-1}>) cost. Same circuit as
    qorder.reference and qorder.hardware (checked in tests/test_equivalence.py).

DataReuploadingClassifier
    Port of the data re-uploading classifier used by Salmenperä, Perkkola &
    Nurminen (2025), originally from Xanadu's qml-benchmarks (Apache-2.0).
    Updated for current PennyLane (qml.broadcast was removed); the circuit,
    loss (eqn. 23 of Pérez-Salinas et al.), initialisation, scaling and
    convergence rule follow the original.

Both expose fit / predict / predict_proba / score, take labels in {0, 1}
and tell the ordering heuristic which qubit each input slot sits on
(`slot_qubits`).
"""
from __future__ import annotations

import time
import warnings
from functools import reduce
from math import ceil
from operator import matmul

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import optax
    import pennylane as qml

    jax.config.update("jax_enable_x64", True)
except ImportError as e:  # pragma: no cover
    raise ImportError("models.py needs pennylane, jax and optax: pip install -r requirements.txt") from e

from .datasets import angle_scaler
from .models_common import steps_to_converge


# --------------------------------------------------------------------------- #
# Generic optax training loop (minibatches + plateau convergence rule)
# --------------------------------------------------------------------------- #
def train_optax(params, loss_fn, X, y, *, learning_rate, batch_size, max_steps,
                convergence_interval, seed, X_val=None, y_val=None, log_every=0):
    """Adam on minibatches; stops when the loss curve is flat (QMLPerm/qml-benchmarks rule).

    Convergence: after 2*interval steps, stop when
    |mean(last interval) - mean(previous interval)| <= std(last interval) / sqrt(interval) / 2.
    Returns (params, info) with the loss history and whether it converged.
    """
    opt = optax.adam(learning_rate)
    opt_state = opt.init(params)
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    val_loss = jax.jit(loss_fn) if X_val is not None else None
    rng = np.random.default_rng(seed)
    n = len(X)
    bs = n if (batch_size is None or batch_size >= n) else batch_size
    hist, val_hist = [], []
    converged = False
    t0 = time.time()
    for step in range(max_steps):
        idx = rng.choice(n, size=bs, replace=False) if bs < n else np.arange(n)
        loss, grads = value_and_grad(params, X[idx], y[idx])
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        loss = float(loss)
        hist.append(loss)
        if not np.isfinite(loss):
            warnings.warn("NaN loss, training aborted")
            break
        if val_loss is not None and (step % 10 == 0):
            val_hist.append((step, float(val_loss(params, X_val, y_val))))
        if log_every and step % log_every == 0:
            print(f"step {step:5d}  loss {loss:.4f}")
        ci = convergence_interval
        if ci and step > 2 * ci:
            a1 = np.mean(hist[-ci:])
            a2 = np.mean(hist[-2 * ci:-ci])
            if abs(a2 - a1) <= np.std(hist[-ci:]) / np.sqrt(ci) / 2:
                converged = True
                break
    info = {"loss_history": np.array(hist), "val_history": val_hist, "converged": converged,
            "steps": len(hist), "train_time_s": time.time() - t0,
            "steps_to_converge": steps_to_converge(hist)}
    if not converged and convergence_interval:
        warnings.warn(f"did not converge in {max_steps} steps")
    return params, info


def _bce(p, y):
    p = jnp.clip(p, 1e-9, 1 - 1e-9)
    return -jnp.mean(y * jnp.log(p) + (1 - y) * jnp.log(1 - p))


# --------------------------------------------------------------------------- #
# Shallow circuit
# --------------------------------------------------------------------------- #
def shallow_circuit_ops(theta, x, n_qubits, n_layers, reupload):
    """Gate sequence of the shallow circuit (x may be batched: shape (B, n))."""
    xq = (lambda q: x[..., q])
    for l in range(n_layers):
        for q in range(n_qubits):
            if l == 0 or reupload:
                qml.RY(xq(q), wires=q)
            qml.RY(theta[l, q], wires=q)
        for q in range(0, n_qubits - 1, 2):
            qml.CZ(wires=[q, q + 1])
        for q in range(1, n_qubits - 1, 2):
            qml.CZ(wires=[q, q + 1])
    for q in range(n_qubits):
        qml.RY(theta[n_layers, q], wires=q)


def observable_op(n_qubits, observable, measured_qubit=0):
    if observable == "local":
        return qml.PauliZ(measured_qubit)
    if observable == "global":
        return reduce(matmul, [qml.PauliZ(q) for q in range(n_qubits)])
    raise ValueError(observable)


class ShallowClassifier:
    """Shallow VQC with a local or global cost and linear readout p = sigmoid(a <O> + b)."""

    def __init__(self, n_qubits=8, n_layers=2, observable="local", reupload=False,
                 learning_rate=0.05, batch_size=64, max_steps=1000, convergence_interval=100,
                 init="uniform", random_state=0, device="default.qubit", measured_qubit=0):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.observable = observable
        self.reupload = reupload
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.convergence_interval = convergence_interval
        self.init = init
        self.random_state = random_state
        self.device = device
        self.measured_qubit = measured_qubit
        self.params_ = None
        self.scaler_ = None
        self._build()

    # the heuristic needs to know where each input slot lives
    @property
    def slot_qubits(self):
        return list(range(self.n_qubits))

    def _build(self):
        dev = qml.device(self.device, wires=self.n_qubits)
        n, L, reup, obs, mq = self.n_qubits, self.n_layers, self.reupload, self.observable, self.measured_qubit

        @qml.qnode(dev, interface="jax", diff_method="backprop")
        def circuit(theta, X):
            shallow_circuit_ops(theta, X, n, L, reup)
            return qml.expval(observable_op(n, obs, mq))

        self.circuit = circuit
        self._expval = jax.jit(circuit)

    def init_theta(self, key_seed):
        rng = np.random.default_rng(key_seed)
        shape = (self.n_layers + 1, self.n_qubits)
        if self.init == "uniform":
            return rng.uniform(0, 2 * np.pi, size=shape)
        if self.init == "small":
            return rng.normal(0, 0.1, size=shape)
        if self.init == "zeros":
            return np.zeros(shape)
        raise ValueError(self.init)

    def _proba(self, params, X):
        return jax.nn.sigmoid(params["a"] * self.circuit(params["theta"], X) + params["b"])

    def fit(self, X, y, X_val=None, y_val=None):
        self.scaler_ = angle_scaler(X)
        A = jnp.asarray(self.scaler_.transform(X))
        y = jnp.asarray(y, dtype=float)
        Av = None if X_val is None else jnp.asarray(self.scaler_.transform(X_val))
        yv = None if y_val is None else jnp.asarray(y_val, dtype=float)
        params = {"theta": jnp.asarray(self.init_theta(self.random_state)),
                  "a": jnp.asarray(1.0), "b": jnp.asarray(0.0)}

        def loss_fn(p, Xb, yb):
            return _bce(self._proba(p, Xb), yb)

        self.loss_fn_ = loss_fn
        params, self.info_ = train_optax(
            params, loss_fn, A, y, learning_rate=self.learning_rate, batch_size=self.batch_size,
            max_steps=self.max_steps, convergence_interval=self.convergence_interval,
            seed=self.random_state, X_val=Av, y_val=yv)
        self.params_ = {k: np.asarray(v) for k, v in params.items()}
        return self

    def transform(self, X):
        return self.scaler_.transform(X)

    def expval(self, X):
        return np.asarray(self._expval(jnp.asarray(self.params_["theta"]), jnp.asarray(self.transform(X))))

    def predict_proba(self, X):
        E = self.expval(X)
        return 1 / (1 + np.exp(-(self.params_["a"] * E + self.params_["b"])))

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def loss(self, X, y):
        p = np.clip(self.predict_proba(X), 1e-9, 1 - 1e-9)
        y = np.asarray(y, float)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def export(self):
        """Everything needed to rebuild the trained circuit elsewhere (Qiskit, hardware)."""
        return {"kind": "shallow", "n_qubits": self.n_qubits, "n_layers": self.n_layers,
                "observable": self.observable, "reupload": self.reupload, "measured_qubit": self.measured_qubit,
                "theta": np.asarray(self.params_["theta"]).tolist(),
                "a": float(self.params_["a"]), "b": float(self.params_["b"]),
                "scaler_min": self.scaler_.min_.tolist(), "scaler_scale": self.scaler_.scale_.tolist()}


# --------------------------------------------------------------------------- #
# Data re-uploading classifier (Salmenperä et al. / qml-benchmarks)
# --------------------------------------------------------------------------- #
class DataReuploadingClassifier:
    """Multi-qubit data re-uploading classifier, 3 features per qubit via qml.Rot."""

    def __init__(self, n_layers=5, observable_type="single", entangling_layer="block",
                 learning_rate=0.01, batch_size=32, max_steps=10000, convergence_interval=600,
                 random_state=42, device="default.qubit"):
        self.n_layers = n_layers
        self.observable_type = observable_type
        self.entangling_layer = entangling_layer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.convergence_interval = convergence_interval
        self.random_state = random_state
        self.device = device
        self.rng = np.random.default_rng(random_state)
        self.measured_qubit = 0
        self.params_ = None

    def slot_qubits_for(self, n_features):
        """Feature slot s goes to qubit s // 3."""
        return [s // 3 for s in range(n_features)]

    def _entangle(self, layer):
        n = self.n_qubits_
        if self.entangling_layer == "block":
            start = 0 if layer % 2 == 0 else 1  # qml.broadcast 'double' / 'double_odd'
            for q in range(start, n - 1, 2):
                qml.CZ(wires=[q, q + 1])
        elif self.entangling_layer in ("ring", "circle"):
            if n == 2:
                qml.CZ(wires=[0, 1])
            elif n > 2:
                for q in range(n):
                    qml.CZ(wires=[q, (q + 1) % n])
        else:
            raise ValueError(self.entangling_layer)

    def _build(self):
        n = self.n_qubits_
        dev = qml.device(self.device, wires=n)

        @qml.qnode(dev, interface="jax")
        def circuit(params, x):
            for layer in range(self.n_layers):
                for i in range(n):
                    ang = x[3 * i:3 * i + 3] * params["omegas"][layer, 3 * i:3 * i + 3]
                    qml.Rot(*ang, wires=i)
                    qml.Rot(*params["thetas"][i, layer, :], wires=i)
                self._entangle(layer)
            for i in range(n):
                ang = (x[3 * i:3 * i + 3] * params["omegas"][self.n_layers, 3 * i:3 * i + 3]
                       + params["thetas"][i, self.n_layers, :])
                qml.Rot(*ang, wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(n)]

        def as_array(params, x):
            return jnp.array(circuit(params, x))

        self.circuit = circuit
        self.forward = jax.jit(jax.vmap(as_array, in_axes=(None, 0)))

    def _key(self):
        return jax.random.PRNGKey(int(self.rng.integers(100000)))

    def _transform(self, X):
        X = self.scaler_.transform(X)
        return jnp.pad(jnp.asarray(X), ((0, 0), (0, (3 - X.shape[1]) % 3)))

    def fit(self, X, y):
        y = np.asarray(y)
        y_pm = jnp.asarray(np.where(y == 1, 1.0, -1.0))
        self.n_features_ = X.shape[1]
        self.n_qubits_ = ceil(self.n_features_ / 3)
        self.observable_weight = {"single": 1, "half": max(1, self.n_qubits_ // 2),
                                  "full": self.n_qubits_}[self.observable_type]
        self._build()
        self.params_ = {
            "thetas": jax.random.uniform(self._key(), shape=(self.n_qubits_, self.n_layers + 1, 3)),
            "omegas": jax.random.uniform(self._key(), shape=(self.n_layers + 1, self.n_qubits_ * 3)),
            "alphas": jnp.ones(shape=(2, self.observable_weight)),
        }
        self.scaler_ = angle_scaler(X)
        A = self._transform(X)
        w = self.observable_weight

        def loss_fn(params, x, yb):
            ym = jnp.tile(yb[:, None], (1, w))
            y0, y1 = (1 - ym) / 2, (1 + ym) / 2
            ev = self.forward(params, x)[:, :w]
            p0, p1 = (1 - ev) / 2, (1 + ev) / 2  # as in the original code
            loss = 0.5 * jnp.sum((params["alphas"][0] * p0 - y0) ** 2 + (params["alphas"][1] * p1 - y1) ** 2)
            return loss / len(yb)

        params, self.info_ = train_optax(
            self.params_, loss_fn, A, y_pm, learning_rate=self.learning_rate,
            batch_size=self.batch_size, max_steps=self.max_steps,
            convergence_interval=self.convergence_interval, seed=int(self.rng.integers(1 << 31)))
        self.params_ = params
        return self

    def predict_proba(self, X):
        ev = np.asarray(self.forward(self.params_, self._transform(X)))[:, :self.observable_weight]
        return (1 + np.mean(ev, axis=1)) / 2  # P(class 1); class 1 <-> label +1

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def loss(self, X, y):
        p = np.clip(self.predict_proba(X), 1e-9, 1 - 1e-9)
        y = np.asarray(y, float)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def build_model(cfg: dict):
    """Model from a config dict: {'kind': 'shallow'|'drc', ...kwargs}."""
    cfg = dict(cfg)
    kind = cfg.pop("kind")
    if kind == "shallow":
        return ShallowClassifier(**cfg)
    if kind == "drc":
        return DataReuploadingClassifier(**cfg)
    raise ValueError(kind)
