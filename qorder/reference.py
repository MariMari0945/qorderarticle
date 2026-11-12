"""Pure-NumPy reference implementation of the shallow circuit.

It has no quantum-library dependency, so it is used (i) in the unit tests,
to check the PennyLane and Qiskit versions against an independent
implementation, and (ii) for quick offline pilots. The experiments
themselves run on PennyLane (models.py).

Shallow circuit (one feature per qubit, n qubits, L layers)
-----------------------------------------------------------
for l in 0..L-1:
    RY(x_i)        on every qubit i     (only at l = 0, or every layer if reupload)
    RY(theta[l,i]) on every qubit i
    CZ on (0,1), (2,3), ...             (even bonds)
    CZ on (1,2), (3,4), ...             (odd bonds)
RY(theta[L,i]) on every qubit i         (final rotation; CZ commutes with Z, so
                                         without it the last CZs would be invisible)
measure  local:  <Z_0>      global: <Z_0 Z_1 ... Z_{n-1}>

All CZs of one layer commute (they are diagonal), so one layer acts as a
single diagonal entangler on the chain. The backward light cone of Z_0 grows
by one qubit per layer: after L layers it contains qubits 0..L, so features
on qubits > L cannot influence a local-cost model (checked in the tests).
"""
from __future__ import annotations

import numpy as np


def light_cone_qubits(n_layers: int, n_qubits: int, measured_qubit: int = 0) -> list:
    """Qubits that can influence <Z_m>: those within chain distance L of the measured qubit m."""
    return [q for q in range(n_qubits) if abs(q - measured_qubit) <= n_layers]


def light_cone_size(n_layers: int, n_qubits: int, measured_qubit: int = 0) -> int:
    """Number of qubits that can influence <Z_m> (L+1 at the end of the chain, up to 2L+1 in the middle)."""
    return len(light_cone_qubits(n_layers, n_qubits, measured_qubit))


def _ry(state, c, s, q, n):
    """Apply RY on qubit q. state: (M, 2**n); c, s: (M,) cos/sin of angle/2."""
    st = state.reshape(-1, 2 ** q, 2, 2 ** (n - q - 1))
    a, b = st[:, :, 0, :], st[:, :, 1, :]
    c = c[:, None, None]
    s = s[:, None, None]
    out = np.empty_like(st)
    out[:, :, 0, :] = c * a - s * b
    out[:, :, 1, :] = s * a + c * b
    return out.reshape(state.shape)


def _gen(state, q, n):
    """Apply dRY/dtheta * RY^-1 = [[0, -1/2], [1/2, 0]] on qubit q."""
    st = state.reshape(-1, 2 ** q, 2, 2 ** (n - q - 1))
    out = np.empty_like(st)
    out[:, :, 0, :] = -0.5 * st[:, :, 1, :]
    out[:, :, 1, :] = 0.5 * st[:, :, 0, :]
    return out.reshape(state.shape)


def _cz_sign(n):
    """Diagonal of one entangling layer: CZ on every bond (q, q+1) of the chain."""
    idx = (np.arange(2 ** n)[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1  # big-endian bits
    sign = np.ones(2 ** n)
    for a in range(n - 1):
        sign *= np.where((idx[:, a] == 1) & (idx[:, a + 1] == 1), -1.0, 1.0)
    return sign


def _observable_diag(n, observable, measured_qubit=0):
    idx = (np.arange(2 ** n)[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1
    if observable == "local":
        return np.where(idx[:, measured_qubit] == 0, 1.0, -1.0)
    if observable == "global":
        return np.prod(np.where(idx == 0, 1.0, -1.0), axis=1)
    raise ValueError(observable)


def _gate_list(theta, X, n_layers, reupload):
    """Sequence of ('ry', qubit, angles (B,), (l, q) param index) / ('cz',)."""
    B, n = X.shape
    gates = []
    for l in range(n_layers):
        for q in range(n):
            enc = X[:, q] if (l == 0 or reupload) else 0.0
            gates.append(("ry", q, theta[l, q] + enc + np.zeros(B), (l, q)))
        gates.append(("cz",))
    for q in range(n):
        gates.append(("ry", q, theta[n_layers, q] + np.zeros(B), (n_layers, q)))
    return gates


def _forward(gates, B, n):
    psi = np.zeros((B, 2 ** n))
    psi[:, 0] = 1.0
    sign = _cz_sign(n)
    for g in gates:
        if g[0] == "cz":
            psi = psi * sign
        else:
            psi = _ry(psi, np.cos(g[2] / 2), np.sin(g[2] / 2), g[1], n)
    return psi


def expval(theta, X, n_layers, reupload=False, observable="local", measured_qubit=0):
    """<O>(x) for every sample. theta: (L+1, n), X: (B, n) -> (B,)."""
    theta = np.asarray(theta, float)
    X = np.asarray(X, float)
    B, n = X.shape
    psi = _forward(_gate_list(theta, X, n_layers, reupload), B, n)
    return psi ** 2 @ _observable_diag(n, observable, measured_qubit)


def expval_and_grad(theta, X, n_layers, reupload=False, observable="local", measured_qubit=0):
    """<O>(x) and d<O>/dtheta per sample, by adjoint differentiation.

    returns E: (B,), G: (L+1, n, B)
    """
    theta = np.asarray(theta, float)
    X = np.asarray(X, float)
    B, n = X.shape
    gates = _gate_list(theta, X, n_layers, reupload)
    psi = _forward(gates, B, n)
    diag = _observable_diag(n, observable, measured_qubit)
    E = psi ** 2 @ diag
    lam = psi * diag
    sign = _cz_sign(n)
    G = np.zeros(theta.shape + (B,))
    for g in reversed(gates):
        if g[0] == "cz":
            psi = psi * sign
            lam = lam * sign
            continue
        _, q, ang, (l, qq) = g
        G[l, qq] += 2 * np.sum(lam * _gen(psi, q, n), axis=1)
        c, s = np.cos(ang / 2), np.sin(-ang / 2)
        psi = _ry(psi, c, s, q, n)
        lam = _ry(lam, c, s, q, n)
    return E, G


# --------------------------------------------------------------------------- #
# Tiny trainer (Adam, full batch) - pilots and tests only
# --------------------------------------------------------------------------- #
def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train_reference(X_train, y_train, n_layers, reupload=False, observable="local",
                    steps=200, lr=0.05, seed=0, measured_qubit=0):
    """Train theta and a linear readout p = sigmoid(a <O> + b) with binary cross-entropy."""
    rng = np.random.default_rng(seed)
    n = X_train.shape[1]
    theta = rng.uniform(0, 2 * np.pi, size=(n_layers + 1, n))
    a, b = 1.0, 0.0
    params = [theta, np.array(a), np.array(b)]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    y = np.asarray(y_train, float)
    history = []
    for t in range(1, steps + 1):
        E, G = expval_and_grad(params[0], X_train, n_layers, reupload, observable, measured_qubit)
        z = params[1] * E + params[2]
        p = np.clip(_sigmoid(z), 1e-9, 1 - 1e-9)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        history.append(loss)
        dz = (p - y) / len(y)
        grads = [np.einsum("b,lqb->lq", dz * params[1], G), np.sum(dz * E), np.sum(dz)]
        for i, g in enumerate(grads):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g ** 2
            params[i] = params[i] - lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return {"theta": params[0], "a": float(params[1]), "b": float(params[2]), "loss_history": history}


def predict_proba_reference(model, X, n_layers, reupload=False, observable="local", measured_qubit=0):
    E = expval(model["theta"], X, n_layers, reupload, observable, measured_qubit)
    return _sigmoid(model["a"] * E + model["b"])


class ShallowReferenceClassifier:
    """Same interface as models.ShallowClassifier, trained with the NumPy simulator.

    Full-batch Adam with a fixed number of steps. Slower than PennyLane+JAX for
    big sweeps, but needs nothing beyond NumPy - used for offline pilots and to
    test the runner end to end.
    """

    def __init__(self, n_qubits=8, n_layers=2, observable="local", reupload=False,
                 learning_rate=0.1, max_steps=150, random_state=0, measured_qubit=0, **_ignored):
        self.measured_qubit = measured_qubit
        self.n_qubits, self.n_layers = n_qubits, n_layers
        self.observable, self.reupload = observable, reupload
        self.learning_rate, self.max_steps = learning_rate, max_steps
        self.random_state = random_state

    @property
    def slot_qubits(self):
        return list(range(self.n_qubits))

    def fit(self, X, y):
        from .datasets import angle_scaler
        self.scaler_ = angle_scaler(X)
        A = self.scaler_.transform(X)
        self.model_ = train_reference(A, y, self.n_layers, self.reupload, self.observable,
                                      steps=self.max_steps, lr=self.learning_rate, seed=self.random_state,
                                      measured_qubit=self.measured_qubit)
        h = self.model_["loss_history"]
        from .models_common import steps_to_converge
        self.info_ = {"steps": len(h), "steps_to_converge": steps_to_converge(h), "converged": False,
                      "loss_history": np.array(h)}
        return self

    def predict_proba(self, X):
        return predict_proba_reference(self.model_, self.scaler_.transform(X), self.n_layers,
                                       self.reupload, self.observable, self.measured_qubit)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def loss(self, X, y):
        p = np.clip(self.predict_proba(X), 1e-9, 1 - 1e-9)
        y = np.asarray(y, float)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def export(self):
        return {"kind": "shallow", "n_qubits": self.n_qubits, "n_layers": self.n_layers,
                "observable": self.observable, "reupload": self.reupload, "measured_qubit": self.measured_qubit,
                "theta": np.asarray(self.model_["theta"]).tolist(),
                "a": self.model_["a"], "b": self.model_["b"],
                "scaler_min": self.scaler_.min_.tolist(), "scaler_scale": self.scaler_.scale_.tolist()}


# --------------------------------------------------------------------------- #
# Training with finite-shot gradient estimates (round 4)
# --------------------------------------------------------------------------- #
class ShallowShotsClassifier(ShallowReferenceClassifier):
    """Shallow classifier trained with gradients estimated from a finite number of shots.

    Same circuit, initialisation, readout p = sigmoid(a <O> + b), loss, minibatch Adam
    and convergence rule as models.ShallowClassifier, but the quantities a device would
    estimate are noisy:

    * each expectation value <O>(x) is estimated from `shots` samples of the measured
      qubit (binomial sampling, variance (1 - <O>^2) / shots);
    * each parameter derivative is estimated by the parameter-shift rule,
      d<O>/dtheta = (<O>_+ - <O>_-) / 2, from two such estimates; its noise variance
      is approximated by (1 - <O>^2) / (2 shots) (exact when <O>_+- ~ <O>).

    With shots=None the gradients are exact, which gives a control trained with the
    very same loop. Test predictions are always computed exactly, so that the effect
    measured is the effect of noisy gradients on training.
    """

    def __init__(self, n_qubits=8, n_layers=2, observable="local", reupload=False,
                 learning_rate=0.05, batch_size=64, max_steps=1500, convergence_interval=100,
                 shots=None, random_state=0, measured_qubit=0, **_ignored):
        super().__init__(n_qubits=n_qubits, n_layers=n_layers, observable=observable, reupload=reupload,
                         learning_rate=learning_rate, max_steps=max_steps, random_state=random_state,
                         measured_qubit=measured_qubit)
        self.batch_size, self.convergence_interval = batch_size, convergence_interval
        self.shots = None if shots in (None, 0, "exact", "none") else int(shots)

    def fit(self, X, y):
        from .datasets import angle_scaler
        from .models_common import steps_to_converge
        self.scaler_ = angle_scaler(X)
        A = self.scaler_.transform(X)
        y = np.asarray(y, float)
        rng = np.random.default_rng(self.random_state)
        theta = rng.uniform(0, 2 * np.pi, size=(self.n_layers + 1, self.n_qubits))  # same init as ShallowClassifier
        noise = np.random.default_rng(10_000 + self.random_state)  # shot noise, independent of batches
        params = [theta, np.array(1.0), np.array(0.0)]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        n = len(A)
        bs = n if (self.batch_size is None or self.batch_size >= n) else self.batch_size
        hist, converged, ci = [], False, self.convergence_interval
        for t in range(1, self.max_steps + 1):
            idx = rng.choice(n, size=bs, replace=False) if bs < n else np.arange(n)
            E, G = expval_and_grad(params[0], A[idx], self.n_layers, self.reupload, self.observable,
                                   self.measured_qubit)
            if self.shots:
                var = np.clip(1 - E ** 2, 0, 1) / self.shots
                G = G + noise.normal(size=G.shape) * np.sqrt(var / 2)[None, None, :]
                k = noise.binomial(self.shots, np.clip((1 + E) / 2, 0, 1))  # exact sampling of <O>
                E = 2 * k / self.shots - 1
            z = params[1] * E + params[2]
            p = np.clip(_sigmoid(z), 1e-9, 1 - 1e-9)
            yb = y[idx]
            loss = float(-np.mean(yb * np.log(p) + (1 - yb) * np.log(1 - p)))
            hist.append(loss)
            dz = (p - yb) / len(yb)
            grads = [np.einsum("b,lqb->lq", dz * params[1], G), np.sum(dz * E), np.sum(dz)]
            for i, g in enumerate(grads):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g ** 2
                params[i] = params[i] - self.learning_rate * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
            if ci and len(hist) > 2 * ci:
                a1, a2 = np.mean(hist[-ci:]), np.mean(hist[-2 * ci:-ci])
                if abs(a2 - a1) <= np.std(hist[-ci:]) / np.sqrt(ci) / 2:
                    converged = True
                    break
        self.model_ = {"theta": params[0], "a": float(params[1]), "b": float(params[2]), "loss_history": hist}
        E_train = expval(params[0], A, self.n_layers, self.reupload, self.observable, self.measured_qubit)
        self.info_ = {"steps": len(hist), "steps_to_converge": steps_to_converge(hist), "converged": converged,
                      "loss_history": np.array(hist),
                      # margin of the trained circuit: shot noise has variance (1 - <O>^2)/shots,
                      # so outputs far from 0 are less affected by it
                      "mean_abs_expval": float(np.mean(np.abs(E_train)))}
        return self
