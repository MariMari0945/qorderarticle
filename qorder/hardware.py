"""Evaluate trained shallow models with Qiskit: finite shots, noisy fake backends, IBM hardware.

Training never runs here. A model trained in the simulator is exported
(ShallowClassifier.export(), plus the feature ordering) and only its *test
predictions* are computed on the chosen backend:

    mode = "ideal"            exact sampling with shots (qiskit StatevectorSampler)
    mode = "fake:FakeFez"     noisy local simulation of an IBM device (qiskit-ibm-runtime fake provider + Aer)
    mode = "ibm:<backend>"    real IBM Quantum hardware (uses QPU time - check estimate_qpu_seconds first)

QPU time estimate (IBM's rule of thumb for Sampler jobs):
    seconds ~= 2 + 0.00035 * executions,   executions = shots * number of parameter sets
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from . import reference


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #
def estimate_qpu_seconds(n_samples: int, shots: int, n_jobs: int = 1) -> float:
    """IBM rule-of-thumb QPU seconds for n_jobs Sampler jobs of n_samples x shots each."""
    return n_jobs * (2.0 + 0.00035 * n_samples * shots)


def budget_table(n_models: int, n_samples: int, shots: int, monthly_seconds: float = 600.0):
    """How much of the monthly allowance a validation plan uses (one job per model)."""
    s = estimate_qpu_seconds(n_samples, shots, n_jobs=n_models)
    return {"n_models": n_models, "n_samples": n_samples, "shots": shots,
            "estimated_seconds": round(s, 1), "fraction_of_budget": round(s / monthly_seconds, 3)}


# --------------------------------------------------------------------------- #
# Circuit
# --------------------------------------------------------------------------- #
def shallow_qiskit_circuit(theta, n_qubits, n_layers, reupload=False, observable="local", measured_qubit=0):
    """Same circuit as models.ShallowClassifier / reference.expval, with inputs as parameters x[i]."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.circuit import ParameterVector

    theta = np.asarray(theta, float)
    x = ParameterVector("x", n_qubits)
    n_meas = 1 if observable == "local" else n_qubits
    qc = QuantumCircuit(QuantumRegister(n_qubits, "q"), ClassicalRegister(n_meas, "c"))
    for l in range(n_layers):
        for q in range(n_qubits):
            if l == 0 or reupload:
                qc.ry(x[q] + float(theta[l, q]), q)
            else:
                qc.ry(float(theta[l, q]), q)
        for q in range(0, n_qubits - 1, 2):
            qc.cz(q, q + 1)
        for q in range(1, n_qubits - 1, 2):
            qc.cz(q, q + 1)
    for q in range(n_qubits):
        qc.ry(float(theta[n_layers, q]), q)
    if observable == "local":
        qc.measure(measured_qubit, 0)
    else:
        qc.measure(range(n_qubits), range(n_qubits))
    return qc, x


def _expval_from_counts(counts: dict, observable: str) -> float:
    shots = sum(counts.values())
    if observable == "local":
        return (counts.get("0", 0) - counts.get("1", 0)) / shots
    tot = 0
    for bits, c in counts.items():
        tot += c * (1 if bits.count("1") % 2 == 0 else -1)
    return tot / shots


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
def get_sampler(mode: str, ibm_token: str | None = None, ibm_instance: str | None = None, seed: int | None = None):
    """Return (sampler, backend_or_None, description)."""
    if mode == "ideal":
        from qiskit.primitives import StatevectorSampler
        return StatevectorSampler(seed=seed), None, "ideal (StatevectorSampler)"
    if mode.startswith("fake:"):
        from qiskit_ibm_runtime import SamplerV2
        from qiskit_ibm_runtime import fake_provider
        name = mode.split(":", 1)[1]
        candidates = [name] if name != "auto" else ["FakeFez", "FakeMarrakesh", "FakeTorino",
                                                    "FakeSherbrooke", "FakeBrisbane", "FakeKyiv"]
        for cand in candidates:
            if hasattr(fake_provider, cand):
                backend = getattr(fake_provider, cand)()
                opts = {"simulator": {"seed_simulator": seed}} if seed is not None else None
                return SamplerV2(mode=backend, options=opts), backend, f"fake backend {cand} (noisy, local)"
        raise ValueError(f"none of {candidates} in qiskit_ibm_runtime.fake_provider")
    if mode.startswith("ibm:"):
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        name = mode.split(":", 1)[1]
        token = ibm_token or os.environ.get("IBM_QUANTUM_TOKEN")
        kw = {"channel": "ibm_quantum_platform"}
        if token:
            kw["token"] = token
        if ibm_instance or os.environ.get("IBM_QUANTUM_INSTANCE"):
            kw["instance"] = ibm_instance or os.environ.get("IBM_QUANTUM_INSTANCE")
        service = QiskitRuntimeService(**kw)
        backend = service.least_busy(operational=True, simulator=False) if name in ("", "least_busy") \
            else service.backend(name)
        return SamplerV2(mode=backend), backend, f"IBM hardware {backend.name}"
    raise ValueError(mode)


def transpile_for(qc, backend, seed=1234, optimization_level=3):
    if backend is None:
        return qc
    try:
        from qiskit.transpiler import generate_preset_pass_manager
    except ImportError:  # older Qiskit
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend,
                                      seed_transpiler=seed)
    return pm.run(qc)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def scale_inputs(export: dict, X_raw):
    """Apply the stored feature ordering and the train-fitted angle scaler."""
    X = np.asarray(X_raw, float)
    if export.get("perm") is not None:
        X = X[:, np.asarray(export["perm"])]
    return X * np.asarray(export["scaler_scale"]) + np.asarray(export["scaler_min"])


def evaluate(export: dict, X_test_raw, y_test, mode="ideal", shots=1000, max_samples=None,
             seed=1234, ibm_token=None, ibm_instance=None, budget_seconds=None,
             out_path=None):
    """Test accuracy of a trained shallow model on a backend, next to its exact-simulation accuracy.

    budget_seconds: refuse to submit to IBM hardware if the estimate exceeds it.
    """
    A = scale_inputs(export, X_test_raw)
    y = np.asarray(y_test)
    if max_samples is not None and max_samples < len(A):
        idx = np.random.default_rng(seed).choice(len(A), max_samples, replace=False)
        A, y = A[idx], y[idx]
    n, L = export["n_qubits"], export["n_layers"]
    obs, reup = export["observable"], export["reupload"]
    theta = np.asarray(export["theta"])
    est = estimate_qpu_seconds(len(A), shots)
    if mode.startswith("ibm:") and budget_seconds is not None and est > budget_seconds:
        raise RuntimeError(f"Estimated {est:.1f} s of QPU time exceeds budget {budget_seconds} s")

    sampler, backend, desc = get_sampler(mode, ibm_token, ibm_instance, seed=seed)
    mq = export.get("measured_qubit", 0)
    qc, _ = shallow_qiskit_circuit(theta, n, L, reup, obs, mq)
    isa = transpile_for(qc, backend, seed)
    t0 = time.time()
    job = sampler.run([(isa, A)], shots=shots)
    res = job.result()[0]
    bits = res.data.c
    E = np.array([_expval_from_counts(bits.get_counts(loc=i), obs) for i in range(len(A))])
    p = 1 / (1 + np.exp(-(export["a"] * E + export["b"])))
    E_exact = reference.expval(theta, A, L, reup, obs, mq)
    p_exact = 1 / (1 + np.exp(-(export["a"] * E_exact + export["b"])))
    out = {
        "mode": mode, "backend": desc, "shots": shots, "n_samples": int(len(A)),
        "accuracy": float(np.mean((p > 0.5) == y)),
        "accuracy_exact": float(np.mean((p_exact > 0.5) == y)),
        "mean_abs_expval_error": float(np.mean(np.abs(E - E_exact))),
        "estimated_qpu_seconds": round(est, 2),
        "wall_time_s": round(time.time() - t0, 2),
        "two_qubit_depth": int(isa.depth(lambda ins: ins.operation.num_qubits == 2)),
        "job_id": getattr(job, "job_id", lambda: None)() if mode.startswith("ibm:") else None,
    }
    if out_path:
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
    return out
