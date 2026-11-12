"""Feature orderings: the proposed heuristic and the baselines.

Convention used everywhere in this package
------------------------------------------
An *ordering* is an integer array ``perm`` of length ``n_features`` such that
``X[:, perm]`` is the data actually fed to the circuit. In other words,
``perm[s]`` is the index of the original feature that goes into encoding
*slot* ``s``. Slots are mapped to qubits by the model (one feature per qubit
in the shallow circuit; three per qubit in the data re-uploading classifier),
so the model tells the heuristic which qubit each slot lives on.

Proposed heuristic
------------------
1. Rank features by mutual information with the label, estimated on the
   training split only.
2. Sort slots by graph distance of their qubit to the measured qubit.
3. Give the most informative feature to the closest slot, and so on.
Ties are broken deterministically (lower original index / lower slot index
first), so the heuristic always returns the same ordering for the same data.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif

from .topology import bfs_distances


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #
def mutual_information(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 0,
                       n_neighbors: int = 3) -> np.ndarray:
    """Mutual information I(x_j; y) for each feature (kNN estimator, sklearn)."""
    return mutual_info_classif(X_train, y_train, discrete_features=False,
                               n_neighbors=n_neighbors, random_state=random_state)


def abs_correlation(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """|Pearson correlation| between each feature and the label (constant features -> 0).

    A cheaper, lower-variance importance estimate than kNN mutual information,
    but it only sees linear (monotone) dependence. Used by 'heuristic_corr'.
    """
    X = np.asarray(X_train, float)
    y = np.asarray(y_train, float)
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    den = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, (Xc * yc[:, None]).sum(axis=0) / den, 0.0)
    return np.abs(r)


def mrmr_scores(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 0) -> np.ndarray:
    """Greedy mRMR (minimum redundancy, maximum relevance), Peng et al. (2005), MID form.

    Relevance = I(x_j; y); redundancy = mean I(x_j; x_s) over already selected
    features. Returns scores whose descending order is the mRMR selection order
    (so rank_features() reproduces it). Unlike univariate MI, two redundant
    features do not both land next to the measured qubit.
    """
    from sklearn.feature_selection import mutual_info_regression
    X = np.asarray(X_train, float)
    n = X.shape[1]
    rel = mutual_information(X, y_train, random_state=random_state)
    red = np.zeros((n, n))
    for j in range(n):
        red[:, j] = mutual_info_regression(X, X[:, j], random_state=random_state)
    selected, remaining = [], list(range(n))
    while remaining:
        if not selected:
            crit = [rel[j] for j in remaining]
        else:
            crit = [rel[j] - np.mean(red[j, selected]) for j in remaining]
        best = remaining[int(np.argmax(crit))]  # ties -> lower index (remaining is sorted)
        selected.append(best)
        remaining.remove(best)
    scores = np.empty(n)
    scores[selected] = np.arange(n, 0, -1, dtype=float)
    return scores


def rank_features(scores: np.ndarray) -> np.ndarray:
    """Feature indices from most to least informative; ties -> lower index first."""
    scores = np.asarray(scores, dtype=float)
    # lexsort sorts by last key first: primary = -score, secondary = index
    return np.lexsort((np.arange(len(scores)), -scores))


def slot_order(slot_qubits, graph, measured_qubit: int = 0) -> np.ndarray:
    """Slots sorted by distance of their qubit to the measured qubit (ties -> lower slot)."""
    dist = bfs_distances(graph, measured_qubit)
    d = np.array([dist[q] for q in slot_qubits])
    return np.lexsort((np.arange(len(d)), d))


# --------------------------------------------------------------------------- #
# Orderings
# --------------------------------------------------------------------------- #
def heuristic_ordering(X_train, y_train, slot_qubits, graph, measured_qubit: int = 0,
                       scores: np.ndarray | None = None, random_state: int = 0) -> np.ndarray:
    """Proposed ordering: most informative features on the slots closest to the measured qubit."""
    n = X_train.shape[1]
    if len(slot_qubits) != n:
        raise ValueError(f"{len(slot_qubits)} slots for {n} features")
    if scores is None:
        scores = mutual_information(X_train, y_train, random_state=random_state)
    ranked = rank_features(scores)
    slots = slot_order(slot_qubits, graph, measured_qubit)
    perm = np.empty(n, dtype=int)
    perm[slots] = ranked
    return perm


def anti_heuristic_ordering(X_train, y_train, slot_qubits, graph, measured_qubit: int = 0,
                            scores=None, random_state: int = 0) -> np.ndarray:
    """Worst case: most informative features on the slots farthest from the measured qubit."""
    n = X_train.shape[1]
    if scores is None:
        scores = mutual_information(X_train, y_train, random_state=random_state)
    ranked = rank_features(scores)[::-1]
    slots = slot_order(slot_qubits, graph, measured_qubit)
    perm = np.empty(n, dtype=int)
    perm[slots] = ranked
    return perm


def default_ordering(n: int) -> np.ndarray:
    return np.arange(n)


def reversed_ordering(n: int) -> np.ndarray:
    return np.arange(n)[::-1].copy()


def random_ordering(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n)


def swapped_ordering(n: int) -> np.ndarray:
    """Salmenperä et al. 'Swapped': second half first, i.e. np.roll(arange, n/2)."""
    return np.roll(np.arange(n), n // 2)


def every_other_ordering(n: int) -> np.ndarray:
    """Salmenperä et al. 'Every other': interleave first and second halves."""
    first = (n + 1) // 2
    perm = np.empty(n, dtype=int)
    perm[0::2] = np.arange(first)
    perm[1::2] = np.arange(first, n)
    return perm


def make_ordering(name: str, X_train, y_train, slot_qubits, graph, measured_qubit=0,
                  seed: int = 0, scores=None) -> np.ndarray:
    """Build an ordering by name.

    Names: 'heuristic' (mutual information, the proposal as registered),
    'heuristic_corr' (same placement rule, |correlation| as the importance
    estimate - an ablation of the estimator), 'heuristic_mrmr' (mRMR ranking,
    accounts for redundancy between features), 'anti_heuristic', 'default',
    'reversed', 'random', 'swapped', 'every_other'.

    scores: precomputed importances, either an array (mutual information) or a
    dict {'mi': array, 'corr': array}; missing ones are computed here.
    """
    n = X_train.shape[1]
    if isinstance(scores, dict):
        mi, corr = scores.get("mi"), scores.get("corr")
    else:
        mi, corr = scores, None
    if name == "heuristic":
        return heuristic_ordering(X_train, y_train, slot_qubits, graph, measured_qubit,
                                  scores=mi, random_state=0)
    if name == "heuristic_corr":
        if corr is None:
            corr = abs_correlation(X_train, y_train)
        return heuristic_ordering(X_train, y_train, slot_qubits, graph, measured_qubit,
                                  scores=corr)
    if name == "heuristic_mrmr":
        mr = scores.get("mrmr") if isinstance(scores, dict) else None
        if mr is None:
            mr = mrmr_scores(X_train, y_train)
        return heuristic_ordering(X_train, y_train, slot_qubits, graph, measured_qubit, scores=mr)
    if name == "anti_heuristic":
        return anti_heuristic_ordering(X_train, y_train, slot_qubits, graph, measured_qubit,
                                       scores=mi, random_state=0)
    if name == "default":
        return default_ordering(n)
    if name == "reversed":
        return reversed_ordering(n)
    if name == "random":
        return random_ordering(n, seed)
    if name == "swapped":
        return swapped_ordering(n)
    if name == "every_other":
        return every_other_ordering(n)
    raise ValueError(f"Unknown ordering '{name}'")


def is_permutation(perm) -> bool:
    perm = np.asarray(perm)
    return sorted(perm.tolist()) == list(range(len(perm)))
