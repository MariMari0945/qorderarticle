"""Qubit connectivity graphs and graph distances.

A topology is an adjacency dict ``{qubit: set(neighbours)}``. Distances are
shortest-path (BFS) hop counts, which is what the ordering heuristic uses to
decide which qubits are "close" to the measured qubit.
"""
from __future__ import annotations

from collections import deque


def linear_chain(n_qubits: int) -> dict[int, set[int]]:
    """0 - 1 - 2 - ... - (n-1)."""
    return {q: {p for p in (q - 1, q + 1) if 0 <= p < n_qubits} for q in range(n_qubits)}


def ring(n_qubits: int) -> dict[int, set[int]]:
    """Linear chain with the extra edge (n-1) - 0."""
    g = linear_chain(n_qubits)
    if n_qubits > 2:
        g[0].add(n_qubits - 1)
        g[n_qubits - 1].add(0)
    return g


def grid(rows: int, cols: int) -> dict[int, set[int]]:
    """Rectangular grid, qubit index = r * cols + c."""
    g: dict[int, set[int]] = {}
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            nb = set()
            if r > 0:
                nb.add(q - cols)
            if r < rows - 1:
                nb.add(q + cols)
            if c > 0:
                nb.add(q - 1)
            if c < cols - 1:
                nb.add(q + 1)
            g[q] = nb
    return g


def from_edges(n_qubits: int, edges) -> dict[int, set[int]]:
    """Build a topology from an edge list, e.g. a hardware coupling map."""
    g = {q: set() for q in range(n_qubits)}
    for a, b in edges:
        g[a].add(b)
        g[b].add(a)
    return g


def bfs_distances(graph: dict[int, set[int]], source: int) -> dict[int, int]:
    """Hop distance from ``source`` to every reachable qubit."""
    dist = {source: 0}
    queue = deque([source])
    while queue:
        q = queue.popleft()
        for nb in sorted(graph[q]):
            if nb not in dist:
                dist[nb] = dist[q] + 1
                queue.append(nb)
    missing = set(graph) - set(dist)
    if missing:
        raise ValueError(f"Qubits {sorted(missing)} are not connected to qubit {source}.")
    return dist


def get_topology(name: str, n_qubits: int) -> dict[int, set[int]]:
    """Look up a topology by name ('linear', 'ring', 'grid:RxC')."""
    if name == "linear":
        return linear_chain(n_qubits)
    if name == "ring":
        return ring(n_qubits)
    if name.startswith("grid:"):
        r, c = (int(v) for v in name.split(":")[1].split("x"))
        if r * c != n_qubits:
            raise ValueError(f"grid {r}x{c} does not have {n_qubits} qubits")
        return grid(r, c)
    raise ValueError(f"Unknown topology '{name}'")
