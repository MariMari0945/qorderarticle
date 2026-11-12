import numpy as np

from qorder import ordering, topology


def test_bfs_linear_and_ring():
    assert topology.bfs_distances(topology.linear_chain(5), 0) == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    d = topology.bfs_distances(topology.ring(6), 0)
    assert d[5] == 1 and d[3] == 3
    d = topology.bfs_distances(topology.grid(2, 3), 0)
    assert d[5] == 3 and d[3] == 1


def test_heuristic_puts_informative_features_near_measured_qubit():
    scores = np.array([0.1, 0.9, 0.0, 0.5])
    X = np.zeros((10, 4)); y = np.zeros(10)
    perm = ordering.heuristic_ordering(X, y, [0, 1, 2, 3], topology.linear_chain(4), 0, scores=scores)
    assert perm.tolist() == [1, 3, 0, 2]
    # measured qubit in the middle of the chain: slots at distance 1 are 1 and 3
    perm = ordering.heuristic_ordering(X, y, [0, 1, 2, 3], topology.linear_chain(4), 2, scores=scores)
    assert perm[2] == 1 and set(perm[[1, 3]]) == {3, 0} and perm[0] == 2


def test_ties_are_deterministic():
    scores = np.array([0.5, 0.5, 0.5])
    X = np.zeros((4, 3)); y = np.zeros(4)
    perm = ordering.heuristic_ordering(X, y, [0, 1, 2], topology.linear_chain(3), 0, scores=scores)
    assert perm.tolist() == [0, 1, 2]


def test_drc_slots_three_per_qubit():
    scores = np.arange(6)[::-1] / 10  # feature 0 best
    X = np.zeros((4, 6)); y = np.zeros(4)
    slots = [s // 3 for s in range(6)]
    perm = ordering.heuristic_ordering(X, y, slots, topology.linear_chain(2), 0, scores=scores)
    assert set(perm[:3]) == {0, 1, 2}


def test_salmenpera_handpicked_orderings():
    assert ordering.swapped_ordering(20).tolist() == [(i + 10) % 20 for i in range(20)]
    eo = ordering.every_other_ordering(20).tolist()
    assert eo[:4] == [0, 10, 1, 11] and eo[-2:] == [9, 19]
    assert ordering.is_permutation(ordering.every_other_ordering(7))


def test_all_named_orderings_are_permutations():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 8)); y = (X[:, 3] > 0).astype(int)
    g = topology.linear_chain(8)
    for name in ["heuristic", "anti_heuristic", "default", "reversed", "random", "swapped", "every_other"]:
        assert ordering.is_permutation(ordering.make_ordering(name, X, y, list(range(8)), g))
    # the only informative feature must land on the measured qubit
    assert ordering.make_ordering("heuristic", X, y, list(range(8)), g)[0] == 3
    assert ordering.make_ordering("anti_heuristic", X, y, list(range(8)), g)[-1] == 3


def test_abs_correlation_and_corr_heuristic():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 6))
    y = (X[:, 4] - 0.5 * X[:, 2] > 0).astype(int)
    X[:, 0] = 3.0  # constant column must not break anything
    c = ordering.abs_correlation(X, y)
    assert c[0] == 0 and np.argmax(c) == 4 and np.argsort(-c)[1] == 2
    perm = ordering.make_ordering("heuristic_corr", X, y, list(range(6)), topology.linear_chain(6))
    assert perm[0] == 4 and perm[1] == 2 and ordering.is_permutation(perm)


def test_scores_dict_matches_array():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 8)); y = (X[:, 5] > 0).astype(int)
    g = topology.linear_chain(8)
    mi = ordering.mutual_information(X, y)
    a = ordering.make_ordering("heuristic", X, y, list(range(8)), g, scores=mi)
    b = ordering.make_ordering("heuristic", X, y, list(range(8)), g,
                               scores={"mi": mi, "corr": ordering.abs_correlation(X, y)})
    assert a.tolist() == b.tolist()


def test_mrmr_avoids_redundant_pair():
    rng = np.random.default_rng(4)
    a = rng.normal(size=600); b = rng.normal(size=600)
    y = (a + b > 0).astype(int)
    X = np.column_stack([a, a + 0.01 * rng.normal(size=600), b, rng.normal(size=600)])  # col 1 duplicates col 0
    order = ordering.rank_features(ordering.mrmr_scores(X, y))
    assert set(order[:2]) != {0, 1}  # the duplicate pair does not take both top slots
    assert 2 in order[:2]


def test_center_ordering_uses_bfs_from_middle():
    scores = np.arange(8)[::-1].astype(float)  # feature 0 best
    X = np.zeros((4, 8)); y = np.zeros(4)
    perm = ordering.heuristic_ordering(X, y, list(range(8)), topology.linear_chain(8), 3, scores=scores)
    assert perm[3] == 0 and set(perm[[2, 4]]) == {1, 2}
