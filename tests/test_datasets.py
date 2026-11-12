import numpy as np

from qorder import datasets


def test_adversarial_matches_paper_shape():
    X, y, Xt, yt = datasets.adversarial()
    assert X.shape == (300, 20) and Xt.shape == (300, 20)
    assert set(np.unique(y)) == {0, 1}
    # noise columns carry no label information, real ones do
    c_real = np.abs([np.corrcoef(X[:, j], y)[0, 1] for j in range(10)]).max()
    c_noise = np.abs([np.corrcoef(X[:, j], y)[0, 1] for j in range(10, 20)]).max()
    assert c_real > c_noise


def test_adversarial_is_reproducible():
    a = datasets.adversarial()
    b = datasets.adversarial()
    assert all(np.array_equal(u, v) for u, v in zip(a, b))


def test_bundled_real_datasets():
    for X, y, Xt, yt in (datasets.breast_cancer(), datasets.digits_pair()):
        assert X.shape[1] == 8 and Xt.shape[1] == 8
        assert set(np.unique(y)) <= {0, 1}


def test_to_binary_keeps_two_largest_classes():
    y = np.array(["a"] * 5 + ["b"] * 3 + ["c"] * 7)
    yb, mask = datasets._to_binary(y)
    assert mask.sum() == 12 and set(np.unique(yb)) == {0, 1}


def test_wine_loader():
    X, y, Xt, yt = datasets.wine()
    assert X.shape[1] == 8 and set(np.unique(y)) == {0, 1}


def test_openml_loader_with_fake_fetch():
    import sklearn.datasets as skd
    import pandas as pd

    class Fake:
        def __init__(self):
            rng = np.random.default_rng(0)
            self.data = pd.DataFrame(rng.normal(size=(1500, 10)), columns=[f"f{i}" for i in range(10)])
            self.data["cat"] = pd.Categorical(["x"] * 1500)
            self.target = pd.Series(np.where(self.data.f3 > 0, "pos", "neg"))

    orig = skd.fetch_openml
    skd.fetch_openml = lambda **kw: Fake()
    try:
        X, y, Xt, yt = datasets.openml("fake")
    finally:
        skd.fetch_openml = orig
    assert X.shape == (700, 8) and Xt.shape == (300, 8)  # subsampled to 1000, 70/30
