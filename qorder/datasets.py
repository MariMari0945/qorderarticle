"""Datasets used in the experiments.

Every loader returns ``(X_train, y_train, X_test, y_test)`` with labels in
{0, 1}. Anything fitted on data (PCA, scalers, mutual information) is fitted
on the training split only.

The hidden-manifold generator and the noise-feature routine reproduce the
adversarial permutation dataset of Salmenperä, Perkkola & Nurminen (2025),
whose code (https://github.com/Ilmosal/QMLPerm) builds on Xanadu's
qml-benchmarks. Both are Apache-2.0; see NOTICE.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle as sk_shuffle


# --------------------------------------------------------------------------- #
# Hidden manifold + adversarial noise (Salmenperä et al. 2025, Exp. "adv perm")
# --------------------------------------------------------------------------- #
def _hidden_manifold(n_samples, n_features, manifold_dimension):
    """Port of qml-benchmarks' generate_hidden_manifold_model (uses global np.random)."""
    F = np.random.normal(size=(manifold_dimension, n_features))
    C = np.random.normal(size=(n_samples, manifold_dimension), loc=0, scale=1)
    biases = 2 * np.random.uniform(size=(n_features,)) - 1
    X = np.tanh(C @ F / np.sqrt(manifold_dimension) - biases)
    W = np.random.normal(size=(manifold_dimension, manifold_dimension))
    v = np.random.normal(size=(manifold_dimension,))
    y = np.array([np.dot(v, np.tanh(W @ c) / np.sqrt(W.shape[0])) for c in C])
    y = y - np.median(y)
    y = np.where(y < 0, -1, 1)
    return X, y


def hidden_manifold(n_train=300, n_test=300, dimension=10, manifold_dimension=6, seed=1234):
    """Hidden-manifold dataset, generated exactly as in QMLPerm's get_dataset."""
    np.random.seed(seed)
    X, y = _hidden_manifold(n_train + n_test, dimension, manifold_dimension)
    X, y = sk_shuffle(X, y)
    y = (y > 0).astype(int)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def add_noise_features(X_train, X_test, n_new=None, seed=1):
    """Append N(0, 1) features that carry no label information (QMLPerm noise_data.py).

    The original features stay in columns [0, d); the noise goes to [d, d + n_new).
    """
    np.random.seed(seed)
    d = X_train.shape[1]
    n_new = d if n_new is None else n_new
    Xn = np.random.normal(size=(len(X_train), d + n_new))
    Xtn = np.random.normal(size=(len(X_test), d + n_new))
    Xn[:, :d] = X_train
    Xtn[:, :d] = X_test
    return Xn, Xtn


def adversarial(n_train=300, n_test=300, dimension=10, manifold_dimension=6, n_noise=None,
                seed=1234, noise_seed=1):
    """Adversarial permutation dataset: `dimension` real features followed by noise features.

    Defaults reproduce the paper (10 real + 10 noise = 20 features, 300/300 split).
    Use dimension=4, n_noise=4 for an 8-feature version that fits the 8-qubit circuit.
    """
    X, y, Xt, yt = hidden_manifold(n_train, n_test, dimension, manifold_dimension, seed)
    X, Xt = add_noise_features(X, Xt, n_noise, noise_seed)
    return X, y, Xt, yt


# --------------------------------------------------------------------------- #
# Real datasets
# --------------------------------------------------------------------------- #
def _load_mnist_like(fashion: bool):
    """Return (X, y) with 70k flattened 28x28 images. Tries Keras, then OpenML."""
    try:  # Colab ships TensorFlow/Keras
        from tensorflow import keras  # type: ignore
        ds = keras.datasets.fashion_mnist if fashion else keras.datasets.mnist
        (a, b), (c, d) = ds.load_data()
        X = np.concatenate([a, c]).reshape(-1, 784).astype(float)
        y = np.concatenate([b, d]).astype(int)
        return X, y
    except Exception:
        from sklearn.datasets import fetch_openml
        name = "Fashion-MNIST" if fashion else "mnist_784"
        X, y = fetch_openml(name, version=1, return_X_y=True, as_frame=False, parser="auto")
        return X.astype(float), y.astype(int)


def _reduce(X_train, X_test, n_components, method="pca", seed=0):
    # note: for 'random_subset', seed picks the columns (subset_seed in the loaders)
    """Standardise, then reduce to n_components (fitted on train only)."""
    sc = StandardScaler().fit(X_train)
    X_train, X_test = sc.transform(X_train), sc.transform(X_test)
    if method == "pca":
        pca = PCA(n_components=n_components, random_state=seed).fit(X_train)
        return pca.transform(X_train), pca.transform(X_test)
    if method == "first":
        return X_train[:, :n_components], X_test[:, :n_components]
    if method == "random_subset":
        cols = np.sort(np.random.default_rng(seed).choice(X_train.shape[1], n_components, replace=False))
        return X_train[:, cols], X_test[:, cols]
    raise ValueError(method)


def mnist_pair(classes=(0, 1), fashion=False, n_components=8, n_train=1000, n_test=1000,
               seed=0, shuffle_components=False, perm_seed=0):
    """Binary MNIST / Fashion-MNIST, PCA to n_components.

    PCA components come sorted by explained variance, so the *default*
    ordering is already a sensible ordering. Set shuffle_components=True to
    store the components in a fixed random order, so that 'default' is an
    arbitrary order as in raw tabular data (report which one you used).
    `seed` controls the train/test split and the PCA; `perm_seed` controls the
    component permutation, so re-splitting keeps the same arbitrary order.
    """
    X, y = _load_mnist_like(fashion)
    mask = np.isin(y, classes)
    X, y = X[mask], (y[mask] == classes[1]).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=n_train, test_size=n_test,
                                              stratify=y, random_state=seed)
    X_tr, X_te = _reduce(X_tr, X_te, n_components, "pca", seed)
    if shuffle_components:
        p = np.random.default_rng(perm_seed).permutation(n_components)
        X_tr, X_te = X_tr[:, p], X_te[:, p]
    return X_tr, y_tr, X_te, y_te


def breast_cancer(n_components=8, method="first", test_size=0.3, seed=0):
    """Breast Cancer Wisconsin (bundled with scikit-learn, 30 features).

    method='first' keeps the first 8 columns in the dataset's own order (an
    arbitrary 'default'); 'random_subset' picks 8 random columns; 'pca' reduces.
    """
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    X_tr, X_te = _reduce(X_tr, X_te, n_components, method, seed)
    return X_tr, y_tr, X_te, y_te


def digits_pair(classes=(0, 1), n_components=8, test_size=0.3, seed=0):
    """8x8 digits bundled with scikit-learn: an offline stand-in for MNIST (pilot/tests)."""
    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    mask = np.isin(y, classes)
    X, y = X[mask].astype(float), (y[mask] == classes[1]).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    X_tr, X_te = _reduce(X_tr, X_te, n_components, "pca", seed)
    return X_tr, y_tr, X_te, y_te


def _to_binary(y):
    """Labels -> {0, 1}. With more than two classes keep the two most frequent (mask returned)."""
    y = np.asarray(y).astype(str)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("dataset has a single class")
    top = classes[np.argsort(-counts, kind="stable")[:2]]
    top = np.sort(top)
    mask = np.isin(y, top)
    return (y[mask] == top[1]).astype(int), mask


def _tabular_split(X, y, n_features=8, method="first", test_size=0.3, seed=0,
                   max_samples=1000, subset_seed=0):
    """Common path for tabular data: binarise labels, subsample, split, reduce to n_features."""
    X = np.asarray(X, float)
    y, mask = _to_binary(y)
    X = X[mask]
    keep = np.all(np.isfinite(X), axis=1)
    X, y = X[keep], y[keep]
    if max_samples and len(X) > max_samples:
        X, _, y, _ = train_test_split(X, y, train_size=max_samples, stratify=y, random_state=seed)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    X_tr, X_te = _reduce(X_tr, X_te, n_features, method, subset_seed)
    return X_tr, y_tr, X_te, y_te


def openml(dataset, version=1, n_features=8, method="first", test_size=0.3, seed=0,
           max_samples=1000, subset_seed=0):
    """Any OpenML classification dataset (numeric columns only), in its original column order.

    method='first' keeps the first n_features numeric columns as stored by the
    authors (an arbitrary 'default' order); 'random_subset' draws them with
    subset_seed. Needs internet (works in Colab).
    """
    from sklearn.datasets import fetch_openml
    ds = fetch_openml(name=dataset, version=version, as_frame=True, parser="auto")
    Xdf = ds.data.select_dtypes(include="number")
    if Xdf.shape[1] < n_features:
        raise ValueError(f"{dataset}: only {Xdf.shape[1]} numeric features")
    return _tabular_split(Xdf.to_numpy(float), ds.target.to_numpy(), n_features, method,
                          test_size, seed, max_samples, subset_seed)


def wine(n_features=8, method="first", test_size=0.3, seed=0, subset_seed=0):
    """Wine (bundled with scikit-learn, 13 features): the two largest classes."""
    from sklearn.datasets import load_wine
    X, y = load_wine(return_X_y=True)
    return _tabular_split(X, y, n_features, method, test_size, seed, None, subset_seed)


def load(name: str, **kw):
    """Dispatch by name: adversarial, hidden_manifold, mnist, fashion, breast_cancer, digits."""
    if name == "adversarial":
        return adversarial(**kw)
    if name == "hidden_manifold":
        return hidden_manifold(**kw)
    if name == "mnist":
        return mnist_pair(fashion=False, **kw)
    if name == "fashion":
        return mnist_pair(fashion=True, **kw)
    if name == "breast_cancer":
        return breast_cancer(**kw)
    if name == "digits":
        return digits_pair(**kw)
    if name == "wine":
        return wine(**kw)
    if name == "openml":
        return openml(**kw)
    raise ValueError(f"Unknown dataset '{name}'")


# --------------------------------------------------------------------------- #
# Angle scaling
# --------------------------------------------------------------------------- #
def angle_scaler(X_train, low=-np.pi / 2, high=np.pi / 2, clip=False):
    """MinMax scaler to rotation angles, fitted on train only (same range and no clipping, as QMLPerm)."""
    return MinMaxScaler(feature_range=(low, high), clip=clip).fit(X_train)
