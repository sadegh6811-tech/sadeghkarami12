"""هسته: تخمین بعد ذاتی + کتابخانه چندجمله‌ای + کتابخانه مثلثاتی."""
import numpy as np
from itertools import combinations_with_replacement


def _knn_distances(X, k):
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(X)
        d, _ = tree.query(X, k=k + 1)
        return d[:, 1:]
    except ImportError:
        n = X.shape[0]
        sq = np.sum(X * X, axis=1, keepdims=True)
        D2 = sq + sq.T - 2.0 * (X @ X.T)
        np.fill_diagonal(D2, np.inf)
        D = np.sqrt(np.maximum(D2, 0.0))
        k = min(k, n - 1)
        idx = np.argpartition(D, k, axis=1)[:, :k]
        rows = np.arange(n)[:, None]
        d = D[rows, idx]
        d.sort(axis=1)
        return d


def estimate_dimension_twonn(X):
    d = _knn_distances(X, k=2)
    r1, r2 = d[:, 0], d[:, 1]
    mask = (r1 > 1e-12) & (r2 > 1e-12)
    r1, r2 = r1[mask], r2[mask]
    if len(r1) < 10:
        return 1.0
    mu = np.sort(r2 / r1)
    N = len(mu)
    F = np.clip(np.arange(1, N + 1) / N, 1e-6, 1 - 1e-6)
    x = np.log(mu)
    y = -np.log(1.0 - F)
    return float(np.sum(x * y) / np.sum(x * x))


def estimate_dimension_mle(X, k=15):
    k = min(k, X.shape[0] - 1)
    d = _knn_distances(X, k=k)
    d = np.maximum(d, 1e-12)
    log_ratios = np.log(d[:, -1:] / d[:, :-1])
    m = np.mean(log_ratios, axis=1)
    m = m[m > 1e-12]
    if len(m) == 0:
        return 1.0
    return float(1.0 / np.mean(m))


def polynomial_library(X, degree):
    n_samples, n_feats = X.shape
    combos = []
    terms = []
    for d in range(degree + 1):
        for combo in combinations_with_replacement(range(n_feats), d):
            combos.append(combo)
            if d == 0:
                terms.append(np.ones(n_samples))
            else:
                prod = np.ones(n_samples)
                for i in combo:
                    prod = prod * X[:, i]
                terms.append(prod)
    return np.column_stack(terms), combos


def trigonometric_library(X, freqs_per_var):
    n, d = X.shape
    cols = []
    specs = []
    for i in range(d):
        xi = X[:, i]
        for omega in freqs_per_var.get(i, []):
            cols.append(np.sin(omega * xi))
            specs.append(("sin", i, float(omega)))
            cols.append(np.cos(omega * xi))
            specs.append(("cos", i, float(omega)))
    if not cols:
        return np.zeros((n, 0)), []
    return np.column_stack(cols), specs
