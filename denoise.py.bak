"""حذف نویز داده‌ی منیفولد با تصویر محلی روی فضای مماس."""
import numpy as np
from mathbot_core import estimate_dimension_twonn, estimate_dimension_mle


def estimate_intrinsic_dim_robust(X):
    """تخمین بعد ذاتی مقاوم: میانه‌ی سه تخمین مختلف."""
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-12)
    d1 = estimate_dimension_twonn(Xn)
    d2 = estimate_dimension_mle(Xn, k=10)
    d3 = estimate_dimension_mle(Xn, k=20)
    ests = sorted([d1, d2, d3])
    d = ests[1]
    return int(round(max(1, min(d, X.shape[1] - 1))))


def knn_indices(X, k):
    """اندیس k نزدیک‌ترین همسایه (بدون خود نقطه)."""
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(X)
        _, idx = tree.query(X, k=k + 1)
        return idx[:, 1:]
    except ImportError:
        n = X.shape[0]
        sq = np.sum(X * X, axis=1, keepdims=True)
        D2 = sq + sq.T - 2.0 * (X @ X.T)
        np.fill_diagonal(D2, np.inf)
        idx = np.argpartition(D2, k, axis=1)[:, :k]
        return idx


def local_pca_denoise(X, k=20, d=None, iterations=2):
    """تصویر هر نقطه روی فضای مماس محلی. چند بار تکرار می‌شود."""
    if d is None:
        d = estimate_intrinsic_dim_robust(X)
    n, ndim = X.shape
    if d >= ndim:
        return X.copy()
    k = min(k, n - 1)
    X_clean = X.copy()

    for _ in range(iterations):
        idx = knn_indices(X_clean, k)
        new_X = np.empty_like(X_clean)
        for i in range(n):
            neigh = X_clean[idx[i]]
            center = neigh.mean(axis=0)
            centered = neigh - center
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            V_d = Vt[:d]
            proj = V_d.T @ (V_d @ (X_clean[i] - center))
            new_X[i] = center + proj
        X_clean = new_X

    return X_clean


def estimate_noise_level(X, k=20, d=None):
    """تخمین سیگما نویز از باقی‌مانده‌ی PCA محلی."""
    if d is None:
        d = estimate_intrinsic_dim_robust(X)
    n, ndim = X.shape
    if d >= ndim:
        return 0.0
    k = min(k, n - 1)
    idx = knn_indices(X, k)
    residuals = []

    for i in range(n):
        neigh = X[idx[i]]
        center = neigh.mean(axis=0)
        centered = neigh - center
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        if len(S) > d:
            residuals.extend(S[d:] ** 2 / max(k - 1, 1))

    if not residuals:
        return 0.0
    return float(np.sqrt(np.mean(residuals)))


def auto_denoise(X, sigma_threshold=0.03, k=20, iterations=2):
    """
    اگر نویز نسبی از آستانه بیشتر بود، حذف نویز انجام می‌شود.
    خروجی: (X_clean, info)
    """
    sigma = estimate_noise_level(X, k=k)
    scale = float(X.std())
    sigma_rel = sigma / (scale + 1e-12)
    info = {
        "sigma": sigma,
        "sigma_rel": sigma_rel,
        "denoised": False,
        "d": None,
    }
    if sigma_rel > sigma_threshold:
        d = estimate_intrinsic_dim_robust(X)
        X_clean = local_pca_denoise(X, k=k, d=d, iterations=iterations)
        # سیگما پس از حذف نویز
        sigma_after = estimate_noise_level(X_clean, k=k, d=d)
        info["denoised"] = True
        info["d"] = d
        info["sigma_after"] = sigma_after
        return X_clean, info
    return X.copy(), info
