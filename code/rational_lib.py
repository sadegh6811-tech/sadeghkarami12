"""
Rational Feature Library for Mathbot (v2 - bugfixed)
"""
import itertools
import numpy as np


def _safe_ratio(num, den, eps=1e-10, clip=1e6):
    r = num / (den + eps)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(r, -clip, clip)


def build_rational_library(X, base_names=None, eps=1e-10, clip=1e6):
    n_samples, n_features = X.shape
    if base_names is None:
        base_names = [f"x{i}" for i in range(n_features)]

    cols, names = [], []

    # 1) Base
    for i, nm in enumerate(base_names):
        cols.append(X[:, i].astype(float))
        names.append(nm)

    # 2) Pairwise products
    for i in range(n_features):
        for j in range(i, n_features):
            cols.append(X[:, i] * X[:, j])
            names.append(f"{base_names[i]}*{base_names[j]}")

    # 3) Simple ratios (با clip)
    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                continue
            r = _safe_ratio(X[:, i], X[:, j], eps, clip)
            if np.std(r) > 1e-8:
                cols.append(r)
                names.append(f"{base_names[i]}/{base_names[j]}")

    # 4) Differences
    diffs, diff_names = [], []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            d = X[:, i] - X[:, j]
            if np.std(d) > 1e-8:
                diffs.append(d)
                diff_names.append(f"{base_names[i]}-{base_names[j]}")

    # 5) Linear combos with signs in {-1, +1}
    lin_combs, lin_names = [], []
    for signs in itertools.product([-1, 1], repeat=n_features):
        v = sum(s * X[:, k] for k, s in enumerate(signs))
        if np.std(v) < 1e-8:
            continue
        # نام بدون + ابتدای
        parts = []
        for k, s in enumerate(signs):
            if k == 0:
                parts.append(f"{base_names[k]}" if s > 0 else f"-{base_names[k]}")
            else:
                parts.append(f"+{base_names[k]}" if s > 0 else f"-{base_names[k]}")
        label = "".join(parts)
        lin_combs.append(v)
        lin_names.append(label)

    # 6) Rational features: (Σ±x_k) / (x_i - x_j)
    for num, nname in zip(lin_combs, lin_names):
        for den, dname in zip(diffs, diff_names):
            r = _safe_ratio(num, den, eps, clip)
            if np.std(r) > 1e-8:
                cols.append(r)
                names.append(f"({nname})/({dname})")

    A = np.column_stack(cols)
    return A, names
