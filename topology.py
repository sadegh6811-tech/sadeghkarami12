import numpy as np


def _pairwise_distances(X):
    sq = np.sum(X * X, axis=1, keepdims=True)
    D2 = sq + sq.T - 2.0 * (X @ X.T)
    np.fill_diagonal(D2, 0)
    return np.sqrt(np.maximum(D2, 0))


def _count_components(A):
    n = len(A)
    visited = np.zeros(n, dtype=bool)
    n_comp = 0
    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        while stack:
            v = stack.pop()
            if visited[v]:
                continue
            visited[v] = True
            for u in np.where(A[v] == 1)[0]:
                if not visited[u]:
                    stack.append(u)
        n_comp += 1
    return n_comp


def _count_triangles(A):
    n = len(A)
    F = 0
    for i in range(n):
        for j in range(i + 1, n):
            if not A[i, j]:
                continue
            for k in range(j + 1, n):
                if A[i, k] and A[j, k]:
                    F += 1
    return F


def _auto_radius(D):
    mask = D > 0
    dmax = float(np.max(np.where(mask, D, 0)))
    nearest = np.min(np.where(mask, D, np.inf), axis=1)
    r_low = float(np.median(nearest))
    r_high = dmax
    for _ in range(30):
        r_mid = 0.5 * (r_low + r_high)
        A = (D <= r_mid).astype(int)
        np.fill_diagonal(A, 0)
        if _count_components(A) <= 1:
            r_high = r_mid
        else:
            r_low = r_mid
        if (r_high - r_low) / (r_high + 1e-12) < 0.02:
            break
    return r_high


def analyze_topology(X, d_intrinsic=None, max_points=150):
    n = len(X)
    if n > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_points, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X
    V = len(X_sub)
    D = _pairwise_distances(X_sub)
    r = _auto_radius(D)
    A = (D <= r).astype(int)
    np.fill_diagonal(A, 0)
    b0 = _count_components(A)
    E = int(A.sum() // 2)
    F = _count_triangles(A)
    chi = V - E + F
    if d_intrinsic is None:
        from mathbot_core import estimate_dimension_twonn
        Xn = (X - X.mean(0)) / (X.std(0) + 1e-12)
        d_intrinsic = max(1, int(round(estimate_dimension_twonn(Xn))))
    b2 = 1 if (d_intrinsic == 2 and b0 == 1) else 0
    b1 = b0 + b2 - chi
    if b1 < 0:
        b1 = 0
    genus = 0
    if d_intrinsic == 2 and b2 == 1:
        genus = b1 // 2
    print("[*] تحلیل توپولوژی (زیرنمونه: %d نقطه، شعاع r=%.4f):" % (V, r))
    print("    V=%d  E=%d  F=%d" % (V, E, F))
    print("    Euler characteristic chi = %d" % chi)
    print("    Betti-0: %d" % b0)
    print("    Betti-1: %d" % b1)
    print("    Betti-2: %d" % b2)
    if d_intrinsic == 2:
        print("    Genus: %d" % genus)
    return {"V": V, "E": E, "F": F, "chi": chi,
            "betti_0": b0, "betti_1": b1, "betti_2": b2,
            "genus": genus, "radius": r, "d_intrinsic": d_intrinsic}
