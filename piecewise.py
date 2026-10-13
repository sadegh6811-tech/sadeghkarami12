"""تشخیص منیفولد تکه‌ای."""
import numpy as np
from mathbot_core import estimate_dimension_twonn


def cluster_local(X, n_clusters=3):
    try:
        from scipy.cluster.vq import kmeans2
        centroids, labels = kmeans2(X, n_clusters, minit="++", seed=42)
        return labels, centroids
    except ImportError:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), n_clusters, replace=False)
        centroids = X[idx].copy()
        for _ in range(20):
            D = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(D, axis=1)
            new_c = np.array([
                X[labels == k].mean(0) if (labels == k).any() else centroids[k]
                for k in range(n_clusters)
            ])
            if np.allclose(new_c, centroids):
                break
            centroids = new_c
        return labels, centroids


def discover_piecewise(X, max_degree=2, min_cluster_size=50):
    n, d = X.shape
    print(f"[*] کشف منیفولد تکه‌ای: {n} نقطه در R^{d}")

    best_labels, best_score, best_n = None, np.inf, 1
    for k in [1, 2, 3, 4]:
        labels, _ = cluster_local(X, n_clusters=k)
        score = 0.0
        ok = True
        for c in range(k):
            Xc = X[labels == c]
            if len(Xc) < min_cluster_size:
                ok = False
                break
            score += np.var(Xc)
        if ok and score < best_score:
            best_score = score
            best_labels = labels
            best_n = k

    print(f"[*] تکه‌های انتخاب‌شده: {best_n}")
    from discover import discover_equations, convert_to_original
    results = []
    for c in range(best_n):
        Xc = X[best_labels == c]
        print(f"\n[*] تکه {c+1}: {len(Xc)} نقطه")
        try:
            Xn = (Xc - Xc.mean(0)) / Xc.std(0)
            d_est = max(1, min(int(round(estimate_dimension_twonn(Xn))), d))
            n_eq = d - d_est
            if n_eq <= 0:
                continue
            eqs, S, mu, sigma = discover_equations(Xc, n_eq, degree=max_degree)
            eqs_orig = [convert_to_original(e, mu, sigma, d) for e in eqs]
            results.append({
                "cluster": c, "n_points": len(Xc), "d_intrinsic": d_est,
                "equations": eqs_orig,
            })
            print(f"    d = {d_est}")
            for i, e in enumerate(eqs_orig, 1):
                print(f"    F{i} = {e} = 0")
        except Exception as e:
            print(f"    خطا: {e}")
    return results
