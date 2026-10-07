"""تشخیص فرکانس: جفت دایره‌ای + اسکن شبکه‌ای ω (روش پایدار)."""
import numpy as np


def detect_circle_pairs(X, rtol=0.15):
    n, d = X.shape
    pairs = {}
    used = set()
    for i in range(d):
        for j in range(i + 1, d):
            if i in used or j in used:
                continue
            r2 = X[:, i] ** 2 + X[:, j] ** 2
            mean_r2 = r2.mean()
            if mean_r2 < 1e-6:
                continue
            if r2.std() / mean_r2 < rtol:
                pairs[(i, j)] = float(mean_r2)
                used.add(i)
                used.add(j)
    return pairs


def detect_dominant_frequencies(x, y, n_freqs=2, n_omega=800,
                                 min_omega=0.3, max_omega=30.0,
                                 r2_threshold=0.7):
    """
    اسکن شبکه‌ای ω. برای هر ω، مدل y ≈ a·sin(ωx) + b·cos(ωx) + c را برازش کن.
    قله‌های R² بالای آستانه را برگردان.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 40:
        return []
    if y.std() < 1e-12:
        return []
    yn = (y - y.mean()) / y.std()

    omegas = np.linspace(min_omega, max_omega, n_omega)
    r2_scores = np.zeros(n_omega)
    ones = np.ones(n)

    for k, omega in enumerate(omegas):
        s = np.sin(omega * x)
        c = np.cos(omega * x)
        A = np.column_stack([s, c, ones])
        try:
            coef, _, rank, _ = np.linalg.lstsq(A, yn, rcond=None)
        except Exception:
            continue
        if rank < 3:
            continue
        pred = A @ coef
        ss_res = np.sum((yn - pred) ** 2)
        ss_tot = np.sum(yn ** 2)
        if ss_tot > 1e-12:
            r2_scores[k] = 1.0 - ss_res / ss_tot

    peaks = []
    work = r2_scores.copy()
    for _ in range(n_freqs):
        k = int(np.argmax(work))
        if work[k] < r2_threshold:
            break
        omega_best = omegas[k]
        # بازآرایی سهمی برای دقت بیشتر
        if 0 < k < n_omega - 1:
            y0, y1, y2 = work[k - 1], work[k], work[k + 1]
            denom = y0 - 2 * y1 + y2
            delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
            omega_best = omegas[k] + delta * (omegas[1] - omegas[0])
        if omega_best > 0:
            if not any(abs(omega_best - p) / max(p, 1e-9) < 0.15 for p in peaks):
                peaks.append(float(omega_best))
        lo = max(0, k - 5)
        hi = min(len(work), k + 6)
        work[lo:hi] = 0.0

    return peaks


def detect_all_frequencies(X, top_n=2):
    n, d = X.shape
    all_freqs = {j: [] for j in range(d)}
    circles = detect_circle_pairs(X)
    circle_vars = set()
    for (i, j) in circles:
        circle_vars.add(i)
        circle_vars.add(j)
    for v in circle_vars:
        all_freqs[v] = []

    for i in range(d):
        if i in circle_vars:
            continue
        for j in range(d):
            if i == j:
                continue
            fs = detect_dominant_frequencies(X[:, i], X[:, j], n_freqs=top_n)
            for f in fs:
                if not any(abs(f - g) / max(g, 1e-9) < 0.15 for g in all_freqs[i]):
                    all_freqs[i].append(f)

    for j in all_freqs:
        all_freqs[j].sort()
    return all_freqs, circles
