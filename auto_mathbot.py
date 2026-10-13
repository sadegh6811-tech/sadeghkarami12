"""ربات خودکار: از داده تا معادله + توپولوژی + تحلیل."""
import sys
import numpy as np


def auto_discover(path_or_X, max_degree=3):
    """کشف کامل خودکار."""
    if isinstance(path_or_X, str):
        X = np.genfromtxt(path_or_X, delimiter=",", comments="#")
    else:
        X = np.asarray(path_or_X, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    print("=" * 60)
    print("MathBot — حالت خودکار")
    print("=" * 60)
    print(f"ورودی: {X.shape[0]} نقطه در R^{X.shape[1]}")

    # حذف مختصات ثابت
    stds = X.std(0)
    keep = stds > 1e-10
    X = X[:, keep]
    n = X.shape[1]
    print(f"بعد محیطی مؤثر: {n}")

    if n < 2:
        print("[!] داده کافی نیست.")
        return

    # ۱. حذف نویز
    try:
        from denoise import auto_denoise
        X, denoise_info = auto_denoise(X, sigma_threshold=0.03)
        if denoise_info.get("denoised"):
            print(f"[✓] نویز حذف شد. سیگما: {denoise_info['sigma_rel']:.4f} → {denoise_info.get('sigma_after', 0):.4f}")
        else:
            print("[i] نویز ناچیز بود.")
    except Exception as e:
        print(f"[!] حذف نویز: {e}")

    # ۲. تخمین بعد ذاتی
    from mathbot_core import estimate_dimension_twonn, estimate_dimension_mle
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-12)
    d_twonn = estimate_dimension_twonn(Xn)
    d_mle = estimate_dimension_mle(Xn, k=15)
    print(f"تخمین بعد ذاتی: TwoNN={d_twonn:.2f}, MLE={d_mle:.2f}")

    if abs(d_twonn - d_mle) < 1.5:
        d = int(round((d_twonn + d_mle) / 2))
        reason = "توافق"
    else:
        d = max(1, min(int(d_twonn), int(d_mle)))
        reason = "اختلاف"

    d = max(1, min(d, n))
    print(f"بعد ذاتی نهایی: d = {d}  ({reason})")

    # ۳. تشخیص سادگی: منیفولد تکه‌ای یا یکپارچه؟
    is_piecewise = False
    try:
        from piecewise import cluster_local
        labels, _ = cluster_local(X, n_clusters=3)
        # اگر واریانس درون‌خوشه‌ای خیلی کمتر از کل باشد، تکه‌ای است
        var_total = np.var(X)
        var_within = 0
        for c in range(3):
            Xc = X[labels == c]
            if len(Xc) > 0:
                var_within += len(Xc) * np.var(Xc)
        var_within /= len(X)
        if var_within / var_total < 0.3:
            is_piecewise = True
            print("[i] احتمالاً منیفولد تکه‌ای است.")
    except Exception:
        pass

    # ۴. کشف
    n_eq = n - d
    print(f"\nتعداد معادلات لازم: n - d = {n_eq}")

    if is_piecewise:
        from piecewise import discover_piecewise
        results = discover_piecewise(X, max_degree=max_degree)
        print(f"\n[✓] {len(results)} تکه کشف شد.")
    else:
        from discover import discover_equations, convert_to_original
        eqs, S, mu, sigma = discover_equations(X, n_eq, degree=max_degree)
        print(f"\n[✓] معادلات در مختصات اصلی:")
        for i, e in enumerate(eqs, 1):
            e_orig = convert_to_original(e, mu, sigma, n)
            print(f"    G{i} = {e_orig} = 0")

        # ۵. تحلیل توپولوژی
        try:
            from topology import analyze_topology
            print()
            analyze_topology(Xn)
        except Exception as e:
            print(f"[i] توپولوژی: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python auto_mathbot.py <points.csv> [max_degree]")
        sys.exit(1)
    path = sys.argv[1]
    deg = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    auto_discover(path, max_degree=deg)
