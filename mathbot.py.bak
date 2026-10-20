"""ربات کشف فرمول. استفاده: python mathbot.py <points.csv> [max_degree] [--save]"""
import sys
import os
import numpy as np
import sympy as sp

from mathbot_core import estimate_dimension_twonn, estimate_dimension_mle
from discover import discover_equations, convert_to_original


def load_points(path):
    if path.endswith(".npy"):
        return np.load(path)
    data = np.genfromtxt(path, delimiter=",", comments="#")
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data


def choose_dimension(d_twonn, d_mle, n):
    if abs(d_twonn - d_mle) < 1.5:
        d = (d_twonn + d_mle) / 2.0
        reason = "توافق"
    else:
        d = min(d_twonn, d_mle)
        reason = "اختلاف زیاد → کوچک‌تر"
    d = int(round(max(1, min(d, n))))
    return d, reason


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python mathbot.py <points.csv> [max_degree] [--save]")
        sys.exit(1)

    save_flag = "--save" in args
    args = [a for a in args if a != "--save"]

    path = args[0]
    max_degree = int(args[1]) if len(args) > 1 else 3

    X = load_points(path)
    print(f"[*] بارگذاری: {X.shape[0]} نقطه در R^{X.shape[1]}")

    stds = X.std(axis=0)
    keep = stds > 1e-10
    if not np.all(keep):
        print(f"[*] {np.sum(~keep)} مختصات ثابت حذف شد.")
    X = X[:, keep]
    n = X.shape[1]
    print(f"[*] بعد محیطی مؤثر: n = {n}")
    if n == 0:
        print("[!] تمام مختصات ثابت‌اند.")
        return

    # حذف نویز
    print("[*] بررسی سطح نویز...")
    from denoise import auto_denoise
    X, denoise_info = auto_denoise(X, sigma_threshold=0.03)
    print(f"    سیگما نسبی قبل: {denoise_info['sigma_rel']:.4f}")
    if denoise_info["denoised"]:
        print(f"    حذف نویز انجام شد (d = {denoise_info['d']})")
        sa = denoise_info.get("sigma_after", 0)
        print(f"    سیگما پس از حذف نویز: {sa:.4f}")
    else:
        print("    نویز ناچیز — حذف نویز لازم نبود.")

    # بعد ذاتی
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-12)
    d_twonn = estimate_dimension_twonn(Xn)
    d_mle = estimate_dimension_mle(Xn, k=15)
    print(f"[*] تخمین بعد ذاتی: TwoNN = {d_twonn:.3f} | MLE = {d_mle:.3f}")

    d, reason = choose_dimension(d_twonn, d_mle, n)
    print(f"[*] بعد ذاتی انتخاب‌شده: d = {d}  ({reason})")

    n_eq = n - d
    if n_eq <= 0:
        print("[✓] نقاط کل فضا را پر می‌کنند.")
        return
    print(f"[*] تعداد معادلات لازم: n - d = {n_eq}")

    eqs_norm, S, mu, sigma = discover_equations(X, n_eq, degree=max_degree)

    print("\n[*] مقادیر تکین (۱۰ تای اول):")
    if len(S) > 0:
        print("    ", np.round(S[:10], 5))

    # تبدیل به مختصات اصلی
    eqs_orig = []
    print("\n[✓] معادلات در مختصات اصلی:")
    for i, eq in enumerate(eqs_norm, 1):
        eq_orig = convert_to_original(eq, mu, sigma, n)
        eqs_orig.append(eq_orig)
        print(f"    G{i} = {eq_orig} = 0")

    print("\n[i] میانگین نقاط:", np.round(mu, 4))
    print("[i] انحراف معیار  :", np.round(sigma, 4))

    # ذخیره
    if save_flag:
        print("\n[*] ذخیره‌ی کشفیات...")
        from saver import save_equations
        paths = save_equations(
            input_file=path, n=n, d=d, n_eq=n_eq,
            equations_orig=eqs_orig,
            equations_normalized=eqs_norm,
            mu=mu, sigma=sigma, singular_values=S,
            denoise_info=denoise_info, max_degree=max_degree,
        )
        print(f"    TXT:  {paths['txt']}")
        print(f"    TEX:  {paths['tex']}")
        print(f"    JSON: {paths['json']}")
        print(f"    LOG:  {paths['log']}")


if __name__ == "__main__":
    main()
