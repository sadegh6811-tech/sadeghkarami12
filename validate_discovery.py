import mpmath as mp
import numpy as np
import random
import sympy as sp

# ۱. تولید ۱۰۰۰ صفر جدید
print("[*] در حال محاسبه ۱۰۰۰ صفر تابع زتا (برای اعتبارسنجی)...")
mp.mp.dps = 20
N = 1000
zeros = []
for i in range(1, N + 1):
    z = mp.zetazero(i)
    zeros.append((i, float(mp.re(z)), float(mp.im(z))))
    if i % 200 == 0:
        print(f"    {i} صفر...")

data = np.array(zeros)
gaps = np.diff(data[:, 2], prepend=data[0, 2])

# ۲. فرمولی که GP کشف کرد
def gp_formula(T):
    T = np.abs(T) + 1e-9
    try:
        numerator = 2.0 ** np.log(T)
        inner = np.log(np.abs(T ** T) + 1e-9)
        denom = np.abs(np.log(np.abs(0.5 * T - 1) + inner + 1e-9))
        return numerator / denom
    except:
        return np.full_like(T, np.nan)

# ۳. فرمول ریمان-فون مانگولت
def riemann_formula(T):
    return 2 * np.pi / np.log(T / (2 * np.pi) + 1e-9)

# ۴. مقایسه
T = data[:, 2]
actual_gap = gaps

pred_gp = gp_formula(T)
pred_riemann = riemann_formula(T)

# حذف مقادیر NaN و inf
mask = np.isfinite(pred_gp) & np.isfinite(pred_riemann) & np.isfinite(actual_gap)
pred_gp = pred_gp[mask]
pred_riemann = pred_riemann[mask]
actual_gap = actual_gap[mask]

mse_gp = np.mean((actual_gap - pred_gp) ** 2)
mse_riemann = np.mean((actual_gap - pred_riemann) ** 2)

print(f"\n═══ نتایج اعتبارسنجی روی {np.sum(mask)} صفر ═══")
print(f"MSE فرمول GP:           {mse_gp:.6f}")
print(f"MSE فرمول ریمان-فون:  {mse_riemann:.6f}")
if mse_gp < mse_riemann:
    print(f"✅ GP بهتر است (بهبود {100*(mse_riemann - mse_gp)/mse_riemann:.1f}%)")
else:
    print(f"⚠️ ریمان-فون بهتر است.")

# ۵. نمودار مقایسه
print("\n═══ مقایسه در ۵۰ صفر اول ═══")
print(f"{'T':>8} | {'Actual':>8} | {'GP':>8} | {'Riemann':>8}")
for i in range(min(20, len(actual_gap))):
    print(f"{T[i]:8.2f} | {actual_gap[i]:8.4f} | {pred_gp[i]:8.4f} | {pred_riemann[i]:8.4f}")
