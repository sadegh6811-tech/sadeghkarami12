import numpy as np
import random

# داده‌های جدول ۴.۳
g = np.array([0.01, 0.05, 0.10, 0.20])
n_max = np.array([0.100, 0.179, 0.288, 0.387])
tau_coh = np.array([11.70, 9.70, 7.20, 4.50])
net_transfer = np.array([0.000, 0.079, 0.188, 0.287])

print("═══ کشف قانون مقیاس‌بندی ═══\n")
print(f"{'g':>6} | {'n_max':>8} | {'tau_coh':>8} | {'Net':>8}")
print("-" * 40)
for i in range(len(g)):
    print(f"{g[i]:>6.2f} | {n_max[i]:>8.3f} | {tau_coh[i]:>8.2f} | {net_transfer[i]:>8.3f}")

# برازش توانی: tau_coh = a * g^(-b)
from scipy.optimize import curve_fit

def power_law(x, a, b):
    return a * x**(-b)

try:
    popt, _ = curve_fit(power_law, g, tau_coh, p0=[0.5, 0.5])
    a, b = popt
    pred = power_law(g, a, b)
    r2 = 1 - np.sum((tau_coh - pred)**2) / np.sum((tau_coh - np.mean(tau_coh))**2)
    print(f"\n[✓] قانون کشف‌شده: tau_coh = {a:.4f} * g^(-{b:.4f})")
    print(f"[✓] R² = {r2:.4f}")
    print(f"[✓] نمای مقیاس‌بندی: b = {b:.4f}")
    
    if 0.3 < b < 1.0:
        print(f"[✓] این با پیش‌بینی نظری (b ≈ 0.5) همخوانی دارد.")
except Exception as e:
    print(f"[!] برازش ناموفق: {e}")

# برازش n_max
try:
    popt, _ = curve_fit(power_law, g, n_max, p0=[0.5, 0.5])
    a, b = popt
    r2 = 1 - np.sum((n_max - power_law(g, a, b))**2) / np.sum((n_max - np.mean(n_max))**2)
    print(f"\n[✓] n_max = {a:.4f} * g^{b:.4f} (R²={r2:.4f})")
except:
    pass
