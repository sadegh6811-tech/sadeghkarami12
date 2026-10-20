import numpy as np
import random

# بارگذاری داده
data = np.loadtxt('/data/data/com.termux/files/home/zeta_data_1000.csv', delimiter=',', skiprows=1)
T = data[:, 2]
actual = data[:, 3]

# فرمول ریمان
def riemann(T):
    return 2 * np.pi / np.log(T / (2 * np.pi) + 1e-9)

# نسبت واقعی
ratio = actual / (riemann(T) + 1e-9)
mask_valid = np.isfinite(ratio) & (ratio < 10) & (ratio > 0)
T_clean = T[mask_valid]
y_clean = ratio[mask_valid]

# تابع تصحیح کشف‌شده توسط GP (از خروجی قبلی)
def correction_gp(x):
    x = np.abs(x) + 1e-9
    inner = np.log(np.abs(np.sin(-2.0 / x)) + 1e-9)
    exponent = -2.0 / x
    result = np.sign(inner) * np.abs(inner) ** np.clip(exponent, -3, 3)
    return result

# بررسی مقادیر تابع تصحیح
corr = correction_gp(T_clean)
print("═══ بررسی تابع تصحیح ═══")
print(f"min(correction) = {np.nanmin(corr):.4e}")
print(f"max(correction) = {np.nanmax(corr):.4e}")
print(f"mean(correction) = {np.nanmean(corr):.4e}")
print(f"تعداد NaN/Inf: {np.sum(~np.isfinite(corr))}")
print(f"تعداد مقادیر > 1e6: {np.sum(np.abs(corr) > 1e6)}")
print()

# محاسبه MSE روی داده تمیز
mask_final = np.isfinite(corr)
mse_corr_only = np.mean((y_clean[mask_final] - corr[mask_final])**2)

# فرمول ریمان با تصحیح
pred_corrected = riemann(T_clean) * corr
mask_final2 = np.isfinite(pred_corrected) & np.isfinite(actual[mask_valid])
mse_corrected = np.mean((actual[mask_valid][mask_final2] - pred_corrected[mask_final2])**2)
mse_original = np.mean((actual[mask_valid][mask_final2] - riemann(T_clean)[mask_final2])**2)

print("═══ MSE نهایی ═══")
print(f"MSE ریمان خام:           {mse_original:.6f}")
print(f"MSE ریمان با تصحیح:      {mse_corrected:.6f}")
print(f"بهبود: {100*(mse_original - mse_corrected)/mse_original:.2f}%")
print()

# بررسی نمونه‌ای
print("═══ نمونه‌ای از مقادیر ═══")
print(f"{'T':>10} | {'actual':>10} | {'ratio':>10} | {'corr':>10}")
print("-" * 50)
for i in [0, 100, 300, 500, 700, 900]:
    if i < len(T_clean):
        print(f"{T_clean[i]:>10.2f} | {actual[mask_valid][i]:>10.4f} | {y_clean[i]:>10.4f} | {corr[i]:>10.4f}")
