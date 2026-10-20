import numpy as np

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

print(f"[*] داده تمیز: {len(T_clean)} نقطه")
print(f"[*] میانگین ratio: {np.mean(y_clean):.6f}")
print(f"[*] انحراف معیار ratio: {np.std(y_clean):.6f}")

# ═══════════════════════════════════════════════════════════
# تفسیر درست درخت GP
# درخت: ('pow', ('abs', ('log', ('const', -2.0))), ('sin', ('div', ('const', -2.0), 'x0')))
# یعنی: (|log(-2)|)^(sin(-2/x))
# ═══════════════════════════════════════════════════════════
def correction_correct(x):
    x = np.abs(x) + 1e-9
    try:
        # log(-2) یک عدد مختلط است. |log(-2)| = |log(2) + i*pi| = sqrt(log(2)^2 + pi^2)
        # برای سادگی، از log(abs(-2)) استفاده می‌کنیم (قدرمطلق در کد GP)
        base = np.abs(np.log(np.abs(-2.0) + 1e-9))  # = |log(2)| ≈ 0.693
        exponent = np.sin(-2.0 / x)
        # کل تابع: (|log(-2)|)^(sin(-2/x))
        result = base ** exponent
        return result
    except:
        return np.full_like(x, np.nan)

# محاسبه مقادیر
corr = correction_correct(T_clean)

print(f"\n═══ بررسی تابع تصحیح (تفسیر درست) ═══")
print(f"min(correction): {np.nanmin(corr):.6e}")
print(f"max(correction): {np.nanmax(corr):.6e}")
print(f"mean(correction): {np.nanmean(corr):.6f}")
print(f"std(correction): {np.nanstd(corr):.6f}")
print(f"تعداد NaN/Inf: {np.sum(~np.isfinite(corr))}")

# بررسی نمونه‌ای
print(f"\n═══ نمونه‌ای از مقادیر ═══")
print(f"{'T':>10} | {'ratio':>10} | {'corr':>10}")
print("-" * 35)
for i in [0, 100, 300, 500, 700, 900]:
    if i < len(T_clean):
        print(f"{T_clean[i]:>10.2f} | {y_clean[i]:>10.6f} | {corr[i]:>10.6f}")

# محاسبه MSE
mask_final = np.isfinite(corr)
mse_corr_only = np.mean((y_clean[mask_final] - corr[mask_final])**2)
print(f"\nMSE تابع تصحیح (فقط روی ratio): {mse_corr_only:.6f}")

# اعمال روی فرمول ریمان
pred_corrected = riemann(T_clean) * corr
mask_final2 = np.isfinite(pred_corrected)
mse_with_corr = np.mean((actual[mask_valid][mask_final2] - pred_corrected[mask_final2])**2)
mse_without = np.mean((actual[mask_valid][mask_final2] - riemann(T_clean)[mask_final2])**2)

print(f"\n═══ مقایسه نهایی ═══")
print(f"MSE ریمان خام:      {mse_without:.6f}")
print(f"MSE ریمان با تصحیح: {mse_with_corr:.6f}")
print(f"بهبود: {100*(mse_without - mse_with_corr)/mse_without:.2f}%")
