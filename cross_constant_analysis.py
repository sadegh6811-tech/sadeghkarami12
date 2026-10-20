import numpy as np

# ═══════════════════════════════════════════════════════════
# ۱. بارگذاری داده‌ها (۳۰۰ و ۱۰۰۰ صفر)
# ═══════════════════════════════════════════════════════════
data_300 = np.loadtxt('/data/data/com.termux/files/home/zeta_data.csv', delimiter=',', skiprows=1)
data_1000 = np.loadtxt('/data/data/com.termux/files/home/zeta_data_1000.csv', delimiter=',', skiprows=1)

# ═══════════════════════════════════════════════════════════
# ۲. فرمول GP (که در ۳۰۰ نقطه کشف شد)
# ═══════════════════════════════════════════════════════════
def gp_formula(T):
    T = np.abs(T) + 1e-9
    try:
        numerator = 2.0 ** np.log(T)
        inner = np.log(np.abs(T ** T) + 1e-9)
        denom = np.abs(np.log(np.abs(0.5 * T - 1) + inner + 1e-9))
        return numerator / denom
    except:
        return np.full_like(T, np.nan)

# ═══════════════════════════════════════════════════════════
# ۳. فرمول ریمان-فون مانگولت
# ═══════════════════════════════════════════════════════════
def riemann_formula(T):
    return 2 * np.pi / np.log(T / (2 * np.pi) + 1e-9)

# ═══════════════════════════════════════════════════════════
# ۴. ثابت‌های کشف‌شده
# ═══════════════════════════════════════════════════════════
C_GP = 0.6625    # ثابت بهینه برای فرمول GP
C_RIEMANN = 0.9775  # ثابت بهینه برای فرمول ریمان

# ═══════════════════════════════════════════════════════════
# ۵. تحلیل روی ۱۰۰۰ صفر
# ═══════════════════════════════════════════════════════════
T = data_1000[:, 2]
actual = data_1000[:, 3]

# حذف مقادیر نامعتبر
mask = np.isfinite(gp_formula(T)) & np.isfinite(riemann_formula(T))
T = T[mask]
actual = actual[mask]

gp_base = gp_formula(T)
riemann_base = riemann_formula(T)

print("═══ تحلیل ثابت‌های متقابل (Cross-Constant Analysis) ═══\n")

# ─── ۵.۱: فرمول GP با ثابت خودش (C=0.66) ───
pred_gp_own = C_GP * gp_base
mse_gp_own = np.mean((actual - pred_gp_own)**2)

# ─── ۵.۲: فرمول GP با ثابت ریمان (C=0.97) ───
pred_gp_riemann = C_RIEMANN * gp_base
mse_gp_riemann = np.mean((actual - pred_gp_riemann)**2)

# ─── ۵.۳: فرمول ریمان با ثابت خودش (C=0.97) ───
pred_riemann_own = C_RIEMANN * riemann_base
mse_riemann_own = np.mean((actual - pred_riemann_own)**2)

# ─── ۵.۴: فرمول ریمان با ثابت GP (C=0.66) ───
pred_riemann_gp = C_GP * riemann_base
mse_riemann_gp = np.mean((actual - pred_riemann_gp)**2)

# ─── ۵.۵: ترکیب خطی (نصف GP + نصف ریمان) ───
pred_hybrid = 0.5 * pred_gp_own + 0.5 * pred_riemann_own
mse_hybrid = np.mean((actual - pred_hybrid)**2)

# ═══════════════════════════════════════════════════════════
# ۶. نمایش نتایج
# ═══════════════════════════════════════════════════════════
print(f"{'روش':<40} | {'MSE':<10}")
print("-" * 55)
print(f"{'فرمول GP با ثابت خودش (0.66)':<40} | {mse_gp_own:.6f}")
print(f"{'فرمول GP با ثابت ریمان (0.97)':<40} | {mse_gp_riemann:.6f}")
print(f"{'فرمول ریمان با ثابت خودش (0.97)':<40} | {mse_riemann_own:.6f}")
print(f"{'فرمول ریمان با ثابت GP (0.66)':<40} | {mse_riemann_gp:.6f}")
print(f"{'ترکیب خطی (نصف-نصف)':<40} | {mse_hybrid:.6f}")

# ═══════════════════════════════════════════════════════════
# ۷. تحلیل نهایی
# ═══════════════════════════════════════════════════════════
print("\n═══ نتیجه‌گیری ═══")
best_method = min(
    [("GP-own", mse_gp_own), ("GP-Riemann", mse_gp_riemann),
     ("Riemann-own", mse_riemann_own), ("Riemann-GP", mse_riemann_gp),
     ("Hybrid", mse_hybrid)],
    key=lambda x: x[1]
)
print(f"بهترین روش: {best_method[0]} با MSE = {best_method[1]:.6f}")

if mse_hybrid < mse_riemann_own:
    print("\n✅ ترکیب خطی GP و ریمان، بهتر از هر کدام به تنهایی است!")
    print("   این یعنی دو فرمول، اطلاعات مکملی دارند.")
else:
    print("\n⚠️ ترکیب خطی بهتر از ریمان به تنهایی نیست.")
    print("   این یعنی ریمان، اطلاعات کافی را دارد و GP اطلاعات جدیدی ندارد.")

# ═══════════════════════════════════════════════════════════
# ۸. مقایسه در ۱۰ صفر اول
# ═══════════════════════════════════════════════════════════
print("\n═══ مقایسه در ۱۰ صفر اول (۱۰۰۰ نقطه) ═══")
print(f"{'T':>8} | {'Actual':>8} | {'GP+C_GP':>10} | {'GP+C_Rie':>10} | {'Rie+C_Rie':>10} | {'Hybrid':>10}")
print("-" * 80)
for i in range(10):
    print(f"{T[i]:8.2f} | {actual[i]:8.4f} | {pred_gp_own[i]:10.4f} | {pred_gp_riemann[i]:10.4f} | {pred_riemann_own[i]:10.4f} | {pred_hybrid[i]:10.4f}")
