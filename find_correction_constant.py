import numpy as np
import mpmath as mp

# ═══════════════════════════════════════════════════════════
# ۱. تولید ۱۰۰۰ صفر جدید (اگر قبلاً نساخته‌اید)
# ═══════════════════════════════════════════════════════════
import os
if not os.path.exists('/data/data/com.termux/files/home/zeta_data_1000.csv'):
    print("[*] تولید ۱۰۰۰ صفر تابع زتا...")
    mp.mp.dps = 20
    N = 1000
    zeros = []
    for i in range(1, N + 1):
        z = mp.zetazero(i)
        zeros.append((i, float(mp.re(z)), float(mp.im(z))))
        if i % 200 == 0:
            print(f"    {i} صفر محاسبه شد...")
    data = np.array(zeros)
    gaps = np.diff(data[:, 2], prepend=data[0, 2])
    final_data = np.column_stack([data, gaps])
    np.savetxt('/data/data/com.termux/files/home/zeta_data_1000.csv', 
               final_data, delimiter=',', header='index,real,imag,gap', comments='')
    print(f"[+] داده ذخیره شد.")

# بارگذاری داده‌ها
data_300 = np.loadtxt('/data/data/com.termux/files/home/zeta_data.csv', delimiter=',', skiprows=1)
data_1000 = np.loadtxt('/data/data/com.termux/files/home/zeta_data_1000.csv', delimiter=',', skiprows=1)

# ═══════════════════════════════════════════════════════════
# ۲. فرمول GP (که در ۳۰۰ نقطه MSE=0.513 داشت)
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
# ۳. فرمول ریمان-فون مانگولت (برای مقایسه)
# ═══════════════════════════════════════════════════════════
def riemann_formula(T):
    return 2 * np.pi / np.log(T / (2 * np.pi) + 1e-9)

# ═══════════════════════════════════════════════════════════
# ۴. یافتن ثابت بهینه C
# ═══════════════════════════════════════════════════════════
def find_best_constant(data, use_riemann=False):
    T = data[:, 2]
    actual = data[:, 3]
    
    if use_riemann:
        base = riemann_formula(T)
    else:
        base = gp_formula(T)
    
    # حذف مقادیر نامعتبر
    mask = np.isfinite(base) & np.isfinite(actual)
    base = base[mask]
    actual = actual[mask]
    
    best_C = 1.0
    best_mse = float('inf')
    
    # جستجوی شبکه‌ای برای C
    for C in np.linspace(0.1, 3.0, 300):
        pred = C * base
        mse = np.mean((actual - pred)**2)
        if mse < best_mse:
            best_mse = mse
            best_C = C
    
    return best_C, best_mse

# ═══════════════════════════════════════════════════════════
# ۵. اجرا
# ═══════════════════════════════════════════════════════════
print("\n═══ ۱. تحلیل روی ۳۰۰ صفر اول ═══")
C_300, mse_300 = find_best_constant(data_300, use_riemann=False)
print(f"ثابت بهینه C برای فرمول GP (۳۰۰ صفر): {C_300:.4f}")
print(f"MSE با ثابت: {mse_300:.6f}")

print("\n═══ ۲. تحلیل روی ۱۰۰۰ صفر ═══")
C_1000, mse_1000 = find_best_constant(data_1000, use_riemann=False)
print(f"ثابت بهینه C برای فرمول GP (۱۰۰۰ صفر): {C_1000:.4f}")
print(f"MSE با ثابت: {mse_1000:.6f}")

print("\n═══ ۳. مقایسه با فرمول ریمان-فون (با ثابت) ═══")
C_riemann, mse_riemann = find_best_constant(data_1000, use_riemann=True)
print(f"ثابت بهینه C برای فرمول ریمان (۱۰۰۰ صفر): {C_riemann:.4f}")
print(f"MSE با ثابت: {mse_riemann:.6f}")

print("\n═══ ۴. آیا ثابت C یکسان است؟ ═══")
print(f"C در ۳۰۰ صفر: {C_300:.4f}")
print(f"C در ۱۰۰۰ صفر: {C_1000:.4f}")
if abs(C_300 - C_1000) < 0.1:
    print("✅ ثابت C پایدار است! این یعنی فرمول GP یک قانون جهانی دارد.")
else:
    print("⚠️ ثابت C تغییر می‌کند. این یعنی فرمول GP به اندازه‌ی کافی عمومی نیست.")

# ═══════════════════════════════════════════════════════════
# ۶. نمودار مقایسه
# ═══════════════════════════════════════════════════════════
print("\n═══ ۵. مقایسه در ۱۰ صفر اول (۱۰۰۰ نقطه) ═══")
T = data_1000[:10, 2]
actual = data_1000[:10, 3]
pred_C = C_1000 * gp_formula(T)
pred_riemann = C_riemann * riemann_formula(T)

print(f"{'T':>8} | {'Actual':>8} | {'GP+C':>8} | {'Riemann+C':>8}")
print("-" * 45)
for i in range(10):
    print(f"{T[i]:8.2f} | {actual[i]:8.4f} | {pred_C[i]:8.4f} | {pred_riemann[i]:8.4f}")
