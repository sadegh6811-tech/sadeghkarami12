import numpy as np

# ═══════════════════════════════════════════════════════════
# ۱. بارگذاری داده‌ها
# ═══════════════════════════════════════════════════════════
data = np.loadtxt('/data/data/com.termux/files/home/zeta_data_1000.csv', delimiter=',', skiprows=1)
T = data[:, 2]
actual = data[:, 3]

# ═══════════════════════════════════════════════════════════
# ۲. فرمول‌ها
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

def riemann_formula(T):
    return 2 * np.pi / np.log(T / (2 * np.pi) + 1e-9)

# اعمال ثابت‌ها
C_GP = 0.6625
C_RIEMANN = 0.9775
pred_gp = C_GP * gp_formula(T)
pred_riemann = C_RIEMANN * riemann_formula(T)

# ═══════════════════════════════════════════════════════════
# ۳. آماده‌سازی داده برای نمودار (حذف نقاط پرت)
# ═══════════════════════════════════════════════════════════
# برای اینکه نمودار خوانا باشد، نقاط پرت را حذف می‌کنیم
mask = np.isfinite(pred_gp) & np.isfinite(pred_riemann) & np.isfinite(actual)
mask &= (actual < 10) & (pred_gp < 10) & (pred_riemann < 10)
T_plot = T[mask]
actual_plot = actual[mask]
gp_plot = pred_gp[mask]
riemann_plot = pred_riemann[mask]

# فقط ۱۰۰ نقطه اول را برای خوانایی انتخاب می‌کنیم
n_plot = min(100, len(T_plot))
T_plot = T_plot[:n_plot]
actual_plot = actual_plot[:n_plot]
gp_plot = gp_plot[:n_plot]
riemann_plot = riemann_plot[:n_plot]

# ═══════════════════════════════════════════════════════════
# ۴. تابع رسم نمودار متنی
# ═══════════════════════════════════════════════════════════
def ascii_multi_plot(x, series_dict, title, width=70, height=20):
    all_y = np.concatenate([v for v in series_dict.values()])
    y_min, y_max = np.min(all_y), np.max(all_y)
    
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    markers = {'Actual': 'O', 'GP': '*', 'Riemann': '#'}
    
    for name, y_vals in series_dict.items():
        marker = markers.get(name, '?')
        for i in range(len(x)):
            if not np.isfinite(y_vals[i]): continue
            col = int((x[i] - np.min(x)) / (np.max(x) - np.min(x) + 1e-9) * (width - 1))
            row = int((y_vals[i] - y_min) / (y_max - y_min + 1e-9) * (height - 1))
            row = height - 1 - row
            if 0 <= row < height and 0 <= col < width:
                if grid[row][col] == ' ':
                    grid[row][col] = marker
                elif grid[row][col] != marker:
                    grid[row][col] = '+'  # همپوشانی
    
    print(f"\n═══ {title} ═══")
    print(f"  ↑ Gap (y)")
    for r in grid:
        print("  |" + "".join(r))
    print("  +" + "-" * width + "→")
    print(f"   T (imag) از {np.min(x):.1f} تا {np.max(x):.1f}")
    print(f"\n  راهنما: O=Actual, *=GP, #=Riemann, +=همپوشانی")

# ═══════════════════════════════════════════════════════════
# ۵. رسم نمودار
# ═══════════════════════════════════════════════════════════
ascii_multi_plot(T_plot, {
    'Actual': actual_plot,
    'GP': gp_plot,
    'Riemann': riemann_plot
}, "مقایسه Actual vs GP vs Riemann (۱۰۰ نقطه اول)")

# ═══════════════════════════════════════════════════════════
# ۶. تحلیل خطا در بازه‌های مختلف
# ═══════════════════════════════════════════════════════════
print("\n═══ تحلیل خطا در بازه‌های مختلف T ═══")
bins = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 800), (800, 1200)]
print(f"{'بازه T':>15} | {'MSE GP':>10} | {'MSE Riemann':>12} | {'برنده':>10}")
print("-" * 55)
for lo, hi in bins:
    m = (T >= lo) & (T < hi)
    if np.sum(m) > 0:
        mse_g = np.mean((actual[m] - pred_gp[m])**2)
        mse_r = np.mean((actual[m] - pred_riemann[m])**2)
        winner = "GP" if mse_g < mse_r else "Riemann"
        print(f"{lo:>6} - {hi:<7} | {mse_g:>10.4f} | {mse_r:>12.4f} | {winner:>10}")
