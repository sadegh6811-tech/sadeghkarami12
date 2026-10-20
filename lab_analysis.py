import numpy as np

# ═══════════════════════════════════════════════════════════
# ۱. بارگذاری داده‌های آزمایشگاهی
# ═══════════════════════════════════════════════════════════
data = np.loadtxt('/data/data/com.termux/files/home/lab_data.csv', delimiter=',', skiprows=1)
X = data[:, 0]  # distance
y = data[:, 1]  # signal

print("═══ تحلیل آزمایشگاه: گام ۱ ═══\n")

# ═══════════════════════════════════════════════════════════
# ۲. برازش مدل توانی: signal = c / d^p
# ═══════════════════════════════════════════════════════════
# برای جلوگیری از تقسیم بر صفر در d=0، d=0 را حذف می‌کنیم.
mask = X > 0
X_pos = X[mask]
y_pos = y[mask]

# تبدیل لگاریتمی: log(signal) = log(c) - p * log(d)
log_y = np.log(np.abs(y_pos) + 1e-9)
log_d = np.log(X_pos)

# حداقل مربعات خطی
A = np.column_stack([np.ones(len(log_d)), log_d])
result, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
log_c, neg_p = result
c = np.exp(log_c)
p = -neg_p

print(f"[*] برازش توانی: signal ≈ {c:.2f} / d^{p:.4f}")
print(f"[*] نمای توان (p): {p:.4f}")

if p < 0.5:
    print("  ✅ p < 0.5 → کاهش بسیار کند (نزدیک به رفتار آنی)")
elif p < 1.5:
    print("  ⚠️ 0.5 < p < 1.5 → کاهش متوسط (شبه‌کلاسیک)")
else:
    print("  ❌ p > 1.5 → کاهش سریع (رفتار کلاسیک)")

# ═══════════════════════════════════════════════════════════
# ۳. شبیه‌سازی برای alpha‌های مختلف (نزدیک شدن به رفتار آنی)
# ═══════════════════════════════════════════════════════════
print("\n═══ شبیه‌سازی برای alpha‌های مختلف ═══\n")

dim = 9
n_nodes = 2**dim
coords = np.array([[int(b) for b in format(i, f'0{dim}b')] for i in range(n_nodes)])
coords = 2 * coords - 1

dist_matrix = np.zeros((n_nodes, n_nodes))
for i in range(n_nodes):
    dist_matrix[i] = np.sum(coords[i] != coords, axis=1)
adj_matrix = (dist_matrix == 1).astype(int)

def simulate_and_fit(alpha, beta=0.5, n_steps=30):
    state = np.zeros(n_nodes)
    state[0] = 100.0
    history = [state.copy()]
    deg = np.sum(adj_matrix, axis=1, keepdims=True) + 1e-9
    adj_norm = adj_matrix / deg
    
    for t in range(1, n_steps):
        neighbor_avg = adj_norm @ state
        new_state = alpha * state + beta * neighbor_avg
        history.append(new_state.copy())
        state = new_state
    
    final = history[-1]
    
    # نمونه‌گیری از گره‌ها
    X_sim, y_sim = [], []
    for target in range(1, n_nodes):
        d = dist_matrix[0, target]
        s = final[target]
        # نویز ۵٪
        noisy = s + np.random.normal(0, 0.05 * abs(s) + 0.01)
        X_sim.append(d)
        y_sim.append(noisy)
    
    X_sim = np.array(X_sim)
    y_sim = np.array(y_sim)
    
    # برازش توانی
    mask = X_sim > 0
    log_y = np.log(np.abs(y_sim[mask]) + 1e-9)
    log_d = np.log(X_sim[mask])
    A = np.column_stack([np.ones(len(log_d)), log_d])
    result, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
    p = -result[1]
    
    # همبستگی
    corr = np.corrcoef(X_sim, y_sim)[0, 1]
    
    return p, corr

print(f"{'alpha':>8} | {'p (نما)':>12} | {'همبستگی':>12} | {'وضعیت':>20}")
print("-" * 65)

for alpha in [0.5, 0.7, 0.9, 0.95, 0.99, 0.999]:
    np.random.seed(42)
    p, corr = simulate_and_fit(alpha, beta=0.5)
    if p < 0.5:
        status = "✅ شبه‌آنی"
    elif p < 1.5:
        status = "⚠️ شبه‌کلاسیک"
    else:
        status = "❌ کلاسیک"
    print(f"{alpha:>8.3f} | {p:>12.4f} | {corr:>12.4f} | {status:>20}")

# ═══════════════════════════════════════════════════════════
# ۴. معیار آنی‌بودن (Instantaneity Metric)
# ═══════════════════════════════════════════════════════════
print("\n═══ معیار آنی‌بودن ═══\n")
print("Instantaneity = 1 - |correlation|")
print("  → ۱: انتقال کاملاً آنی")
print("  → ۰: انتقال کاملاً کلاسیک\n")

# داده‌های اصلی
corr_main = np.corrcoef(X, y)[0, 1]
instantaneity = 1 - abs(corr_main)
print(f"[*] داده‌های آزمایشگاه (alpha=0.95):")
print(f"    همبستگی: {corr_main:.4f}")
print(f"    معیار آنی‌بودن: {instantaneity:.4f}")

# بهترین حالت (alpha=0.999)
np.random.seed(42)
p_best, corr_best = simulate_and_fit(0.999, beta=0.5)
instant_best = 1 - abs(corr_best)
print(f"\n[*] بهترین حالت (alpha=0.999):")
print(f"    همبستگی: {corr_best:.4f}")
print(f"    معیار آنی‌بودن: {instant_best:.4f}")

if instant_best > 0.5:
    print(f"\n  ✅ با alpha=0.999، به رفتار شبه‌آنی رسیدیم.")
else:
    print(f"\n  ⚠️ هنوز به رفتار آنی نرسیده‌ایم. باید alpha را باز هم افزایش دهیم.")
