import numpy as np

np.random.seed(42)

# ═══════════════════════════════════════════════════════════
# لایه ۱: ساخت شبکه هایپرکیوب ۹بعدی
# ═══════════════════════════════════════════════════════════
print("[*] ساخت شبکه هایپرکیوب ۹بعدی (۵۱۲ گره)...")
dim = 9
n_nodes = 2**dim
coords = np.array([[int(b) for b in format(i, f'0{dim}b')] for i in range(n_nodes)])
coords = 2 * coords - 1

dist_matrix = np.zeros((n_nodes, n_nodes))
for i in range(n_nodes):
    dist_matrix[i] = np.sum(coords[i] != coords, axis=1)

adj_matrix = (dist_matrix == 1).astype(int)
print(f"[*] آماده‌سازی شد: {n_nodes} گره، {int(np.sum(adj_matrix)/2)} یال")

# ═══════════════════════════════════════════════════════════
# لایه ۲: شبیه‌سازی دینامیک انتشار (نسخه اصلاح‌شده)
# ═══════════════════════════════════════════════════════════
def simulate_entanglement(n_steps=30, alpha=0.95, beta=0.5):
    """انتشار درهم‌تنیدگی با سیگنال قوی‌تر"""
    state = np.zeros(n_nodes)
    state[0] = 100.0  # سیگنال اولیه بسیار قوی!
    history = [state.copy()]
    
    # نرمال‌سازی ماتریس مجاورت
    deg = np.sum(adj_matrix, axis=1, keepdims=True) + 1e-9
    adj_norm = adj_matrix / deg
    
    for t in range(1, n_steps):
        neighbor_avg = adj_norm @ state
        new_state = alpha * state + beta * neighbor_avg
        history.append(new_state.copy())
        state = new_state
    return np.array(history)

print("\n[*] شبیه‌سازی انتشار درهم‌تنیدگی...")
history = simulate_entanglement(n_steps=30)
final_state = history[-1]

# ═══════════════════════════════════════════════════════════
# لایه ۳: تولید داده‌های آزمایشگاهی با نویز
# ═══════════════════════════════════════════════════════════
print("[*] تولید داده‌های تجربی...")
n_samples = 1000
X_lab, y_lab = [], []

for _ in range(n_samples):
    target = np.random.randint(1, n_nodes)
    d = dist_matrix[0, target]
    signal = final_state[target]
    # نویز ۵٪ نسبت به سیگنال
    noisy = signal + np.random.normal(0, 0.05 * abs(signal) + 0.01)
    X_lab.append([d])
    y_lab.append(noisy)

X_lab = np.array(X_lab)
y_lab = np.array(y_lab)

# ═══════════════════════════════════════════════════════════
# لایه ۴: تحلیل
# ═══════════════════════════════════════════════════════════
print("\n═══ نتایج آزمایشگاه مجازی ═══")
for d in range(0, 10):
    mask = (X_lab[:, 0] == d)
    if np.sum(mask) > 0:
        mean_s = np.mean(y_lab[mask])
        std_s = np.std(y_lab[mask])
        print(f"  d={d}: سیگنال = {mean_s:8.4f} ± {std_s:.4f} (n={np.sum(mask)})")

# رفع خطا: y_lab یک آرایه ۱بعدی است
corr = np.corrcoef(X_lab[:, 0], y_lab)[0, 1]
print(f"\n[*] همبستگی (فاصله، سیگنال): {corr:.4f}")

if abs(corr) < 0.2:
    print("  ✅ سیگنال تقریباً مستقل از فاصله (شاهد بر انتقال آنی).")
else:
    print("  ⚠️ سیگنال با فاصله کاهش می‌یابد (رفتار کلاسیک).")

np.savetxt('/data/data/com.termux/files/home/lab_data.csv',
           np.column_stack((X_lab.flatten(), y_lab)),
           delimiter=',', header='distance,signal', comments='')
print("\n[+] داده‌های آزمایشگاهی ذخیره شد: lab_data.csv")
