import numpy as np
import time

np.random.seed(42)

# ═══════════════════════════════════════════════════════════
# لایه ۱: ساخت شبکه‌ی هایپرکیوب ۹بعدی (فضای فیزیکی)
# ═══════════════════════════════════════════════════════════
print("[*] ساخت شبکه‌ی هایپرکیوب ۹بعدی (۵۱۲ گره)...")
dim = 9
n_nodes = 2**dim

# مختصات گره‌ها در فضای ۹بعدی {+1, -1}
coords = np.array([[int(b) for b in format(i, f'0{dim}b')] for i in range(n_nodes)])
coords = 2 * coords - 1

# ماتریس فاصله همینگ (Hamming Distance)
# فاصله بین دو گره = تعداد مختصاتی که با هم تفاوت دارند
dist_matrix = np.zeros((n_nodes, n_nodes))
for i in range(n_nodes):
    dist_matrix[i] = np.sum(coords[i] != coords, axis=1)

# ماتریس مجاورت (یال‌ها)
adj_matrix = (dist_matrix == 1).astype(int)

print(f"[*] آماده‌سازی شد: {n_nodes} گره، {int(np.sum(adj_matrix)/2)} یال")

# ═══════════════════════════════════════════════════════════
# لایه ۲: شبیه‌سازی دینامیک انتشار (فیزیک کوانتومی/کلاسیک)
# ═══════════════════════════════════════════════════════════
def simulate_entanglement_decay(n_steps=15, alpha=0.8, beta=0.2):
    """
    شبیه‌سازی انتشار درهم‌تنیدگی.
    alpha: ضریب نگهداشت حالت (Decay)
    beta: ضریب انتقال به همسایه‌ها (Coupling)
    """
    # حالت اولیه: گره 0 دارای انرژی (اسپین +1)، بقیه صفر
    state = np.zeros(n_nodes)
    state[0] = 1.0 
    
    history = [state.copy()]
    
    for t in range(1, n_steps):
        # معادله انتشار: حالت جدید = (حالت قبلی * alpha) + (میانگین همسایه‌ها * beta)
        # برای شبیه‌سازی انتقال آنی، نیازی به تأخیر (delay) نیست.
        neighbor_sum = adj_matrix @ state
        new_state = alpha * state + beta * (neighbor_sum / (np.sum(adj_matrix, axis=1) + 1e-9))
        
        history.append(new_state.copy())
        state = new_state
        
    return np.array(history)

# اجرای شبیه‌سازی
print("\n[*] شبیه‌سازی انتشار درهم‌تنیدگی در زمان...")
history = simulate_entanglement_decay(n_steps=15)

# ═══════════════════════════════════════════════════════════
# لایه ۳: تولید داده‌های آزمایشگاهی (با نویز دستگاه)
# ═══════════════════════════════════════════════════════════
print("[*] تولید داده‌های تجربی (با نویز دستگاه ۵٪)...")
n_samples = 500
X_lab = []
y_lab = []

for _ in range(n_samples):
    # یک گره تصادفی (حسگر) و یک گره مرجع (منبع) انتخاب می‌کنیم
    target_node = np.random.randint(1, n_nodes)
    source_node = 0
    
    # فاصله هندسی (هایپرکیوب)
    d_9d = dist_matrix[source_node, target_node]
    
    # زمان رسیدن سیگنال (از شبیه‌سازی)
    # ما مقدار نهایی حالت را به عنوان "شدت سیگنال دریافتی" در نظر می‌گیریم
    signal_strength = history[-1, target_node]
    
    # اعمال نویز گاوسی (مشابه نویز فوتون در آزمایشگاه)
    noisy_signal = signal_strength + np.random.normal(0, 0.05)
    
    X_lab.append([d_9d])
    y_lab.append(noisy_signal)

X_lab = np.array(X_lab)
y_lab = np.array(y_lab)

# ═══════════════════════════════════════════════════════════
# لایه ۴: تحلیل داده‌ها (آیا اثر فاصله دیده می‌شود؟)
# ═══════════════════════════════════════════════════════════
print("\n═══ نتایج آزمایشگاه مجازی ═══")

# گروه‌بندی داده‌ها بر اساس فاصله (۰ تا ۹)
for d in range(0, 10):
    mask = (X_lab[:, 0] == d)
    if np.sum(mask) > 0:
        mean_signal = np.mean(y_lab[mask])
        std_signal = np.std(y_lab[mask])
        print(f"  فاصله d={d}: سیگنال دریافتی = {mean_signal:.4f} ± {std_signal:.4f} (n={np.sum(mask)})")

# محاسبه همبستگی بین فاصله و سیگنال دریافتی
corr = np.corrcoef(X_lab[:, 0], y_lab[:, 1])[0, 1]
print(f"\n[*] همبستگی (فاصله، سیگنال): {corr:.4f}")

if abs(corr) < 0.2:
    print("  ✅ نتیجه: سیگنال تقریباً مستقل از فاصله است (شاهد بر انتقال غیرمحلی).")
else:
    print("  ⚠️ نتیجه: سیگنال با فاصله کاهش می‌یابد (رفتار کلاسیک).")

# ذخیره داده برای ربات
np.savetxt('/data/data/com.termux/files/home/lab_data.csv', 
           np.column_stack((X_lab, y_lab)), 
           delimiter=',', header='distance,signal', comments='')
print("\n[+] داده‌های آزمایشگاهی ذخیره شد: lab_data.csv")
