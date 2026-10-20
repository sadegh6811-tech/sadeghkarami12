import numpy as np
import random

np.random.seed(42)
random.seed(42)

# ساخت شبکه
dim = 9
n_nodes = 2**dim
coords = np.array([[int(b) for b in format(i, f'0{dim}b')] for i in range(n_nodes)])
coords = 2 * coords - 1
dist_matrix = np.zeros((n_nodes, n_nodes))
for i in range(n_nodes):
    dist_matrix[i] = np.sum(coords[i] != coords, axis=1)
adj_matrix = (dist_matrix == 1).astype(int)
deg = np.sum(adj_matrix, axis=1, keepdims=True) + 1e-9
adj_norm = adj_matrix / deg

# شبیه‌سازی با پارامترهای بهینه
def simulate_with_kappa(kappa, n_steps=5, alpha=0.5):
    state = np.zeros(n_nodes)
    state[0] = 100.0
    for t in range(n_steps):
        global_mean = np.mean(state)
        neighbor_mean = adj_norm @ state
        mix = kappa * global_mean + (1 - kappa) * neighbor_mean
        state = alpha * state + (1 - alpha) * mix
    return state

print("═══ کشف kappa با شبیه‌سازی بهینه ═══\n")

for kappa_true in [0.0, 0.3, 0.7]:
    np.random.seed(42)
    final_state = simulate_with_kappa(kappa_true, n_steps=5, alpha=0.5)
    
    X_lab, y_lab = [], []
    for target in range(1, n_nodes):
        d = dist_matrix[0, target]
        s = final_state[target]
        noisy = s + np.random.normal(0, 0.01 * abs(s) + 0.01)
        X_lab.append(d)
        y_lab.append(noisy)
    X_lab = np.array(X_lab).reshape(-1, 1)
    y_lab = np.array(y_lab)
    
    from scipy.optimize import curve_fit
    
    def model(d, a, b, tau):
        return a + b * np.exp(-d / (tau + 1e-9))
    
    try:
        popt, _ = curve_fit(model, X_lab.flatten(), y_lab, p0=[0.1, 1.0, 2.0], maxfev=5000)
        pred = model(X_lab.flatten(), *popt)
        r2 = 1 - np.sum((y_lab - pred)**2) / np.sum((y_lab - np.mean(y_lab))**2)
        print(f"kappa = {kappa_true}:")
        print(f"  پارامترهای برازش: a={popt[0]:.4f}, b={popt[1]:.4f}, tau={popt[2]:.4f}")
        print(f"  R² = {r2:.4f}")
        print(f"  → tau (مقیاس کاهش): {popt[2]:.4f}")
        if popt[2] > 5:
            print(f"  ✅ کاهش بسیار کند (رفتار شبه‌آنی)")
        elif popt[2] > 1:
            print(f"  ⚠️ کاهش متوسط")
        else:
            print(f"  ❌ کاهش سریع (رفتار کلاسیک)")
    except Exception as e:
        print(f"kappa = {kappa_true}: برازش ناموفق ({e})")
    print()
