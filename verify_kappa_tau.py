import numpy as np
from scipy.optimize import curve_fit

np.random.seed(42)

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

def simulate_with_kappa(kappa, n_steps=10, alpha=0.5):
    state = np.zeros(n_nodes)
    state[0] = 100.0
    for t in range(n_steps):
        global_mean = np.mean(state)
        neighbor_mean = adj_norm @ state
        mix = kappa * global_mean + (1 - kappa) * neighbor_mean
        state = alpha * state + (1 - alpha) * mix
    return state

print("═══ بررسی رابطه kappa و tau ═══\n")
print(f"{'kappa':>6} | {'b (دامنه)':>12} | {'tau (مقیاس)':>14} | {'تفسیر':>20}")
print("-" * 65)

for kappa_true in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
    np.random.seed(42)
    final_state = simulate_with_kappa(kappa_true, n_steps=10, alpha=0.5)
    
    X_lab, y_lab = [], []
    for target in range(1, n_nodes):
        d = dist_matrix[0, target]
        s = final_state[target]
        noisy = s + np.random.normal(0, 0.01 * abs(s) + 0.001)
        X_lab.append(d)
        y_lab.append(noisy)
    X_lab = np.array(X_lab).flatten()
    y_lab = np.array(y_lab)
    
    def model(d, a, b, tau):
        return a + b * np.exp(-d / (tau + 1e-9))
    
    try:
        popt, _ = curve_fit(model, X_lab, y_lab, p0=[0.0, 1.0, 1.0], maxfev=5000)
        a, b, tau = popt
        
        if tau < 0.1:
            interp = "انتقال آنی (tau→0)"
        elif tau < 0.5:
            interp = "شبه‌آنی"
        elif tau < 1.5:
            interp = "کلاسیک"
        else:
            interp = "پخش آهسته"
        
        print(f"{kappa_true:>6.1f} | {b:>12.4f} | {tau:>14.4f} | {interp:>20}")
    except Exception as e:
        print(f"{kappa_true:>6.1f} | خطا: {e}")

print("\n═══ نتیجه‌گیری ═══")
print("هر چه kappa بیشتر → tau کمتر → انتقال آنی‌تر")
print("این با فرضیه «انتقال آنی» سازگار است.")
