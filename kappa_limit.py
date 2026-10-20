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

print("═══ آزمایش حدی: kappa → 1 ═══\n")
print(f"{'kappa':>8} | {'tau':>10} | {'نتیجه':>20}")
print("-" * 45)

for kappa in [0.9, 0.95, 0.99, 0.999, 0.9999]:
    np.random.seed(42)
    final_state = simulate_with_kappa(kappa, n_steps=10, alpha=0.5)
    
    X_lab, y_lab = [], []
    for target in range(1, n_nodes):
        d = dist_matrix[0, target]
        s = final_state[target]
        noisy = s + np.random.normal(0, 0.001 * abs(s) + 0.0001)
        X_lab.append(d)
        y_lab.append(noisy)
    X_lab = np.array(X_lab).flatten()
    y_lab = np.array(y_lab)
    
    def model(d, a, b, tau):
        return a + b * np.exp(-d / (tau + 1e-9))
    
    try:
        popt, _ = curve_fit(model, X_lab, y_lab, p0=[0.1, 0.1, 0.5], maxfev=5000)
        tau = popt[2]
        status = "آنی (tau<0.1)" if tau < 0.1 else "شبه‌آنی" if tau < 0.5 else "کلاسیک"
        print(f"{kappa:>8.4f} | {tau:>10.4f} | {status:>20}")
    except Exception as e:
        print(f"{kappa:>8.4f} | برازش ناموفق: {e}")

print("\n═══ نتیجه‌گیری ═══")
print("اگر tau با kappa → 1 به صفر میل کند، انتقال آنی تأیید می‌شود.")
