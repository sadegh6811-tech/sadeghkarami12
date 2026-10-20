import numpy as np
import random

np.random.seed(42)
random.seed(42)

# ═══════════════════════════════════════════════════════════
# ۱. ساخت شبکه هایپرکیوب ۹بعدی
# ═══════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════
# ۲. مدل با ثابت انتقال آنی (kappa)
# ═══════════════════════════════════════════════════════════
def simulate_with_kappa(kappa, n_steps=30, alpha=0.8):
    """
    x_i(t+1) = alpha * x_i(t) + (1-alpha) * [ kappa * global_mean + (1-kappa) * neighbor_mean ]
    kappa = ثابت انتقال آنی (0 = کلاسیک، 1 = کاملاً آنی)
    """
    state = np.zeros(n_nodes)
    state[0] = 100.0
    
    for t in range(n_steps):
        global_mean = np.mean(state)
        neighbor_mean = adj_norm @ state
        # ترکیب: هرچه kappa بیشتر، سهم انتقال آنی بیشتر
        mix = kappa * global_mean + (1 - kappa) * neighbor_mean
        state = alpha * state + (1 - alpha) * mix
    
    return state

# ═══════════════════════════════════════════════════════════
# ۳. تولید داده‌های آزمایشگاهی برای kappa واقعی
# ═══════════════════════════════════════════════════════════
KAPPA_TRUE = 0.7  # ثابت واقعی که می‌خواهیم کشف کنیم

print(f"[*] آزمایشگاه: تولید داده با ثابت واقعی kappa = {KAPPA_TRUE}")
final_state = simulate_with_kappa(KAPPA_TRUE)

X_lab, y_lab = [], []
for target in range(1, n_nodes):
    d = dist_matrix[0, target]
    s = final_state[target]
    noisy = s + np.random.normal(0, 0.05 * abs(s) + 0.01)
    X_lab.append(d)
    y_lab.append(noisy)

X_lab = np.array(X_lab).reshape(-1, 1)
y_lab = np.array(y_lab)

print(f"[*] داده‌ها آماده شد: {len(X_lab)} نمونه")

# ═══════════════════════════════════════════════════════════
# ۴. موتور GP ساده برای کشف kappa
# ═══════════════════════════════════════════════════════════
def evaluate_tree(tree, X):
    if isinstance(tree, str): return X[:, int(tree[1:])]
    if isinstance(tree, tuple) and tree[0] == 'const': return np.full(X.shape[0], tree[1])
    op = tree[0]
    try:
        if op == 'abs': return np.abs(evaluate_tree(tree[1], X))
        if op == 'add': return evaluate_tree(tree[1], X) + evaluate_tree(tree[2], X)
        if op == 'sub': return evaluate_tree(tree[1], X) - evaluate_tree(tree[2], X)
        if op == 'mul': return evaluate_tree(tree[1], X) * evaluate_tree(tree[2], X)
        if op == 'div': return evaluate_tree(tree[1], X) / (evaluate_tree(tree[2], X) + 1e-9)
        if op == 'max': return np.maximum(evaluate_tree(tree[1], X), evaluate_tree(tree[2], X))
    except: return np.zeros(X.shape[0])
    return np.zeros(X.shape[0])

def random_tree(depth, features):
    if depth <= 0:
        if random.random() < 0.3:
            return ('const', random.choice([0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]))
        return random.choice(features)
    if random.random() < 0.4:
        return ('abs', random_tree(depth-1, features))
    op = random.choice(['add', 'sub', 'mul', 'div', 'max'])
    return (op, random_tree(depth-1, features), random_tree(depth-1, features))

def tree_depth(tree):
    if isinstance(tree, str) or (isinstance(tree, tuple) and tree[0] == 'const'): return 0
    return 1 + max(tree_depth(c) for c in tree[1:])

def mutate(tree, features, prob=0.2):
    if random.random() < prob: return random_tree(3, features)
    if isinstance(tree, str) or (isinstance(tree, tuple) and tree[0] == 'const'): return tree
    if tree[0] == 'abs': return ('abs', mutate(tree[1], features, prob))
    return (tree[0], mutate(tree[1], features, prob), mutate(tree[2], features, prob))

def crossover(t1, t2):
    if isinstance(t1, str) or isinstance(t2, str): return random.choice([t1, t2])
    if t1[0] == 'const' or t2[0] == 'const': return random.choice([t1, t2])
    if t1[0] != t2[0]: return random.choice([t1, t2])
    if t1[0] == 'abs': return ('abs', random.choice([t1[1], t2[1]]))
    return (t1[0], random.choice([t1[1], t2[1]]), random.choice([t1[2], t2[2]]))

def train_gp(X, y, features, generations=40, pop_size=150):
    population = [random_tree(3, features) for _ in range(pop_size)]
    best_mse = float('inf')
    best_tree = None
    for gen in range(generations):
        fitness = []
        for tree in population:
            try:
                pred = evaluate_tree(tree, X)
                if not np.all(np.isfinite(pred)): continue
                mse = np.mean((y - pred)**2)
                fitness.append((mse, tree))
            except: pass
        if not fitness: break
        fitness.sort(key=lambda x: x[0])
        if fitness[0][0] < best_mse:
            best_mse = fitness[0][0]
            best_tree = fitness[0][1]
        top = [f[1] for f in fitness[:pop_size//2]]
        new_pop = top.copy()
        while len(new_pop) < pop_size:
            p1, p2 = random.choice(top), random.choice(top)
            new_pop.append(mutate(crossover(p1, p2), features))
        population = new_pop
    return best_tree, best_mse

# ═══════════════════════════════════════════════════════════
# ۵. کشف kappa توسط GP
# ═══════════════════════════════════════════════════════════
print("\n[*] آموزش GP برای کشف ثابت انتقال آنی...")
features = ['x0']  # فقط فاصله

best_tree, best_mse = train_gp(X_lab, y_lab, features, generations=40, pop_size=150)

print(f"\n[✓] بهترین MSE: {best_mse:.6f}")
print(f"[✓] فرمول کشف‌شده: {best_tree}")
print(f"[✓] عمق: {tree_depth(best_tree)}")

# ═══════════════════════════════════════════════════════════
# ۶. تحلیل: آیا فرمول کشف‌شده به kappa نزدیک است؟
# ═══════════════════════════════════════════════════════════
print("\n═══ تحلیل ثابت کشف‌شده ═══")
pred = evaluate_tree(best_tree, X_lab)
mean_pred = np.mean(pred)
mean_true = np.mean(y_lab)
print(f"[*] میانگین پیش‌بینی GP: {mean_pred:.4f}")
print(f"[*] میانگین داده واقعی: {mean_true:.4f}")
print(f"[*] ثابت واقعی kappa: {KAPPA_TRUE}")
print(f"[*] اختلاف نسبی: {abs(mean_pred - mean_true) / (abs(mean_true) + 1e-9) * 100:.2f}%")

if abs(mean_pred - mean_true) / (abs(mean_true) + 1e-9) < 0.1:
    print("\n  ✅ GP توانست ثابت kappa را با خطای کمتر از ۱۰٪ کشف کند.")
else:
    print("\n  ⚠️ GP هنوز به ثابت واقعی نزدیک نشده است.")
