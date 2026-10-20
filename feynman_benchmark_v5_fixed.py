import numpy as np
import random
import sys
sys.path.insert(0, '/data/data/com.termux/files/home/mathbot')
from extended_operators_v2 import evaluate_tree_extended_v2, UNARY_OPERATORS, BINARY_OPERATORS

np.random.seed(42)
random.seed(42)

# ═══════════════════════════════════════════════════════════
# ۱. معادلات فاینمن
# ═══════════════════════════════════════════════════════════
equations = [
    {'name': 'I.9.18', 'desc': 'gravity', 'n_vars': 3,
     'formula': lambda x: 6.67e-11 * x[:, 0] * x[:, 1] / (x[:, 2]**2 + 1e-9)},
    {'name': 'I.12.1', 'desc': 'friction', 'n_vars': 3,
     'formula': lambda x: 0.5 * x[:, 0] * (x[:, 1]**2) + x[:, 2]},
    {'name': 'I.8.14', 'desc': 'distance', 'n_vars': 2,
     'formula': lambda x: np.sqrt((x[:, 0] - x[:, 1])**2 + 1e-9)},
    {'name': 'I.6.20', 'desc': 'gaussian', 'n_vars': 3,
     'formula': lambda x: np.exp(-x[:, 0]**2 / (2 * x[:, 1]**2 + 1e-9)) / (x[:, 1] * np.sqrt(2*np.pi) + 1e-9)},
    {'name': 'I.13.4', 'desc': 'kinetic', 'n_vars': 3,
     'formula': lambda x: 0.5 * x[:, 0] * (x[:, 1]**2 + x[:, 2]**2)},
    {'name': 'II.2.42', 'desc': 'coulomb', 'n_vars': 2,
     'formula': lambda x: 8.99e9 * x[:, 0] / (x[:, 1]**2 + 1e-9)},
]

def generate_data(eq, n_samples=200):
    X = np.random.uniform(0.5, 5.0, (n_samples, eq['n_vars']))
    y = eq['formula'](X)
    return X, y

# ═══════════════════════════════════════════════════════════
# ۲. الگوی عمومی توانی
# ═══════════════════════════════════════════════════════════
def fit_power_law(X, y):
    try:
        X_pos = np.abs(X) + 1e-9
        y_pos = np.abs(y) + 1e-9
        log_y = np.log(y_pos)
        log_X = np.log(X_pos)
        A = np.column_stack([np.ones(len(log_X)), log_X])
        result, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
        intercept = result[0]
        powers = result[1:]
        c = np.exp(intercept)
        pred = c * np.prod(X_pos ** powers, axis=1)
        ss_res = np.sum((y - pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return pred, r2, powers, c
    except:
        return np.zeros_like(y), -1e9, None, 0

# ═══════════════════════════════════════════════════════════
# ۳. GP
# ═══════════════════════════════════════════════════════════
def tree_depth(tree):
    if isinstance(tree, str) or (isinstance(tree, tuple) and tree[0] == 'const'): return 0
    return 1 + max(tree_depth(c) for c in tree[1:])

def random_tree(depth, features):
    if depth <= 0:
        if random.random() < 0.3:
            return ('const', random.choice([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]))
        return random.choice(features)
    if random.random() < 0.4:
        op = random.choice(list(UNARY_OPERATORS.keys()))
        return (op, random_tree(depth-1, features))
    op = random.choice(list(BINARY_OPERATORS.keys()))
    return (op, random_tree(depth-1, features), random_tree(depth-1, features))

def mutate(tree, features, prob=0.2):
    if random.random() < prob: return random_tree(3, features)
    if isinstance(tree, str) or (isinstance(tree, tuple) and tree[0] == 'const'): return tree
    if tree[0] in UNARY_OPERATORS: return (tree[0], mutate(tree[1], features, prob))
    return (tree[0], mutate(tree[1], features, prob), mutate(tree[2], features, prob))

def crossover(t1, t2):
    if isinstance(t1, str) or isinstance(t2, str): return random.choice([t1, t2])
    if t1[0] == 'const' or t2[0] == 'const': return random.choice([t1, t2])
    if t1[0] != t2[0]: return random.choice([t1, t2])
    if t1[0] in UNARY_OPERATORS: return (t1[0], random.choice([t1[1], t2[1]]))
    return (t1[0], random.choice([t1[1], t2[1]]), random.choice([t1[2], t2[2]]))

def train_gp(X, y, features, generations=50, pop_size=300):
    population = [random_tree(5, features) for _ in range(pop_size)]
    best_r2 = -1e9
    best_tree = None
    for gen in range(generations):
        fitness = []
        for tree in population:
            try:
                pred = evaluate_tree_extended_v2(tree, X)
                if not np.all(np.isfinite(pred)): continue
                ss_res = np.sum((y - pred)**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                fitness.append((r2, tree))
            except: pass
        if not fitness: break
        fitness.sort(key=lambda x: -x[0])
        if fitness[0][0] > best_r2:
            best_r2 = fitness[0][0]
            best_tree = fitness[0][1]
        top = [f[1] for f in fitness[:pop_size//2]]
        new_pop = top.copy()
        while len(new_pop) < pop_size:
            p1, p2 = random.choice(top), random.choice(top)
            new_pop.append(mutate(crossover(p1, p2), features))
        population = new_pop
    return best_tree, best_r2

# ═══════════════════════════════════════════════════════════
# ۴. اجرا
# ═══════════════════════════════════════════════════════════
print("═══ Feynman Benchmark v5 (Power-Law + GP + Ensemble) ═══\n")

results = []
for eq in equations:
    X, y = generate_data(eq, n_samples=200)
    features = [f'x{i}' for i in range(X.shape[1])]
    
    pred_pl, r2_pl, powers, c = fit_power_law(X, y)
    best_tree, r2_gp = train_gp(X, y, features, generations=50, pop_size=300)
    pred_gp = evaluate_tree_extended_v2(best_tree, X) if best_tree else np.zeros_like(y)
    
    r2s = [max(r2_pl, 0), max(r2_gp, 0)]
    total = sum(r2s)
    weights = [r/total for r in r2s] if total > 0 else [0.5, 0.5]
    pred_ensemble = weights[0] * pred_pl + weights[1] * pred_gp
    
    ss_res = np.sum((y - pred_ensemble)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2_ensemble = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    print(f"  {eq['name']:>8} ({eq['desc']:>10}): R² = {r2_ensemble:.4f}  "
          f"[PL: {r2_pl:.4f}, GP: {r2_gp:.4f}]")
    
    if powers is not None:
        powers_str = ", ".join([f"{p:.2f}" for p in powers])
        print(f"           Powers: [{powers_str}], c = {c:.3e}")
    
    results.append({'name': eq['name'], 'r2': r2_ensemble, 'r2_pl': r2_pl, 'r2_gp': r2_gp})

r2_values = [r['r2'] for r in results]
print(f"\n═══ جمع‌بندی ═══")
print(f"  میانگین R²: {np.mean(r2_values):.4f}")
print(f"  بهترین: {np.max(r2_values):.4f}")
print(f"  بدترین: {np.min(r2_values):.4f}")
print(f"  R² > 0.9: {sum(1 for r in r2_values if r > 0.9)}/{len(r2_values)}")
