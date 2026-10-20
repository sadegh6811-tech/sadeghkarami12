"""
موتور نهایی v7: Analysis (v4) + Power-Law (روی داده خام) + GP (نرمال‌شده)
"""
import numpy as np
import random
import sys
sys.path.insert(0, '/data/data/com.termux/files/home/mathbot')
from extended_operators_v2 import evaluate_tree_extended_v2, UNARY_OPERATORS, BINARY_OPERATORS
from physics_knowledge_base import PHYSICS_PATTERNS, evaluate_pattern

np.random.seed(42)
random.seed(42)

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

# ═══════════════════════════════════════════════════════════
# Analysis (همان v4)
# ═══════════════════════════════════════════════════════════
def analysis_core(X_orig, y):
    n, d = X_orig.shape
    best_match = None
    best_r2 = -float('inf')
    
    for name, pattern in PHYSICS_PATTERNS.items():
        if pattern['vars'] != d: continue
        features = [f'x{i}' for i in range(d)]
        try:
            tree = pattern['template'](features)
            pred = evaluate_pattern(tree, X_orig)
            if pred is None or not np.all(np.isfinite(pred)): continue
            if np.std(pred) < 1e-9: continue
            
            a = np.dot(pred, y) / (np.dot(pred, pred) + 1e-9)
            pred_scaled = a * pred
            mse = np.mean((y - pred_scaled)**2)
            r2 = 1 - mse / np.var(y)
            
            if r2 > best_r2:
                best_r2 = r2
                best_match = (name, tree, a, r2, pred_scaled)
        except: pass
    
    return best_match

# ═══════════════════════════════════════════════════════════
# Power-Law (روی داده خام — فقط اگر همه مقادیر مثبت باشند)
# ═══════════════════════════════════════════════════════════
def power_law_core_raw(X_orig, y):
    # فقط اگر همه X و y مثبت باشند
    if np.any(X_orig <= 0) or np.any(y <= 0):
        return None
    
    try:
        log_y = np.log(y)
        log_X = np.log(X_orig)
        
        A = np.column_stack([np.ones(len(log_X)), log_X])
        result, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
        
        intercept = result[0]
        powers = result[1:]
        c = np.exp(intercept)
        
        pred = c * np.prod(X_orig ** powers, axis=1)
        
        mse = np.mean((y - pred)**2)
        r2 = 1 - mse / np.var(y)
        
        if r2 < -10 or not np.isfinite(r2):
            return None
        
        return pred, r2, powers, c
    except:
        return None

# ═══════════════════════════════════════════════════════════
# GP
# ═══════════════════════════════════════════════════════════
def gp_core(X, y, features, generations=50, pop_size=300):
    population = [random_tree(5, features) for _ in range(pop_size)]
    best_mse = float('inf')
    best_tree = None
    for gen in range(generations):
        fitness = []
        for tree in population:
            try:
                pred = evaluate_tree_extended_v2(tree, X)
                if not np.all(np.isfinite(pred)): continue
                mse = np.mean((y - pred)**2)
                fitness.append((mse + 0.001 * tree_depth(tree), mse, tree))
            except: pass
        if not fitness: break
        fitness.sort(key=lambda x: x[0])
        if fitness[0][1] < best_mse:
            best_mse = fitness[0][1]
            best_tree = fitness[0][2]
        top = [f[2] for f in fitness[:pop_size//2]]
        new_pop = top.copy()
        while len(new_pop) < pop_size:
            p1, p2 = random.choice(top), random.choice(top)
            new_pop.append(mutate(crossover(p1, p2), features))
        population = new_pop
    return best_tree, best_mse

# ═══════════════════════════════════════════════════════════
# Ensemble سه‌گانه
# ═══════════════════════════════════════════════════════════
def ensemble_sensor(analysis_res, pl_res, gp_res, y):
    preds = []
    weights = []
    
    if analysis_res is not None and analysis_res[3] > 0.1:
        preds.append(analysis_res[4])
        weights.append(analysis_res[3])
    
    if pl_res is not None and pl_res[1] > 0.1:
        preds.append(pl_res[0])
        weights.append(pl_res[1])
    
    if gp_res is not None:
        pred_gp = gp_res[2] * np.std(y) + np.mean(y)
        r2_gp = 1 - gp_res[1] / np.var(y)
        if r2_gp > 0.1:
            preds.append(pred_gp)
            weights.append(r2_gp)
    
    if not preds:
        return None, -1e9, [0,0,0]
    
    total = sum(weights)
    weights_norm = [w/total for w in weights] if total > 0 else [1.0/len(preds)] * len(preds)
    
    final_pred = sum(w * p for w, p in zip(weights_norm, preds))
    final_mse = np.mean((y - final_pred)**2)
    final_r2 = 1 - final_mse / np.var(y)
    
    while len(weights_norm) < 3: weights_norm.append(0)
    
    return final_pred, final_r2, weights_norm[:3]

# ═══════════════════════════════════════════════════════════
# موتور نهایی
# ═══════════════════════════════════════════════════════════
def final_engine(X_orig, y):
    n, d = X_orig.shape
    X_norm = (X_orig - np.mean(X_orig, axis=0)) / (np.std(X_orig, axis=0) + 1e-9)
    y_mean, y_std = np.mean(y), np.std(y) + 1e-9
    y_norm = (y - y_mean) / y_std
    
    # ۱. Analysis روی داده اصلی
    analysis_res = analysis_core(X_orig, y)
    
    # ۲. Power-Law روی داده اصلی (اگر همه مثبت باشند)
    pl_res = power_law_core_raw(X_orig, y)
    
    # ۳. GP روی داده نرمال‌شده
    features = [f'x{i}' for i in range(d)]
    gp_tree, gp_mse = gp_core(X_norm, y_norm, features, generations=50, pop_size=300)
    gp_res = None
    if gp_tree is not None:
        pred_gp = evaluate_tree_extended_v2(gp_tree, X_norm)
        if np.all(np.isfinite(pred_gp)):
            gp_res = (gp_tree, gp_mse, pred_gp)
    
    # ۴. Ensemble
    final_pred, final_r2, weights = ensemble_sensor(analysis_res, pl_res, gp_res, y)
    
    return final_pred, final_r2, analysis_res, pl_res, gp_res, weights

# ═══════════════════════════════════════════════════════════
# اجرا
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("═══ موتور نهایی v7 (Power-Law روی داده خام) ═══\n")
    
    equations = [
        {'name': 'I.8.14', 'formula': lambda x: np.sqrt((x[:,0]-x[:,1])**2 + (x[:,2]-x[:,3])**2), 'n': 4, 'desc': 'فاصله'},
        {'name': 'I.9.18', 'formula': lambda x: (x[:,0]*x[:,1])/((x[:,2]-x[:,3])**2 + 1e-6), 'n': 4, 'desc': 'گرانش'},
        {'name': 'II.2.42', 'formula': lambda x: x[:,0]/(x[:,1]**2 + 1e-6), 'n': 2, 'desc': 'کولن'},
        {'name': 'I.6.20', 'formula': lambda x: np.exp(-(x[:,0]**2)/(2*x[:,1]**2))/(np.sqrt(2*np.pi)*x[:,1]), 'n': 2, 'desc': 'گاوسی'},
        {'name': 'I.12.1', 'formula': lambda x: x[:,0]*x[:,1]*x[:,2], 'n': 3, 'desc': 'اصطکاک'},
        {'name': 'I.13.4', 'formula': lambda x: 0.5*x[:,0]*(x[:,1]**2 + x[:,2]**2), 'n': 3, 'desc': 'جنبشی'},
    ]
    
    results = []
    for eq in equations:
        np.random.seed(42)
        X_orig = np.random.uniform(0.5, 2.0, (300, eq['n']))
        y = eq['formula'](X_orig)
        
        final_pred, final_r2, a_res, pl_res, g_res, weights = final_engine(X_orig, y)
        results.append((eq['name'], final_r2))
        
        a_info = f"A: {a_res[0]}({a_res[3]:.3f})" if a_res else "A: None"
        pl_info = f"PL: {pl_res[1]:.3f}" if pl_res else "PL: None"
        g_info = f"G: {1-g_res[1]/np.var(y):.3f}" if g_res else "G: None"
        
        print(f"{eq['name']:>10} ({eq['desc']:>8}): R² = {final_r2:>7.4f}")
        print(f"            {a_info} | {pl_info} | {g_info}")
    
    print("\n═══ گزارش نهایی ═══")
    r2_values = [r[1] for r in results if np.isfinite(r[1])]
    print(f"میانگین R²: {np.mean(r2_values):.4f}")
    print(f"بهترین: {np.max(r2_values):.4f}")
    print(f"بدترین: {np.min(r2_values):.4f}")
    print(f"R² > 0.9: {sum(1 for r in r2_values if r > 0.9)}/{len(r2_values)}")
