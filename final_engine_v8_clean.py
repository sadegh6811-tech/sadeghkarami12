"""
موتور نهایی v8: نسخه تمیز و بدون باگ
- Analysis (روی داده اصلی)
- Power-Law (روی داده با offset مثبت)
- GP (روی داده نرمال‌شده)
- Ensemble (با وزن‌دهی ایمن)
"""
import numpy as np
import random
import sys
sys.path.insert(0, '/data/data/com.termux/files/home/mathbot')
from extended_operators_v2 import evaluate_tree_extended_v2, UNARY_OPERATORS, BINARY_OPERATORS
from physics_knowledge_base import PHYSICS_PATTERNS, evaluate_pattern

np.random.seed(42)
random.seed(42)

# ═══════════════════════════════════════════════════════════
# ۱. توابع پایه
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

# ═══════════════════════════════════════════════════════════
# ۲. محاسبه R² ایمن
# ═══════════════════════════════════════════════════════════
def safe_r2(y_true, y_pred):
    if y_pred is None or not np.all(np.isfinite(y_pred)):
        return -1e9
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot < 1e-12:
        return -1e9
    r2 = 1 - ss_res / ss_tot
    return r2 if np.isfinite(r2) else -1e9

# ═══════════════════════════════════════════════════════════
# ۳. Analysis Core (ایمن)
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
            
            # جستجوی پارامتری ایمن
            a = np.dot(pred, y) / (np.dot(pred, pred) + 1e-9)
            if not np.isfinite(a): continue
            
            pred_scaled = a * pred
            r2 = safe_r2(y, pred_scaled)
            
            if r2 > best_r2:
                best_r2 = r2
                best_match = (name, tree, a, r2, pred_scaled)
        except: pass
    
    return best_match

# ═══════════════════════════════════════════════════════════
# ۴. Power-Law Core (با offset برای مقادیر منفی)
# ═══════════════════════════════════════════════════════════
def power_law_core(X_orig, y):
    # offset برای همه مقادیر (حتی منفی)
    offset_X = max(0.01, np.min(np.abs(X_orig)) * 0.1) if np.min(np.abs(X_orig)) > 0 else 1.0
    offset_y = max(0.01, np.min(np.abs(y)) * 0.1) if np.min(np.abs(y)) > 0 else 1.0
    
    X_pos = np.abs(X_orig) + offset_X
    y_pos = np.abs(y) + offset_y
    
    try:
        log_y = np.log(y_pos)
        log_X = np.log(X_pos)
        
        A = np.column_stack([np.ones(len(log_X)), log_X])
        result, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)
        
        intercept = result[0]
        powers = result[1:]
        c = np.exp(intercept)
        
        pred = c * np.prod(X_pos ** powers, axis=1)
        r2 = safe_r2(y, pred)
        
        if r2 < -10:
            return None
        
        return (pred, r2, powers, c)
    except:
        return None

# ═══════════════════════════════════════════════════════════
# ۵. GP Core (روی داده نرمال‌شده)
# ═══════════════════════════════════════════════════════════
def gp_core(X_norm, y_norm, features, generations=50, pop_size=300):
    population = [random_tree(5, features) for _ in range(pop_size)]
    best_mse = float('inf')
    best_tree = None
    
    for gen in range(generations):
        fitness = []
        for tree in population:
            try:
                pred = evaluate_tree_extended_v2(tree, X_norm)
                if not np.all(np.isfinite(pred)): continue
                mse = np.mean((y_norm - pred)**2)
                if not np.isfinite(mse): continue
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
# ۶. Ensemble Sensor (ایمن)
# ═══════════════════════════════════════════════════════════
def ensemble_sensor(candidates, y):
    """candidates: list of (name, pred, r2)"""
    valid = [(name, pred, r2) for name, pred, r2 in candidates 
             if pred is not None and np.all(np.isfinite(pred)) and r2 > 0.1]
    
    if not valid:
        return None, -1e9, []
    
    total_r2 = sum(r2 for _, _, r2 in valid)
    weights = [r2 / total_r2 for _, _, r2 in valid]
    final_pred = sum(w * pred for w, (_, pred, _) in zip(weights, valid))
    final_r2 = safe_r2(y, final_pred)
    
    return final_pred, final_r2, list(zip([v[0] for v in valid], weights))

# ═══════════════════════════════════════════════════════════
# ۷. موتور نهایی
# ═══════════════════════════════════════════════════════════
def final_engine(X_orig, y):
    n, d = X_orig.shape
    X_norm = (X_orig - np.mean(X_orig, axis=0)) / (np.std(X_orig, axis=0) + 1e-9)
    y_mean, y_std = np.mean(y), np.std(y) + 1e-9
    y_norm = (y - y_mean) / y_std
    
    candidates = []
    
    # ۱. Analysis
    analysis_res = analysis_core(X_orig, y)
    if analysis_res is not None:
        candidates.append((f"A:{analysis_res[0]}", analysis_res[4], analysis_res[3]))
    
    # ۲. Power-Law
    pl_res = power_law_core(X_orig, y)
    if pl_res is not None:
        candidates.append(("PL", pl_res[0], pl_res[1]))
    
    # ۳. GP
    features = [f'x{i}' for i in range(d)]
    gp_tree, gp_mse = gp_core(X_norm, y_norm, features, generations=50, pop_size=300)
    if gp_tree is not None:
        pred_gp_norm = evaluate_tree_extended_v2(gp_tree, X_norm)
        if np.all(np.isfinite(pred_gp_norm)):
            pred_gp = pred_gp_norm * y_std + y_mean
            r2_gp = safe_r2(y, pred_gp)
            candidates.append(("GP", pred_gp, r2_gp))
    
    # ۴. Ensemble
    final_pred, final_r2, weights = ensemble_sensor(candidates, y)
    
    return final_pred, final_r2, candidates, weights

# ═══════════════════════════════════════════════════════════
# ۸. اجرا
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("═══ موتور نهایی v8 (بدون باگ) ═══\n")
    
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
        
        final_pred, final_r2, candidates, weights = final_engine(X_orig, y)
        results.append((eq['name'], final_r2))
        
        print(f"{eq['name']:>10} ({eq['desc']:>8}): R² = {final_r2:>7.4f}")
        for name, pred, r2 in candidates:
            print(f"            {name}: R² = {r2:.3f}")
        if weights:
            print(f"            weights: {[(w[0], f'{w[1]:.2f}') for w in weights]}")
        print()
    
    print("═══ گزارش نهایی ═══")
    r2_values = [r[1] for r in results if np.isfinite(r[1])]
    print(f"میانگین R²: {np.mean(r2_values):.4f}")
    print(f"بهترین: {np.max(r2_values):.4f}")
    print(f"بدترین: {np.min(r2_values):.4f}")
    print(f"R² > 0.9: {sum(1 for r in r2_values if r > 0.9)}/{len(r2_values)}")
