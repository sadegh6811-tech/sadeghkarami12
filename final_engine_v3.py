"""
موتور نهایی v3: نرمال‌سازی + جستجوی پارامتری + Ensemble + Analysis + GP
هدف: R² > 0.9 در Feynman
"""
import numpy as np
import random
import sys
sys.path.insert(0, '/data/data/com.termux/files/home/mathbot')
from extended_operators_v2 import evaluate_tree_extended_v2, UNARY_OPERATORS, BINARY_OPERATORS
from physics_knowledge_base import PHYSICS_PATTERNS, evaluate_pattern

# ═══════════════════════════════════════════════════════════
# ۱. توابع پایه
# ═══════════════════════════════════════════════════════════
def tree_depth(tree):
    if isinstance(tree, str) or (isinstance(tree, tuple) and tree[0] == 'const'):
        return 0
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
    if random.random() < prob:
        return random_tree(3, features)
    if isinstance(tree, str) or (isinstance(tree, tuple) and tree[0] == 'const'):
        return tree
    if tree[0] in UNARY_OPERATORS:
        return (tree[0], mutate(tree[1], features, prob))
    return (tree[0], mutate(tree[1], features, prob), mutate(tree[2], features, prob))

def crossover(t1, t2):
    if isinstance(t1, str) or isinstance(t2, str):
        return random.choice([t1, t2])
    if t1[0] == 'const' or t2[0] == 'const':
        return random.choice([t1, t2])
    if t1[0] != t2[0]:
        return random.choice([t1, t2])
    if t1[0] in UNARY_OPERATORS:
        return (t1[0], random.choice([t1[1], t2[1]]))
    return (t1[0], random.choice([t1[1], t2[1]]), random.choice([t1[2], t2[2]]))

# ═══════════════════════════════════════════════════════════
# ۲. جستجوی پارامتری (Linear Regression روی الگو)
# ═══════════════════════════════════════════════════════════
def fit_pattern_parameters(pattern_pred, y):
    """با استفاده از Least Squares، بهترین ضریب را برای الگو پیدا می‌کند"""
    if pattern_pred is None or not np.all(np.isfinite(pattern_pred)):
        return None, float('inf')
    
    # جلوگیری از تقسیم بر صفر
    std_pred = np.std(pattern_pred)
    if std_pred < 1e-9:
        return None, float('inf')
    
    # ضریب بهینه: a = (pred · y) / (pred · pred)
    a = np.dot(pattern_pred, y) / (np.dot(pattern_pred, pattern_pred) + 1e-9)
    pred_scaled = a * pattern_pred
    
    mse = np.mean((y - pred_scaled)**2)
    return a, mse

# ═══════════════════════════════════════════════════════════
# ۳. هسته Analysis (با جستجوی پارامتری)
# ═══════════════════════════════════════════════════════════
def analysis_core_v3(X_norm, y_norm):
    """Analysis با نرمال‌سازی + جستجوی پارامتری"""
    n, d = X_norm.shape
    best_match = None
    best_r2 = -float('inf')
    
    for name, pattern in PHYSICS_PATTERNS.items():
        if pattern['vars'] != d: continue
        
        features = [f'x{i}' for i in range(d)]
        try:
            tree = pattern['template'](features)
            pred = evaluate_pattern(tree, X_norm)
            if pred is None or not np.all(np.isfinite(pred)): continue
            
            # جستجوی پارامتری
            a, mse = fit_pattern_parameters(pred, y_norm)
            if a is None: continue
            
            r2 = 1 - mse / np.var(y_norm)
            if r2 > best_r2:
                best_r2 = r2
                best_match = (name, tree, a, r2, a * pred)
        except: pass
    
    return best_match

# ═══════════════════════════════════════════════════════════
# ۴. هسته GP
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
# ۵. سنسور Ensemble (ترکیب خطی)
# ═══════════════════════════════════════════════════════════
def ensemble_sensor(analysis_result, gp_result, y):
    """سنسور Ensemble: ترکیب خطی Analysis و GP"""
    if analysis_result is None and gp_result is None:
        return None, 0.0, None
    
    preds = []
    weights = []
    
    if analysis_result is not None:
        name, tree, a, r2, pred_a = analysis_result
        preds.append(pred_a)
        weights.append(max(r2, 0.0))
    
    if gp_result is not None:
        tree_g, mse_g, pred_g = gp_result
        if np.all(np.isfinite(pred_g)):
            preds.append(pred_g)
            r2_g = 1 - mse_g / np.var(y)
            weights.append(max(r2_g, 0.0))
    
    if not preds:
        return None, 0.0, None
    
    # نرمال‌سازی وزن‌ها
    total_w = sum(weights)
    if total_w < 1e-9:
        weights = [1.0 / len(preds)] * len(preds)
    else:
        weights = [w / total_w for w in weights]
    
    # ترکیب خطی
    final_pred = sum(w * p for w, p in zip(weights, preds))
    final_mse = np.mean((y - final_pred)**2)
    final_r2 = 1 - final_mse / np.var(y)
    
    return final_pred, final_r2, weights

# ═══════════════════════════════════════════════════════════
# ۶. موتور نهایی
# ═══════════════════════════════════════════════════════════
def final_engine(X_orig, y):
    """موتور نهایی: Analysis + GP + Ensemble"""
    n, d = X_orig.shape
    
    # نرمال‌سازی
    X_norm = (X_orig - np.mean(X_orig, axis=0)) / (np.std(X_orig, axis=0) + 1e-9)
    y_mean, y_std = np.mean(y), np.std(y) + 1e-9
    y_norm = (y - y_mean) / y_std
    
    # ── هسته Analysis (روی داده نرمال‌شده) ──
    analysis_result = analysis_core_v3(X_norm, y_norm)
    
    # ── هسته GP (روی داده نرمال‌شده) ──
    features = [f'x{i}' for i in range(d)]
    gp_tree, gp_mse = gp_core(X_norm, y_norm, features, generations=50, pop_size=300)
    gp_result = None
    if gp_tree is not None:
        pred_gp = evaluate_tree_extended_v2(gp_tree, X_norm)
        if np.all(np.isfinite(pred_gp)):
            gp_result = (gp_tree, gp_mse, pred_gp)
    
    # ── سنسور Ensemble ──
    final_pred_norm, final_r2, weights = ensemble_sensor(analysis_result, gp_result, y_norm)
    
    # برگرداندن به مقیاس اصلی
    if final_pred_norm is not None:
        final_pred = final_pred_norm * y_std + y_mean
    else:
        final_pred = None
    
    return final_pred, final_r2, analysis_result, gp_result, weights

# ═══════════════════════════════════════════════════════════
# ۷. اجرا
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("═══ موتور نهایی v3 (نرمال‌سازی + Ensemble) ═══\n")
    
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
        X_orig = np.random.uniform(0.1, 2.0, (300, eq['n']))
        y = eq['formula'](X_orig)
        
        final_pred, final_r2, analysis_res, gp_res, weights = final_engine(X_orig, y)
        
        results.append((eq['name'], final_r2))
        
        a_info = f"A: {analysis_res[0]}(R²={analysis_res[3]:.3f})" if analysis_res else "A: None"
        g_info = f"G: R²={1 - gp_res[1]/np.var(y):.3f}" if gp_res else "G: None"
        w_info = f"W: [{weights[0]:.2f}, {weights[1]:.2f}]" if weights and len(weights) == 2 else ""
        
        print(f"{eq['name']:>10} ({eq['desc']:>8}): R² = {final_r2:>7.4f}")
        print(f"            {a_info} | {g_info} | {w_info}")
    
    print("\n═══ گزارش نهایی ═══")
    r2_values = [r[1] for r in results if np.isfinite(r[1])]
    print(f"میانگین R²: {np.mean(r2_values):.4f}")
    print(f"بهترین: {np.max(r2_values):.4f}")
    print(f"بدترین: {np.min(r2_values):.4f}")
    print(f"تعداد R² > 0.9: {sum(1 for r in r2_values if r > 0.9)}/{len(r2_values)}")
