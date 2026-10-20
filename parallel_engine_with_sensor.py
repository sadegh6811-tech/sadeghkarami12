"""
معماری موازی با سنسور هوشمند
- هسته ۴: تحلیل‌گر (روی داده اصلی)
- هسته GP: جستجوگر (روی داده نرمال‌شده)
- سنسور: تصمیم‌گیری بر اساس اعتماد
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
# ۲. هسته GP (مستقل، روی داده نرمال‌شده)
# ═══════════════════════════════════════════════════════════
def gp_core(X_norm, y, features, generations=50, pop_size=300):
    population = [random_tree(5, features) for _ in range(pop_size)]
    best_mse = float('inf')
    best_tree = None
    
    for gen in range(generations):
        fitness = []
        for tree in population:
            try:
                pred = evaluate_tree_extended_v2(tree, X_norm)
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
# ۳. هسته تحلیل‌گر (روی داده اصلی)
# ═══════════════════════════════════════════════════════════
def analysis_core(X_orig, y):
    n, d = X_orig.shape
    best_match = None
    best_corr = 0.0
    
    for name, pattern in PHYSICS_PATTERNS.items():
        if pattern['vars'] != d: continue
        
        features = [f'x{i}' for i in range(d)]
        try:
            tree = pattern['template'](features)
            pred = evaluate_pattern(tree, X_orig)
            if pred is None or not np.all(np.isfinite(pred)): continue
            if np.std(pred) < 1e-9: continue
            
            corr = abs(np.corrcoef(pred, y)[0, 1])
            if corr > best_corr:
                best_corr = corr
                best_match = (name, tree, corr, pred)
        except: pass
    
    return best_match

# ═══════════════════════════════════════════════════════════
# ۴. سنسور هوشمند (Intelligent Sensor)
# ═══════════════════════════════════════════════════════════
def intelligent_sensor(analysis_result, gp_result, y):
    """
    سنسور تصمیم‌گیری:
    - analysis_result: (name, tree, corr, pred) یا None
    - gp_result: (tree, mse, pred) یا None
    - y: هدف واقعی
    """
    scores = {}
    
    # ── ارزیابی تحلیل‌گر ──
    if analysis_result is not None:
        name, tree, corr, pred_a = analysis_result
        mse_a = np.mean((y - pred_a)**2)
        r2_a = 1 - mse_a / np.var(y)
        
        # اعتماد بر اساس همبستگی
        trust_a = min(1.0, corr / 0.95)  # 0 تا 1
        scores['Analysis'] = {
            'score': r2_a * trust_a,
            'r2': r2_a,
            'trust': trust_a,
            'law': name,
            'tree': tree
        }
    
    # ── ارزیابی GP ──
    if gp_result is not None:
        tree_g, mse_g, pred_g = gp_result
        r2_g = 1 - mse_g / np.var(y) if mse_g < float('inf') else -float('inf')
        
        # اعتماد بر اساس سادگی درخت
        depth = tree_depth(tree_g)
        trust_g = max(0.3, 1.0 - depth / 10)  # درخت ساده‌تر، اعتماد بیشتر
        
        scores['GP'] = {
            'score': r2_g * trust_g,
            'r2': r2_g,
            'trust': trust_g,
            'depth': depth,
            'tree': tree_g
        }
    
    # ── تصمیم نهایی ──
    if not scores:
        return None, 'None', 0.0
    
    # انتخاب برنده
    winner = max(scores.keys(), key=lambda k: scores[k]['score'])
    
    # اگر هر دو توافق داشتند (R² مشابه)، Analysis را ترجیح بده (چون تفسیرپذیرتر است)
    if 'Analysis' in scores and 'GP' in scores:
        diff = abs(scores['Analysis']['r2'] - scores['GP']['r2'])
        if diff < 0.05 and scores['Analysis']['r2'] > 0.7:
            winner = 'Analysis'
    
    return scores[winner]['tree'], winner, scores[winner]['r2']

# ═══════════════════════════════════════════════════════════
# ۵. موتور موازی با سنسور
# ═══════════════════════════════════════════════════════════
def parallel_engine_with_sensor(X_orig, y, features, generations=50, pop_size=300):
    """موتور موازی با سنسور هوشمند"""
    # داده نرمال‌شده برای GP
    X_norm = (X_orig - np.mean(X_orig, axis=0)) / (np.std(X_orig, axis=0) + 1e-9)
    
    # ── اجرای موازی ──
    # ۱. تحلیل‌گر (روی داده اصلی)
    analysis_result = analysis_core(X_orig, y)
    
    # ۲. GP (روی داده نرمال‌شده)
    gp_tree, gp_mse = gp_core(X_norm, y, features, generations=generations, pop_size=pop_size)
    gp_result = None
    if gp_tree is not None:
        pred_gp = evaluate_tree_extended_v2(gp_tree, X_norm)
        if np.all(np.isfinite(pred_gp)):
            gp_result = (gp_tree, gp_mse, pred_gp)
    
    # ── سنسور تصمیم می‌گیرد ──
    final_tree, winner, final_r2 = intelligent_sensor(analysis_result, gp_result, y)
    
    return final_tree, winner, final_r2, analysis_result, gp_result

# ═══════════════════════════════════════════════════════════
# ۶. اجرا
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("═══ معماری موازی با سنسور هوشمند ═══\n")
    
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
        features = [f'x{i}' for i in range(eq['n'])]
        
        final_tree, winner, final_r2, analysis_res, gp_res = parallel_engine_with_sensor(
            X_orig, y, features, generations=50, pop_size=300
        )
        
        results.append((eq['name'], final_r2, winner))
        
        # نمایش جزئیات
        a_info = f"Analysis: R²={analysis_res[2]:.3f}" if analysis_res else "Analysis: None"
        g_info = f"GP: R²={1 - gp_res[1]/np.var(y):.3f}" if gp_res else "GP: None"
        
        print(f"{eq['name']:>10} ({eq['desc']:>8}): R² = {final_r2:>7.4f} | برنده: {winner:>8}")
        print(f"            {a_info} | {g_info}")
    
    print("\n═══ گزارش نهایی ═══")
    r2_values = [r[1] for r in results if np.isfinite(r[1])]
    print(f"میانگین R²: {np.mean(r2_values):.4f}")
    print(f"پیروزی Analysis: {sum(1 for r in results if r[2] == 'Analysis')}")
    print(f"پیروزی GP: {sum(1 for r in results if r[2] == 'GP')}")
