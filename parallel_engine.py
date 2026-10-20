"""
معماری موازی: هسته ۴ (تحلیل‌گر) + هسته GP (جستجوگر)
"""
import numpy as np
import random
import sys
sys.path.insert(0, '/data/data/com.termux/files/home/mathbot')
from extended_operators_v2 import evaluate_tree_extended_v2, UNARY_OPERATORS, BINARY_OPERATORS
from physics_knowledge_base import PHYSICS_PATTERNS, detect_physics_law

# ═══════════════════════════════════════════════════════════
# ۱. موتور GP (جستجوگر)
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
# ۲. هسته GP (جستجوگر)
# ═══════════════════════════════════════════════════════════
def gp_core(X, y, features, generations=50, pop_size=300, seed_tree=None):
    """هسته GP: جستجوی تصادفی در فضای درخت‌ها"""
    population = []
    if seed_tree is not None:
        population.append(seed_tree)
    while len(population) < pop_size:
        population.append(random_tree(5, features))
    
    best_mse = float('inf')
    best_tree = None
    
    for gen in range(generations):
        fitness = []
        for tree in population:
            try:
                pred = evaluate_tree_extended_v2(tree, X)
                if not np.all(np.isfinite(pred)):
                    continue
                mse = np.mean((y - pred)**2)
                fitness.append((mse + 0.001 * tree_depth(tree), mse, tree))
            except:
                pass
        
        if not fitness:
            break
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
# ۳. هسته تحلیل‌گر (با کتابخانه فیزیک)
# ═══════════════════════════════════════════════════════════
def analysis_core(X, y, features):
    """هسته ۴: تحلیل‌گر - بهترین قانون فیزیک را تشخیص می‌دهد"""
    match = detect_physics_law(X, y)
    if match:
        name, tree, corr, desc = match
        try:
            pred = evaluate_tree_extended_v2(tree, X)
            if np.all(np.isfinite(pred)):
                mse = np.mean((y - pred)**2)
                return tree, mse, corr, name
        except:
            pass
    return None, float('inf'), 0.0, None

# ═══════════════════════════════════════════════════════════
# ۴. موتور موازی
# ═══════════════════════════════════════════════════════════
def parallel_engine(X, y, features, generations=50, pop_size=300):
    """
    موتور موازی: GP و تحلیل‌گر همزمان اجرا می‌شوند.
    """
    # ── هسته ۴: تحلیل‌گر ──
    analysis_tree, analysis_mse, analysis_corr, analysis_name = analysis_core(X, y, features)
    
    # ── هسته GP: جستجوگر ──
    # GP با Seed Tree از تحلیل‌گر (اگر وجود داشت)
    gp_tree, gp_mse = gp_core(X, y, features, generations=generations, pop_size=pop_size, seed_tree=analysis_tree)
    
    # ── سنسور ترکیب: انتخاب بهترین ──
    if analysis_mse < gp_mse:
        return analysis_tree, analysis_mse, 'Analysis', analysis_name
    else:
        return gp_tree, gp_mse, 'GP', None

# ═══════════════════════════════════════════════════════════
# ۵. اجرا
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("═══ موتور موازی (Analysis + GP) ═══\n")
    
    equations = [
        {'name': 'I.8.14 (فاصله)', 'formula': lambda x: np.sqrt((x[:,0]-x[:,1])**2 + (x[:,2]-x[:,3])**2), 'n': 4, 'desc': 'فاصله'},
        {'name': 'I.9.18 (گرانش)', 'formula': lambda x: (x[:,0]*x[:,1])/((x[:,2]-x[:,3])**2 + 1e-6), 'n': 4, 'desc': 'گرانش'},
        {'name': 'II.2.42 (کولن)', 'formula': lambda x: x[:,0]/(x[:,1]**2 + 1e-6), 'n': 2, 'desc': 'کولن'},
        {'name': 'I.6.20 (گاوسی)', 'formula': lambda x: np.exp(-(x[:,0]**2)/(2*x[:,1]**2))/(np.sqrt(2*np.pi)*x[:,1]), 'n': 2, 'desc': 'گاوسی'},
        {'name': 'I.12.1 (اصطکاک)', 'formula': lambda x: x[:,0]*x[:,1]*x[:,2], 'n': 3, 'desc': 'اصطکاک'},
        {'name': 'I.13.4 (جنبشی)', 'formula': lambda x: 0.5*x[:,0]*(x[:,1]**2 + x[:,2]**2), 'n': 3, 'desc': 'جنبشی'},
    ]
    
    results = []
    for eq in equations:
        np.random.seed(42)
        X = np.random.uniform(0.1, 2.0, (300, eq['n']))
        y = eq['formula'](X)
        X_norm = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-9)
        
        features = [f'x{i}' for i in range(eq['n'])]
        
        best_tree, best_mse, winner, law_name = parallel_engine(X_norm, y, features, generations=50, pop_size=300)
        
        if best_tree:
            pred = evaluate_tree_extended_v2(best_tree, X_norm)
            if np.all(np.isfinite(pred)):
                mse_orig = np.mean((y - pred)**2)
                r2 = 1 - mse_orig / np.var(y)
            else:
                r2 = -float('inf')
        else:
            r2 = -float('inf')
        
        results.append((eq['name'], r2, winner, law_name))
        print(f"{eq['name']:>18}: R² = {r2:>7.4f}  |  برنده: {winner:>10}  |  قانون: {law_name}")
    
    print("\n═══ گزارش نهایی ═══")
    r2_values = [r[1] for r in results if np.isfinite(r[1])]
    print(f"میانگین R²: {np.mean(r2_values):.4f}")
    
    analysis_wins = sum(1 for r in results if r[2] == 'Analysis')
    gp_wins = sum(1 for r in results if r[2] == 'GP')
    print(f"پیروزی‌های تحلیل‌گر: {analysis_wins}")
    print(f"پیروزی‌های GP: {gp_wins}")
