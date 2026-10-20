"""موتور ترکیبی: برنامه‌نویسی ژنتیک + توزیع‌های گسسته (نسخه اصلاح‌شده)."""
import sys
import numpy as np
import random
import sympy as sp

# ═══════════════════════════════════════════════════════════
# ۱. توزیع‌های گسسته (Discrete Distributions)
# ═══════════════════════════════════════════════════════════
def discrete_distribution_fit(X, y):
    """تخمین احتمال حضور هر ویژگی با توزیع برنولی."""
    n, d = X.shape
    probs = np.zeros(d)
    
    for i in range(d):
        if np.std(X[:, i]) > 1e-6:
            corr = np.corrcoef(X[:, i], y)[0, 1]
            probs[i] = abs(corr) if not np.isnan(corr) else 0.0
    
    # نرمال‌سازی به احتمال (جمع = 1)
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(d) / d
    
    return probs

# ═══════════════════════════════════════════════════════════
# ۲. توابع پایه
# ═══════════════════════════════════════════════════════════
def evaluate_node(node, X):
    """ارزیابی یک گره از درخت."""
    if isinstance(node, str):
        # متغیر x0, x1, ...
        if node.startswith('x'):
            idx = int(node[1:])
            return X[:, idx]
        return None
    op = node[0]
    if op == 'abs':
        return np.abs(evaluate_node(node[1], X))
    elif op == 'max':
        return np.maximum(evaluate_node(node[1], X), evaluate_node(node[2], X))
    elif op == '+':
        return evaluate_node(node[1], X) + evaluate_node(node[2], X)
    elif op == '-':
        return evaluate_node(node[1], X) - evaluate_node(node[2], X)
    elif op == '*':
        return evaluate_node(node[1], X) * evaluate_node(node[2], X)
    elif op == '/':
        denom = evaluate_node(node[2], X)
        denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
        return evaluate_node(node[1], X) / denom

def tree_to_sympy(node, syms):
    """تبدیل درخت به عبارت sympy."""
    if isinstance(node, str):
        idx = int(node[1:])
        return syms[idx]
    op = node[0]
    if op == 'abs':
        return sp.Abs(tree_to_sympy(node[1], syms))
    elif op == 'max':
        return sp.Max(tree_to_sympy(node[1], syms), tree_to_sympy(node[2], syms))
    elif op == '+':
        return tree_to_sympy(node[1], syms) + tree_to_sympy(node[2], syms)
    elif op == '-':
        return tree_to_sympy(node[1], syms) - tree_to_sympy(node[2], syms)
    elif op == '*':
        return tree_to_sympy(node[1], syms) * tree_to_sympy(node[2], syms)
    elif op == '/':
        return tree_to_sympy(node[1], syms) / tree_to_sympy(node[2], syms)

# ═══════════════════════════════════════════════════════════
# ۳. موتور ژنتیک
# ═══════════════════════════════════════════════════════════
def create_random_tree(depth, features, probs):
    """ایجاد درخت با استفاده از توزیع گسسته برای انتخاب متغیر."""
    if depth == 0:
        # انتخاب متغیر بر اساس توزیع
        return np.random.choice(features, p=probs)
    
    op = random.choice(['+', '-', '*', 'abs', 'max'])
    if op in ['+', '-', '*']:
        left = create_random_tree(depth-1, features, probs)
        right = create_random_tree(depth-1, features, probs)
        return (op, left, right)
    elif op == 'abs':
        child = create_random_tree(depth-1, features, probs)
        return ('abs', child)
    elif op == 'max':
        left = create_random_tree(depth-1, features, probs)
        right = create_random_tree(depth-1, features, probs)
        return ('max', left, right)

def genetic_programming(X, y, features, probs, generations=30, population_size=100):
    """اجرای برنامه‌نویسی ژنتیک."""
    population = [create_random_tree(3, features, probs) for _ in range(population_size)]
    
    for gen in range(generations):
        fitness = []
        for tree in population:
            try:
                pred = evaluate_node(tree, X)
                mse = np.mean((y - pred) ** 2)
                if np.isfinite(mse):
                    fitness.append((mse, tree))
            except:
                pass
        
        if not fitness:
            break
        
        fitness.sort(key=lambda x: x[0])
        best = [f[1] for f in fitness[:population_size // 2]]
        
        # تولید نسل بعدی
        new_population = best.copy()
        while len(new_population) < population_size:
            parent = random.choice(best)
            new_population.append(parent)
        population = new_population
    
    return fitness[0][1], fitness[0][0]

# ═══════════════════════════════════════════════════════════
# ۴. سنسور
# ═══════════════════════════════════════════════════════════
def hybrid_sensor(X, y):
    max_vals = np.max(np.abs(X), axis=1)
    boundary_ratio = np.mean(max_vals > 0.9)
    return "discrete" if boundary_ratio > 0.3 else "continuous"

# ═══════════════════════════════════════════════════════════
# ۵. اجرا
# ═══════════════════════════════════════════════════════════
def main():
    if len(sys.argv) < 2:
        print("Usage: python genetic_engine.py <file.csv> <target_col>")
        return
    
    path = sys.argv[1]
    target_col = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    
    X = np.loadtxt(path, delimiter=',', skiprows=1)
    if target_col < 0:
        target_col = X.shape[1] + target_col
    
    y = X[:, target_col]
    X_feat = np.delete(X, target_col, axis=1)
    
    print(f"[*] داده: {X.shape[0]} نقطه، {X.shape[1]} ستون")
    
    method = hybrid_sensor(X_feat, y)
    print(f"[*] سنسور: روش {method}")
    
    if method == "discrete":
        print("[*] اجرای برنامه‌نویسی ژنتیک...")
        features = [f'x{i}' for i in range(X_feat.shape[1])]
        probs = discrete_distribution_fit(X_feat, y)
        
        best_tree, best_mse = genetic_programming(X_feat, y, features, probs, generations=20, population_size=50)
        
        print(f"\n[✓] بهترین معادله (MSE={best_mse:.4f}):")
        print(f"    {best_tree}")
        
        # تبدیل به sympy
        n_feats = X_feat.shape[1]
        syms = sp.symbols(f"x0:{n_feats}")
        expr = tree_to_sympy(best_tree, syms)
        print(f"\n[✓] معادله ریاضی:")
        print(f"    y = {expr}")
    else:
        print("[*] داده پیوسته است. از SVD استفاده کنید.")

if __name__ == "__main__":
    main()
