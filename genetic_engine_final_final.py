"""موتور ژنتیک نهایی - با get_vars_used اصلاح‌شده."""
import sys
import numpy as np
import random
import sympy as sp

def evaluate_node(node, X):
    if isinstance(node, str):
        return X[:, int(node[1:])]
    op = node[0]
    if op == 'abs': return np.abs(evaluate_node(node[1], X))
    return np.maximum(evaluate_node(node[1], X), evaluate_node(node[2], X))

def tree_to_sympy(node, syms):
    if isinstance(node, str): return syms[int(node[1:])]
    op = node[0]
    if op == 'abs': return sp.Abs(tree_to_sympy(node[1], syms))
    return sp.Max(tree_to_sympy(node[1], syms), tree_to_sympy(node[2], syms))

def get_vars_used(node):
    """نسخه اصلاح‌شده: استفاده از بازگشت امن."""
    if isinstance(node, str):
        return [node]
    if not isinstance(node, tuple):
        return []
    result = []
    for child in node[1:]:
        result.extend(get_vars_used(child))
    return result

def create_random_tree(depth, features):
    if depth == 0: return str(np.random.choice(features))
    op = random.choice(['abs', 'max'])
    if op == 'abs': return ('abs', create_random_tree(depth-1, features))
    return ('max', create_random_tree(depth-1, features), create_random_tree(depth-1, features))

def crossover(p1, p2):
    if isinstance(p1, str) or isinstance(p2, str): return random.choice([p1, p2])
    if p1[0] != p2[0]: return random.choice([p1, p2])
    if p1[0] == 'abs': return ('abs', random.choice([p1[1], p2[1]]))
    return ('max', random.choice([p1[1], p2[1]]), random.choice([p1[2], p2[2]]))

def mutate(node, features):
    if isinstance(node, str): return str(np.random.choice(features))
    if random.random() < 0.3: return create_random_tree(2, features)
    return node

def genetic_programming(X, y, features, generations=200, population_size=800):
    population = [create_random_tree(6, features) for _ in range(population_size)]
    best_overall_tree = None
    best_overall_mse = float('inf')
    
    for gen in range(generations):
        fitness = []
        for tree in population:
            try:
                vars_used = set(get_vars_used(tree))
                missing = set(features) - vars_used
                
                # جریمه خطی ساده (نه 1e10)
                if missing:
                    score = 100.0 + len(missing) * 20.0
                    mse = float('inf')
                else:
                    pred = evaluate_node(tree, X)
                    mse = np.mean((y - pred) ** 2)
                    depth = str(tree).count('max') + str(tree).count('abs')
                    score = mse - depth * 0.01
                    
                if np.isfinite(score):
                    fitness.append((score, tree, mse))
            except: pass
        
        if not fitness: break
        fitness.sort(key=lambda x: x[0])
        
        valid_trees = [f for f in fitness if f[2] != float('inf')]
        
        if valid_trees:
            if valid_trees[0][2] < best_overall_mse:
                best_overall_mse = valid_trees[0][2]
                best_overall_tree = valid_trees[0][1]
            best = [f[1] for f in valid_trees[:population_size // 2]]
        else:
            best = [f[1] for f in fitness[:population_size // 2]]
            
        new_population = best.copy()
        while len(new_population) < population_size:
            p1, p2 = random.choice(best), random.choice(best)
            child = mutate(crossover(p1, p2), features)
            new_population.append(child)
        
        population = new_population
        
        if gen % 20 == 0:
            status = f"MSE={best_overall_mse:.4f}" if best_overall_tree else "در جستجو..."
            print(f"  نسل {gen}: بهترین {status} (تعداد معتبر: {len(valid_trees)})")
    
    return best_overall_tree, best_overall_mse

def main():
    if len(sys.argv) < 2: return
    path, target_col = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else -1
    X = np.loadtxt(path, delimiter=',', skiprows=1)
    if target_col < 0: target_col = X.shape[1] + target_col
    y = X[:, target_col]
    X_feat = np.delete(X, target_col, axis=1)
    
    features = [f'x{i}' for i in range(X.shape[1])]
    print(f"[*] متغیرهای اجباری (همه): {features}")
    print(f"[*] اجرای موتور ژنتیک نهایی...")
    
    best_tree, best_mse = genetic_programming(X_feat, y, features)
    
    if best_tree:
        print(f"\n[✓] بهترین معادله (MSE={best_mse:.4f}):")
        print(f"    {best_tree}")
        n_feats = X_feat.shape[1]
        syms = sp.symbols(f"x0:{n_feats}")
        expr = tree_to_sympy(best_tree, syms)
        print(f"\n[✓] معادله ریاضی:")
        print(f"    y = {expr}")
    else:
        print("\n[!] هیچ درخت معتبری پیدا نشد.")

if __name__ == "__main__":
    main()
