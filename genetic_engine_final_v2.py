"""موتور ژنتیک نهایی v2 - با جریمه‌های سخت‌گیرانه."""
import sys
import numpy as np
import random
import sympy as sp

def evaluate_node(node, X):
    if isinstance(node, str):
        return X[:, int(node[1:])]
    op = node[0]
    if op == 'abs':
        return np.abs(evaluate_node(node[1], X))
    elif op == 'max':
        return np.maximum(evaluate_node(node[1], X), evaluate_node(node[2], X))

def tree_to_sympy(node, syms):
    if isinstance(node, str):
        return syms[int(node[1:])]
    op = node[0]
    if op == 'abs':
        return sp.Abs(tree_to_sympy(node[1], syms))
    elif op == 'max':
        return sp.Max(tree_to_sympy(node[1], syms), tree_to_sympy(node[2], syms))

def get_vars_used(node):
    if isinstance(node, str):
        return [node]
    result = []
    for child in node[1:]:
        if isinstance(child, (tuple, str)):
            result.extend(get_vars_used(child))
    return result

def create_random_tree(depth, features):
    if depth == 0:
        return str(np.random.choice(features))
    op = random.choice(['abs', 'max'])
    if op == 'abs':
        return ('abs', create_random_tree(depth-1, features))
    return ('max', create_random_tree(depth-1, features), create_random_tree(depth-1, features))

def crossover(parent1, parent2):
    if isinstance(parent1, str) or isinstance(parent2, str):
        return random.choice([parent1, parent2])
    if parent1[0] != parent2[0]:
        return random.choice([parent1, parent2])
    if parent1[0] == 'abs':
        return ('abs', random.choice([parent1[1], parent2[1]]))
    return ('max', random.choice([parent1[1], parent2[1]]), random.choice([parent1[2], parent2[2]]))

def mutate(node, features):
    if isinstance(node, str):
        return str(np.random.choice(features))
    if random.random() < 0.3:  # افزایش نرخ جهش به ۳۰٪
        return create_random_tree(2, features)
    return node

def genetic_programming(X, y, features, generations=100, population_size=500):
    population = [create_random_tree(4, features) for _ in range(population_size)]
    
    for gen in range(generations):
        fitness = []
        for tree in population:
            try:
                pred = evaluate_node(tree, X)
                mse = np.mean((y - pred) ** 2)
                
                # جریمه ۱: استفاده نکردن از همه متغیرها
                vars_used = set(get_vars_used(tree))
                penalty_vars = (len(features) - len(vars_used)) * 5.0
                
                # جریمه ۲: استفاده نکردن از Abs
                has_abs = 'abs' in str(tree)
                penalty_abs = 0 if has_abs else 2.0
                
                # جریمه ۳: عمق کم درخت (باید حداقل ۳ سطح max داشته باشد)
                depth_max = str(tree).count('max')
                penalty_depth = max(0, 3 - depth_max) * 1.0
                
                score = mse + penalty_vars + penalty_abs + penalty_depth
                
                if np.isfinite(score):
                    fitness.append((score, tree, mse))
            except:
                pass
        
        if not fitness:
            break
        
        fitness.sort(key=lambda x: x[0])
        best = [f[1] for f in fitness[:population_size // 2]]
        
        new_population = best.copy()
        while len(new_population) < population_size:
            p1, p2 = random.choice(best), random.choice(best)
            child = crossover(p1, p2)
            child = mutate(child, features)
            new_population.append(child)
        
        population = new_population
        
        if gen % 20 == 0:
            print(f"  نسل {gen}: بهترین MSE = {fitness[0][2]:.4f}")
    
    return fitness[0][1], fitness[0][2]

def main():
    if len(sys.argv) < 2:
        print("Usage: python genetic_engine_final_v2.py <file.csv> <target_col>")
        return
    
    path = sys.argv[1]
    target_col = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    X = np.loadtxt(path, delimiter=',', skiprows=1)
    if target_col < 0: target_col = X.shape[1] + target_col
    y = X[:, target_col]
    X_feat = np.delete(X, target_col, axis=1)
    
    print(f"[*] داده: {X.shape[0]} نقطه، {X.shape[1]} ستون")
    print("[*] اجرای موتور ژنتیک v2 (با جریمه‌های سخت‌گیرانه)...")
    
    features = [f'x{i}' for i in range(X_feat.shape[1])]
    best_tree, best_mse = genetic_programming(X_feat, y, features)
    
    print(f"\n[✓] بهترین معادله (MSE={best_mse:.4f}):")
    print(f"    {best_tree}")
    
    n_feats = X_feat.shape[1]
    syms = sp.symbols(f"x0:{n_feats}")
    expr = tree_to_sympy(best_tree, syms)
    print(f"\n[✓] معادله ریاضی:")
    print(f"    y = {expr}")

if __name__ == "__main__":
    main()
