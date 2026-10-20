"""موتور ژنتیک مخصوص هایپرکیوب (فقط abs و max)."""
import sys
import numpy as np
import random
import sympy as sp

def evaluate_node(node, X):
    if isinstance(node, str):
        idx = int(node[1:])
        return X[:, idx]
    op = node[0]
    if op == 'abs':
        return np.abs(evaluate_node(node[1], X))
    elif op == 'max':
        return np.maximum(evaluate_node(node[1], X), evaluate_node(node[2], X))

def tree_to_sympy(node, syms):
    if isinstance(node, str):
        idx = int(node[1:])
        return syms[idx]
    op = node[0]
    if op == 'abs':
        return sp.Abs(tree_to_sympy(node[1], syms))
    elif op == 'max':
        return sp.Max(tree_to_sympy(node[1], syms), tree_to_sympy(node[2], syms))

def create_random_tree(depth, features):
    if depth == 0:
        return str(np.random.choice(features))  # تبدیل به رشته استاندارد
    
    # فقط از abs و max استفاده کن
    op = random.choice(['abs', 'max'])
    if op == 'abs':
        return ('abs', create_random_tree(depth-1, features))
    elif op == 'max':
        left = create_random_tree(depth-1, features)
        right = create_random_tree(depth-1, features)
        return ('max', left, right)

def genetic_programming(X, y, features, generations=20, population_size=100):
    population = [create_random_tree(3, features) for _ in range(population_size)]
    
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
        
        new_population = best.copy()
        while len(new_population) < population_size:
            parent = random.choice(best)
            new_population.append(parent)
        population = new_population
    
    return fitness[0][1], fitness[0][0]

def main():
    if len(sys.argv) < 2:
        print("Usage: python genetic_engine_cube.py <file.csv> <target_col>")
        return
    
    path = sys.argv[1]
    target_col = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    
    X = np.loadtxt(path, delimiter=',', skiprows=1)
    if target_col < 0:
        target_col = X.shape[1] + target_col
    
    y = X[:, target_col]
    X_feat = np.delete(X, target_col, axis=1)
    
    print(f"[*] داده: {X.shape[0]} نقطه، {X.shape[1]} ستون")
    print("[*] اجرای موتور ژنتیک (فقط abs و max)...")
    
    features = [f'x{i}' for i in range(X_feat.shape[1])]
    best_tree, best_mse = genetic_programming(X_feat, y, features, generations=20, population_size=100)
    
    print(f"\n[✓] بهترین معادله (MSE={best_mse:.4f}):")
    print(f"    {best_tree}")
    
    # تبدیل به sympy
    n_feats = X_feat.shape[1]
    syms = sp.symbols(f"x0:{n_feats}")
    expr = tree_to_sympy(best_tree, syms)
    print(f"\n[✓] معادله ریاضی:")
    print(f"    y = {expr}")

if __name__ == "__main__":
    main()
