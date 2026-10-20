"""موتور ژنتیک کور - بدون Seed Tree و بدون اجبار متغیرها."""
import sys
import numpy as np
import random
import sympy as sp

def evaluate_tree(tree, X):
    if isinstance(tree, str):
        return X[:, int(tree[1:])]
    op = tree[0]
    if op == 'abs':
        return np.abs(evaluate_tree(tree[1], X))
    return np.maximum(evaluate_tree(tree[1], X), evaluate_tree(tree[2], X))

def tree_to_sympy(tree, syms):
    if isinstance(tree, str):
        return syms[int(tree[1:])]
    op = tree[0]
    if op == 'abs':
        return sp.Abs(tree_to_sympy(tree[1], syms))
    left = tree_to_sympy(tree[1], syms)
    right = tree_to_sympy(tree[2], syms)
    args = []
    for side in (left, right):
        if isinstance(side, sp.Max):
            args.extend(side.args)
        else:
            args.append(side)
    return sp.Max(*args)

def tree_depth(tree):
    if isinstance(tree, str): return 0
    return 1 + max(tree_depth(c) for c in tree[1:])

def random_tree(depth, features):
    if depth <= 0:
        return random.choice(features)
    op = random.choice(['abs', 'max', 'max', 'max'])
    if op == 'abs':
        return ('abs', random_tree(depth-1, features))
    return ('max', random_tree(depth-1, features),
                   random_tree(depth-1, features))

def crossover(t1, t2):
    if isinstance(t1, str) or isinstance(t2, str):
        return random.choice([t1, t2])
    if t1[0] != t2[0]:
        return random.choice([t1, t2])
    if t1[0] == 'abs':
        return ('abs', random.choice([t1[1], t2[1]]))
    return ('max', random.choice([t1[1], t2[1]]), random.choice([t1[2], t2[2]]))

def mutate(tree, features, prob=0.2):
    if random.random() < prob:
        return random_tree(2, features)
    if isinstance(tree, str):
        return tree
    if tree[0] == 'abs':
        return ('abs', mutate(tree[1], features, prob))
    return ('max', mutate(tree[1], features, prob),
                   mutate(tree[2], features, prob))

def genetic_programming(X, y, features, generations=50, pop_size=200):
    print(f"[BLIND] هیچ متغیری اجباری نیست")
    print(f"[BLIND] جمعیت اولیه: {pop_size} درخت تصادفی")
    population = [random_tree(6, features) for _ in range(pop_size)]
    best_overall = None
    best_overall_mse = float('inf')
    
    for gen in range(generations):
        fitness = []
        for tree in population:
            try:
                pred = evaluate_tree(tree, X)
                if not np.all(np.isfinite(pred)): continue
                mse = np.mean((y - pred) ** 2)
                score = mse + tree_depth(tree) * 0.001
                fitness.append((score, mse, tree))
            except: pass
        if not fitness: break
        fitness.sort(key=lambda x: x[0])
        if fitness[0][1] < best_overall_mse:
            best_overall_mse = fitness[0][1]
            best_overall = fitness[0][2]
        top = [f[2] for f in fitness[:pop_size // 2]]
        new_pop = top.copy()
        while len(new_pop) < pop_size:
            p1, p2 = random.choice(top), random.choice(top)
            new_pop.append(mutate(crossover(p1, p2), features))
        population = new_pop
        if gen % 10 == 0:
            print(f"  نسل {gen}: بهترین MSE = {fitness[0][1]:.8f}")
    return best_overall, best_overall_mse

def has_header(path):
    with open(path, 'r') as f:
        first = f.readline().strip()
    try:
        float(first.split(',')[0]); return False
    except ValueError: return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python genetic_engine_blind.py <file.csv> [target_col]")
        return
    path = sys.argv[1]
    target_col = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    header = has_header(path)
    X = np.loadtxt(path, delimiter=',', skiprows=1 if header else 0)
    if target_col < 0: target_col = X.shape[1] + target_col
    y = X[:, target_col]
    X_feat = np.delete(X, target_col, axis=1)
    n_features = X_feat.shape[1]
    features = [f'x{i}' for i in range(n_features)]
    print(f"[BLIND] داده: {X.shape[0]} نقطه، {X.shape[1]} ستون")
    print(f"[BLIND] متغیرها: {n_features}")
    best_tree, best_mse = genetic_programming(X_feat, y, features)
    if best_tree:
        print(f"\n[BLIND] بهترین MSE: {best_mse:.8f}")
        print(f"[BLIND] عمق درخت: {tree_depth(best_tree)}")
        syms = sp.symbols(f"x0:{n_features}")
        expr = tree_to_sympy(best_tree, syms)
        print(f"[BLIND] معادله: y = {expr}")
    else:
        print("[BLIND] هیچ معادله‌ای پیدا نشد.")

if __name__ == "__main__":
    main()
