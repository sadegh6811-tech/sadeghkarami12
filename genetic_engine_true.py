"""موتور ژنتیک واقعی - با اصلاحات: همه متغیرها اجباری + جریمه بالا + Seed."""
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
    if op == 'sin':
        return np.sin(evaluate_tree(tree[1], X))
    if op == 'cos':
        return np.cos(evaluate_tree(tree[1], X))
    return np.maximum(evaluate_tree(tree[1], X), evaluate_tree(tree[2], X))

def tree_to_sympy(tree, syms):
    if isinstance(tree, str):
        return syms[int(tree[1:])]
    op = tree[0]
    if op == 'abs':
        inner = tree_to_sympy(tree[1], syms)
        if isinstance(inner, sp.Max):
            return inner
        return sp.Abs(inner)
    left = tree_to_sympy(tree[1], syms)
    right = tree_to_sympy(tree[2], syms)
    args = []
    for side in (left, right):
        if isinstance(side, sp.Max):
            args.extend(side.args)
        else:
            args.append(side)
    return sp.Max(*args)

def get_vars(tree):
    if isinstance(tree, str):
        return {tree}
    result = set()
    for child in tree[1:]:
        result.update(get_vars(child))
    return result

def tree_depth(tree):
    if isinstance(tree, str):
        return 0
    return 1 + max(tree_depth(c) for c in tree[1:])

def seed_tree_with_all(features):
    if len(features) == 0:
        return None
    if len(features) == 1:
        return ('abs', features[0])
    nodes = [('abs', f) for f in features]
    while len(nodes) > 1:
        new_nodes = []
        for i in range(0, len(nodes) - 1, 2):
            new_nodes.append(('max', nodes[i], nodes[i+1]))
        if len(nodes) % 2 == 1:
            new_nodes.append(nodes[-1])
        nodes = new_nodes
    return nodes[0]

def random_tree(depth, features, mandatory):
    if depth <= 0:
        if random.random() < 0.8 and mandatory:
            return random.choice(list(mandatory))
        return random.choice(features)
    op = random.choice(['abs', 'sin', 'cos', 'max', 'max'])
    if op == 'abs':
        return ('abs', random_tree(depth-1, features, mandatory))
    if op == 'sin':
        return ('sin', random_tree(depth-1, features, mandatory))
    if op == 'cos':
        return ('cos', random_tree(depth-1, features, mandatory))
    return ('max', random_tree(depth-1, features, mandatory),
                   random_tree(depth-1, features, mandatory))

def crossover(t1, t2):
    if isinstance(t1, str) or isinstance(t2, str):
        return random.choice([t1, t2])
    if t1[0] != t2[0]:
        return random.choice([t1, t2])
    if t1[0] in ['abs', 'sin', 'cos']:
        return (t1[0], random.choice([t1[1], t2[1]]))
    return ('max', random.choice([t1[1], t2[1]]), random.choice([t1[2], t2[2]]))

def mutate(tree, features, mandatory, prob=0.2):
    if random.random() < prob:
        return random_tree(2, features, mandatory)
    if isinstance(tree, str):
        return tree
    if tree[0] in ['abs', 'sin', 'cos']:
        return (tree[0], mutate(tree[1], features, mandatory, prob))
    return ('max', mutate(tree[1], features, mandatory, prob),
                   mutate(tree[2], features, mandatory, prob))

def genetic_programming(X, y, features, generations=50, pop_size=200):
    # اصلاح ۱: همه متغیرها اجباری
    mandatory = set(features)
    mandatory_list = list(mandatory)
    print(f"[*] اجباری (نهایی): همه {len(features)} متغیر")
    
    # اصلاح ۲: y_scale و جریمه
    y_scale = np.mean(y**2) + 1e-6
    PENALTY_WEIGHT = 10.0
    
    # اصلاح ۳: Seed در جمعیت اولیه
    seed = seed_tree_with_all(features)
    population = []
    if seed is not None:
        population.append(seed)
    while len(population) < pop_size:
        population.append(random_tree(6, features, mandatory_list))
    
    best_overall = None
    best_overall_mse = float('inf')
    
    for gen in range(generations):
        fitness = []
        for tree in population:
            try:
                pred = evaluate_tree(tree, X)
                if not np.all(np.isfinite(pred)):
                    continue
                mse = np.mean((y - pred) ** 2)
                used = get_vars(tree)
                missing = mandatory - used
                if missing:
                    score = mse + PENALTY_WEIGHT * len(missing) * y_scale
                else:
                    score = mse + tree_depth(tree) * 0.001
                fitness.append((score, mse, tree, len(missing)))
            except:
                pass
        
        if not fitness:
            print(f"  نسل {gen}: هیچ درختی معتبر نبود")
            break
        
        fitness.sort(key=lambda x: x[0])
        valid = [f for f in fitness if f[3] == 0]
        
        if valid:
            if valid[0][1] < best_overall_mse:
                best_overall_mse = valid[0][1]
                best_overall = valid[0][2]
            top = [f[2] for f in valid[:pop_size // 2]]
        else:
            top = [f[2] for f in fitness[:pop_size // 2]]
        
        # اصلاح ۴: حفظ Seed در top
        if seed is not None and seed not in top:
            top[-1] = seed
        
        new_pop = top.copy()
        while len(new_pop) < pop_size:
            p1, p2 = random.choice(top), random.choice(top)
            child = mutate(crossover(p1, p2), features, mandatory_list)
            new_pop.append(child)
        population = new_pop
        
        if gen % 10 == 0:
            status = f"MSE={best_overall_mse:.6f}" if best_overall else "در جستجو..."
            print(f"  نسل {gen}: {status} (کامل: {len(valid)}، بهترین جریمه: {fitness[0][3]})")
    
    return best_overall, best_overall_mse

def has_header(path):
    with open(path, 'r') as f:
        first = f.readline().strip()
    try:
        float(first.split(',')[0])
        return False
    except ValueError:
        return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python genetic_engine_true.py <file.csv> [target_col]")
        return
    path = sys.argv[1]
    target_col = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    
    header = has_header(path)
    X = np.loadtxt(path, delimiter=',', skiprows=1 if header else 0)
    print(f"[*] هدر: {'دارد' if header else 'ندارد'}")
    
    if target_col < 0:
        target_col = X.shape[1] + target_col
    
    y = X[:, target_col]
    X_feat = np.delete(X, target_col, axis=1)
    n_features = X_feat.shape[1]
    features = [f'x{i}' for i in range(n_features)]
    
    print(f"[*] داده: {X.shape[0]} نقطه، {X.shape[1]} ستون")
    print(f"[*] متغیرها: {n_features} ({features})")
    print(f"[*] هدف: ستون {target_col}")
    
    if n_features == 0:
        print("[!] هیچ متغیر مستقلی وجود ندارد.")
        return
    
    best_tree, best_mse = genetic_programming(X_feat, y, features)
    
    if best_tree:
        print(f"\n[+] بهترین معادله (MSE={best_mse:.8f}):")
        print(f"    عمق درخت: {tree_depth(best_tree)}")
        syms = sp.symbols(f"x0:{n_features}")
        expr = tree_to_sympy(best_tree, syms)
        print(f"\n[+] معادله ریاضی:")
        print(f"    y = {expr}")
    else:
        print("\n[!] هیچ معادله کاملی پیدا نشد.")

if __name__ == "__main__":
    main()
