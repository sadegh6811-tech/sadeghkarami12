"""موتور ژنتیک بازار - با گره ثابت (Constant Node) و Synthetic Sensor."""
import sys
import numpy as np
import random
import sympy as sp

def evaluate_tree(tree, X):
    if isinstance(tree, str):
        return X[:, int(tree[1:])]
    op = tree[0]
    if op == 'const':
        return np.full(X.shape[0], tree[1])
    if op == 'add':    return evaluate_tree(tree[1], X) + evaluate_tree(tree[2], X)
    if op == 'sub':    return evaluate_tree(tree[1], X) - evaluate_tree(tree[2], X)
    if op == 'mul':    return evaluate_tree(tree[1], X) * evaluate_tree(tree[2], X)
    if op == 'div':    return evaluate_tree(tree[1], X) / (evaluate_tree(tree[2], X) + 1e-6)
    if op == 'square': return evaluate_tree(tree[1], X) ** 2
    if op == 'sqrt':   return np.sqrt(np.abs(evaluate_tree(tree[1], X)) + 1e-6)
    raise ValueError(f"Unknown op: {op}")

def tree_to_sympy(tree, syms):
    if isinstance(tree, str):
        return syms[int(tree[1:])]
    op = tree[0]
    if op == 'const':
        return sp.Float(tree[1])
    if op == 'add':    return tree_to_sympy(tree[1], syms) + tree_to_sympy(tree[2], syms)
    if op == 'sub':    return tree_to_sympy(tree[1], syms) - tree_to_sympy(tree[2], syms)
    if op == 'mul':    return tree_to_sympy(tree[1], syms) * tree_to_sympy(tree[2], syms)
    if op == 'div':    return tree_to_sympy(tree[1], syms) / tree_to_sympy(tree[2], syms)
    if op == 'square': return tree_to_sympy(tree[1], syms) ** 2
    if op == 'sqrt':   return sp.sqrt(sp.Abs(tree_to_sympy(tree[1], syms)))

def tree_depth(tree):
    if isinstance(tree, str) or tree[0] == 'const': return 0
    return 1 + max(tree_depth(c) for c in tree[1:])

def tree_size(tree):
    if isinstance(tree, str) or tree[0] == 'const': return 1
    return 1 + sum(tree_size(c) for c in tree[1:])

def get_vars(tree):
    if isinstance(tree, str): return {tree}
    if tree[0] == 'const': return set()
    result = set()
    for child in tree[1:]: result.update(get_vars(child))
    return result

# ═══════════════════════════════════════════════════════════
# گره ثابت: تصادفی از بین اعداد
# ═══════════════════════════════════════════════════════════
CONST_POOL = [-2.0, -1.0, -0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 2.0, 3.0]

def random_tree(depth, features):
    if depth <= 0:
        r = random.random()
        if r < 0.3:  # ۳۰٪ شانس گره ثابت
            return ('const', random.choice(CONST_POOL))
        return random.choice(features)
    op = random.choice(['add', 'sub', 'mul', 'mul', 'mul', 'div', 'square', 'sqrt'])
    if op in ['square', 'sqrt']:
        return (op, random_tree(depth-1, features))
    return (op, random_tree(depth-1, features), random_tree(depth-1, features))

def crossover(t1, t2):
    if isinstance(t1, str) or isinstance(t2, str):
        return random.choice([t1, t2])
    if t1[0] != t2[0]:
        return random.choice([t1, t2])
    if t1[0] == 'const':
        return random.choice([t1, t2])
    if t1[0] in ['square', 'sqrt']:
        return (t1[0], random.choice([t1[1], t2[1]]))
    return (t1[0], random.choice([t1[1], t2[1]]), random.choice([t1[2], t2[2]]))

def mutate(tree, features, prob=0.3):
    if random.random() < prob:
        return random_tree(2, features)
    if isinstance(tree, str):
        return tree
    if tree[0] == 'const':
        # گره ثابت را کمی تغییر بده
        return ('const', tree[1] + random.uniform(-0.2, 0.2))
    if tree[0] in ['square', 'sqrt']:
        return (tree[0], mutate(tree[1], features, prob))
    return (tree[0], mutate(tree[1], features, prob), mutate(tree[2], features, prob))


def seed_tree_linear(features):
    """Seed برای بازار: w0*x0 + w1*x1 + ... + wn*xn"""
    nodes = [('mul', ('const', 1.0), f) for f in features]
    while len(nodes) > 1:
        new_nodes = []
        for i in range(0, len(nodes) - 1, 2):
            new_nodes.append(('add', nodes[i], nodes[i+1]))
        if len(nodes) % 2 == 1:
            new_nodes.append(nodes[-1])
        nodes = new_nodes
    return nodes[0]


def seed_tree_linear(features):
    """Seed برای بازار: w0*x0 + w1*x1 + ... + wn*xn"""
    nodes = [('mul', ('const', 1.0), f) for f in features]
    while len(nodes) > 1:
        new_nodes = []
        for i in range(0, len(nodes) - 1, 2):
            new_nodes.append(('add', nodes[i], nodes[i+1]))
        if len(nodes) % 2 == 1:
            new_nodes.append(nodes[-1])
        nodes = new_nodes
    return nodes[0]

def genetic_programming(X, y, features, generations=200, pop_size=1000):
    print(f"[*] عملگرها: +, -, *, /, square, sqrt")
    print(f"[*] گره ثابت: {len(CONST_POOL)} عدد در دسترس")
    mandatory = set(features)
    print(f"[*] اجباری: همه {len(features)} متغیر")
    seed = seed_tree_linear(features)
    population = [seed] + [random_tree(6, features) for _ in range(pop_size - 1)]
    print(f"[*] Seed خطی اضافه شد")
    best_overall = None
    best_overall_mse = float('inf')
    
    for gen in range(generations):
        fitness = []
        for tree in population:
            try:
                pred = evaluate_tree(tree, X)
                if not np.all(np.isfinite(pred)): continue
                if tree_depth(tree) > 6:
                    continue
                mse = np.mean((y - pred) ** 2)
                # جریمه‌ی متغیرهای غایب
                used = get_vars(tree)
                missing = mandatory - used
                missing_penalty = len(missing) * 10.0
                # جریمه‌ی پیچیدگی
                complexity_penalty = 0.05 * tree_size(tree) + 0.005 * tree_depth(tree)
                score = mse + missing_penalty + complexity_penalty
                fitness.append((score, mse, tree))
            except:
                pass
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
        if gen % 30 == 0:
            print(f"  نسل {gen}: MSE={fitness[0][1]:.6f}, size={tree_size(fitness[0][2])}")
    return best_overall, best_overall_mse

def has_header(path):
    with open(path, 'r') as f:
        first = f.readline().strip()
    try:
        float(first.split(',')[0]); return False
    except ValueError: return True

def main():
    if len(sys.argv) < 2: return
    path = sys.argv[1]
    target_col = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    header = has_header(path)
    X = np.loadtxt(path, delimiter=',', skiprows=1 if header else 0)
    if target_col < 0: target_col = X.shape[1] + target_col
    y = X[:, target_col]
    X_feat = np.delete(X, target_col, axis=1)
    n_features = X_feat.shape[1]
    features = [f'x{i}' for i in range(n_features)]
    print(f"[*] داده: {X.shape[0]} نقطه، {X.shape[1]} ستون")
    print(f"[*] متغیرها: {n_features}")
    best_tree, best_mse = genetic_programming(X_feat, y, features)
    if best_tree:
        print(f"\n[+] بهترین MSE: {best_mse:.8f}")
        print(f"[+] اندازه درخت: {tree_size(best_tree)}, عمق: {tree_depth(best_tree)}")
        syms = sp.symbols(f"x0:{n_features}")
        expr = tree_to_sympy(best_tree, syms)
        print(f"[+] معادله: y = {expr}")
    else:
        print("[!] هیچ معادله‌ای پیدا نشد.")

if __name__ == "__main__":
    main()
