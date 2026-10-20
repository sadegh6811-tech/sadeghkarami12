"""
موتور یکپارچه MathBot - ترکیب GP با کتابخانه‌ی کامل توابع
شامل: عملگرهای کامل، مدیریت دامنه، Crossover و Mutation کامل، Seed Tree
"""
import sys
import numpy as np
import random
import sympy as sp

# ═══════════════════════════════════════════════════════════
# ۱. توابع پایه (با مدیریت دامنه)
# ═══════════════════════════════════════════════════════════
def safe_div(a, b):
    return a / (b + 1e-9 * np.sign(b + 1e-30))

def safe_log(x):
    return np.log(np.abs(x) + 1e-9)

def safe_sqrt(x):
    return np.sqrt(np.abs(x) + 1e-9)

def safe_exp(x):
    return np.exp(np.clip(x, -50, 50))

# ═══════════════════════════════════════════════════════════
# ۲. ارزیابی درخت (evaluate_tree)
# ═══════════════════════════════════════════════════════════
def evaluate_tree(tree, X):
    if isinstance(tree, str):
        return X[:, int(tree[1:])]
    if isinstance(tree, tuple) and tree[0] == 'const':
        return np.full(X.shape[0], tree[1])
    
    op = tree[0]
    if op == 'abs': return np.abs(evaluate_tree(tree[1], X))
    if op == 'add': return evaluate_tree(tree[1], X) + evaluate_tree(tree[2], X)
    if op == 'sub': return evaluate_tree(tree[1], X) - evaluate_tree(tree[2], X)
    if op == 'mul': return evaluate_tree(tree[1], X) * evaluate_tree(tree[2], X)
    if op == 'div': return safe_div(evaluate_tree(tree[1], X), evaluate_tree(tree[2], X))
    if op == 'sin': return np.sin(evaluate_tree(tree[1], X))
    if op == 'cos': return np.cos(evaluate_tree(tree[1], X))
    if op == 'exp': return safe_exp(evaluate_tree(tree[1], X))
    if op == 'log': return safe_log(evaluate_tree(tree[1], X))
    if op == 'sqrt': return safe_sqrt(evaluate_tree(tree[1], X))
    if op == 'pow': 
        base = evaluate_tree(tree[1], X)
        exponent = evaluate_tree(tree[2], X)
        return np.sign(base) * np.abs(base) ** np.clip(exponent, -5, 5)
    if op == 'max': return np.maximum(evaluate_tree(tree[1], X), evaluate_tree(tree[2], X))
    if op == 'min': return np.minimum(evaluate_tree(tree[1], X), evaluate_tree(tree[2], X))
    return np.zeros(X.shape[0])

# ═══════════════════════════════════════════════════════════
# ۳. تبدیل به sympy
# ═══════════════════════════════════════════════════════════
def tree_to_sympy(tree, syms):
    if isinstance(tree, str):
        return syms[int(tree[1:])]
    if isinstance(tree, tuple) and tree[0] == 'const':
        return sp.Float(tree[1])
    
    op = tree[0]
    if op == 'abs': return sp.Abs(tree_to_sympy(tree[1], syms))
    if op == 'add': return tree_to_sympy(tree[1], syms) + tree_to_sympy(tree[2], syms)
    if op == 'sub': return tree_to_sympy(tree[1], syms) - tree_to_sympy(tree[2], syms)
    if op == 'mul': return tree_to_sympy(tree[1], syms) * tree_to_sympy(tree[2], syms)
    if op == 'div': return tree_to_sympy(tree[1], syms) / tree_to_sympy(tree[2], syms)
    if op == 'sin': return sp.sin(tree_to_sympy(tree[1], syms))
    if op == 'cos': return sp.cos(tree_to_sympy(tree[1], syms))
    if op == 'exp': return sp.exp(tree_to_sympy(tree[1], syms))
    if op == 'log': return sp.log(sp.Abs(tree_to_sympy(tree[1], syms)) + 1e-9)
    if op == 'sqrt': return sp.sqrt(sp.Abs(tree_to_sympy(tree[1], syms)))
    if op == 'pow': return tree_to_sympy(tree[1], syms) ** tree_to_sympy(tree[2], syms)
    if op == 'max': return sp.Max(tree_to_sympy(tree[1], syms), tree_to_sympy(tree[2], syms))
    if op == 'min': return sp.Min(tree_to_sympy(tree[1], syms), tree_to_sympy(tree[2], syms))
    return sp.Integer(0)

# ═══════════════════════════════════════════════════════════
# ۴. تولید درخت تصادفی (random_tree)
# ═══════════════════════════════════════════════════════════
OPERATORS_UNARY = ['abs', 'sin', 'cos', 'exp', 'log', 'sqrt']
OPERATORS_BINARY = ['add', 'sub', 'mul', 'div', 'pow', 'max', 'min']

def random_tree(depth, features, force_operators=None):
    if depth <= 0:
        if random.random() < 0.3:
            return ('const', random.choice([-1.0, -0.5, 0.0, 0.5, 1.0]))
        return random.choice(features)
    
    ops = (OPERATORS_UNARY + OPERATORS_BINARY) if force_operators is None else force_operators
    op = random.choice(ops)
    
    if op in OPERATORS_UNARY:
        return (op, random_tree(depth-1, features, force_operators))
    return (op, random_tree(depth-1, features, force_operators), 
                random_tree(depth-1, features, force_operators))

# ═══════════════════════════════════════════════════════════
# ۵. Crossover و Mutation
# ═══════════════════════════════════════════════════════════
def tree_depth(tree):
    if isinstance(tree, str) or (isinstance(tree, tuple) and tree[0] == 'const'):
        return 0
    return 1 + max(tree_depth(c) for c in tree[1:])

def crossover(t1, t2):
    if isinstance(t1, str) or isinstance(t2, str):
        return random.choice([t1, t2])
    if t1[0] == 'const' or t2[0] == 'const':
        return random.choice([t1, t2])
    if t1[0] != t2[0]:
        return random.choice([t1, t2])
    if t1[0] in OPERATORS_UNARY:
        return (t1[0], random.choice([t1[1], t2[1]]))
    return (t1[0], random.choice([t1[1], t2[1]]), random.choice([t1[2], t2[2]]))

def mutate(tree, features, prob=0.2):
    if random.random() < prob:
        return random_tree(3, features)
    if isinstance(tree, str) or (isinstance(tree, tuple) and tree[0] == 'const'):
        return tree
    if tree[0] in OPERATORS_UNARY:
        return (tree[0], mutate(tree[1], features, prob))
    return (tree[0], mutate(tree[1], features, prob), mutate(tree[2], features, prob))

# ═══════════════════════════════════════════════════════════
# ۶. Seed Tree (از هایپرکیوب و کره)
# ═══════════════════════════════════════════════════════════
def seed_tree_hypercube(features):
    """درخت اولیه: Max(|x0|, |x1|, ..., |xn|)"""
    nodes = [('abs', f) for f in features]
    while len(nodes) > 1:
        new_nodes = []
        for i in range(0, len(nodes) - 1, 2):
            new_nodes.append(('max', nodes[i], nodes[i+1]))
        if len(nodes) % 2 == 1:
            new_nodes.append(nodes[-1])
        nodes = new_nodes
    return nodes[0]

def seed_tree_sphere(features):
    """درخت اولیه: x0² + x1² + ... + xn²"""
    nodes = [('mul', f, f) for f in features]
    while len(nodes) > 1:
        new_nodes = []
        for i in range(0, len(nodes) - 1, 2):
            new_nodes.append(('add', nodes[i], nodes[i+1]))
        if len(nodes) % 2 == 1:
            new_nodes.append(nodes[-1])
        nodes = new_nodes
    return nodes[0]

# ═══════════════════════════════════════════════════════════
# ۷. موتور ژنتیک (رگرسیون و طبقه‌بندی)
# ═══════════════════════════════════════════════════════════
def genetic_programming(X, y, features, mode='regression', 
                        generations=100, pop_size=500, seed_mode=None):
    print(f"[*] Unified GP - حالت: {mode}")
    
    # ساخت جمعیت اولیه
    population = []
    if seed_mode == 'hypercube':
        population.append(seed_tree_hypercube(features))
    elif seed_mode == 'sphere':
        population.append(seed_tree_sphere(features))
    while len(population) < pop_size:
        population.append(random_tree(5, features))
    
    best_overall = None
    best_score = float('inf') if mode == 'regression' else 0.0
    
    for gen in range(generations):
        fitness = []
        for tree in population:
            try:
                pred = evaluate_tree(tree, X)
                if not np.all(np.isfinite(pred)):
                    continue
                
                if mode == 'regression':
                    mse = np.mean((y - pred) ** 2)
                    score = mse + 0.01 * tree_depth(tree)
                    fitness.append((score, mse, tree))
                else:  # classification
                    pred_class = np.zeros_like(pred)
                    pred_class[pred > 0.1] = 1
                    pred_class[pred < -0.1] = -1
                    acc = np.mean(pred_class == y)
                    score = acc - 0.001 * tree_depth(tree)
                    fitness.append((score, acc, tree))
            except:
                pass
        
        if not fitness:
            break
        
        if mode == 'regression':
            fitness.sort(key=lambda x: x[0])
            if fitness[0][1] < best_score:
                best_score = fitness[0][1]
                best_overall = fitness[0][2]
            top = [f[2] for f in fitness[:pop_size // 2]]
        else:
            fitness.sort(key=lambda x: x[0], reverse=True)
            if fitness[0][1] > best_score:
                best_score = fitness[0][1]
                best_overall = fitness[0][2]
            top = [f[2] for f in fitness[:pop_size // 2]]
        
        new_pop = top.copy()
        while len(new_pop) < pop_size:
            p1, p2 = random.choice(top), random.choice(top)
            child = mutate(crossover(p1, p2), features)
            new_pop.append(child)
        population = new_pop
        
        if gen % 20 == 0:
            print(f"  نسل {gen}: بهترین {'MSE' if mode=='regression' else 'دقت'} = {best_score:.6f}")
    
    return best_overall, best_score

# ═══════════════════════════════════════════════════════════
# ۸. اجرا
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python unified_engine.py <file.csv> [target_col] [mode] [seed_mode]")
        print("  mode: regression | classification")
        print("  seed_mode: none | hypercube | sphere")
        sys.exit(1)
    
    path = sys.argv[1]
    target_col = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    mode = sys.argv[3] if len(sys.argv) > 3 else 'regression'
    seed_mode = sys.argv[4] if len(sys.argv) > 4 else None
    
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    if target_col < 0:
        target_col = data.shape[1] + target_col
    
    y = data[:, target_col]
    X = np.delete(data, target_col, axis=1)
    features = [f'x{i}' for i in range(X.shape[1])]
    
    print(f"[*] داده: {X.shape[0]} نقطه، {X.shape[1]} بعد")
    print(f"[*] هدف: ستون {target_col}")
    print(f"[*] حالت: {mode}")
    print(f"[*] Seed: {seed_mode if seed_mode else 'تصادفی'}")
    
    best_tree, best_score = genetic_programming(
        X, y, features, mode=mode, generations=100, pop_size=500, seed_mode=seed_mode
    )
    
    if best_tree:
        syms = sp.symbols(f"x0:{X.shape[1]}")
        expr = tree_to_sympy(best_tree, syms)
        print(f"\n[✓] بهترین {'MSE' if mode=='regression' else 'دقت'}: {best_score:.6f}")
        print(f"[✓] معادله: y = {sp.simplify(expr)}")
        print(f"[✓] عمق درخت: {tree_depth(best_tree)}")
    else:
        print("[!] هیچ معادله‌ای پیدا نشد.")
