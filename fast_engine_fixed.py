import numpy as np
import random
import os
import json
from multiprocessing import Pool, cpu_count
import time

# ═══════════════════════════════════════════════════════════
# ۱. توابع ارزیابی
# ═══════════════════════════════════════════════════════════
def evaluate_tree(tree, X):
    if isinstance(tree, str): return X[:, int(tree[1:])]
    if isinstance(tree, tuple) and tree[0] == 'const': return np.full(X.shape[0], tree[1])
    op = tree[0]
    try:
        if op == 'abs': return np.abs(evaluate_tree(tree[1], X))
        if op == 'add': return evaluate_tree(tree[1], X) + evaluate_tree(tree[2], X)
        if op == 'sub': return evaluate_tree(tree[1], X) - evaluate_tree(tree[2], X)
        if op == 'mul': return evaluate_tree(tree[1], X) * evaluate_tree(tree[2], X)
        if op == 'div': return evaluate_tree(tree[1], X) / (evaluate_tree(tree[2], X) + 1e-9)
        if op == 'log': return np.log(np.abs(evaluate_tree(tree[1], X)) + 1e-9)
        if op == 'sin': return np.sin(evaluate_tree(tree[1], X))
        if op == 'cos': return np.cos(evaluate_tree(tree[1], X))
        if op == 'sqrt': return np.sqrt(np.abs(evaluate_tree(tree[1], X)) + 1e-9)
        return np.zeros(X.shape[0])
    except:
        return np.zeros(X.shape[0])

def tree_depth(tree):
    if isinstance(tree, str) or (isinstance(tree, tuple) and tree[0] == 'const'): return 0
    return 1 + max(tree_depth(c) for c in tree[1:])

UNARY = ['abs', 'log', 'sin', 'cos', 'sqrt']
BINARY = ['add', 'sub', 'mul', 'div']

def random_tree(depth, features):
    if depth <= 0:
        if random.random() < 0.3:
            return ('const', random.choice([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]))
        return random.choice(features)
    if random.random() < 0.4:
        return (random.choice(UNARY), random_tree(depth-1, features))
    return (random.choice(BINARY), random_tree(depth-1, features), random_tree(depth-1, features))

def mutate(tree, features, prob=0.2):
    if random.random() < prob: return random_tree(3, features)
    if isinstance(tree, str) or (isinstance(tree, tuple) and tree[0] == 'const'): return tree
    if tree[0] in UNARY: return (tree[0], mutate(tree[1], features, prob))
    return (tree[0], mutate(tree[1], features, prob), mutate(tree[2], features, prob))

def crossover(t1, t2):
    if isinstance(t1, str) or isinstance(t2, str): return random.choice([t1, t2])
    if t1[0] == 'const' or t2[0] == 'const': return random.choice([t1, t2])
    if t1[0] != t2[0]: return random.choice([t1, t2])
    if t1[0] in UNARY: return (t1[0], random.choice([t1[1], t2[1]]))
    return (t1[0], random.choice([t1[1], t2[1]]), random.choice([t1[2], t2[2]]))

def evaluate_single(args):
    tree, X, y = args
    try:
        pred = evaluate_tree(tree, X)
        if not np.all(np.isfinite(pred)):
            return None
        mse = np.mean((y - pred)**2)
        return (mse + 0.001*tree_depth(tree), mse, tree)
    except:
        return None

# ═══════════════════════════════════════════════════════════
# ۲. آموزش GP
# ═══════════════════════════════════════════════════════════
def train_gp_fast(X, y, features, generations=100, pop_size=300):
    n_cores = cpu_count()
    print(f"[*] استفاده از {n_cores} هسته CPU")
    
    population = [random_tree(5, features) for _ in range(pop_size)]
    best_mse = float('inf')
    best_tree = None
    
    with Pool(n_cores) as pool:
        for gen in range(generations):
            args = [(tree, X, y) for tree in population]
            results = pool.map(evaluate_single, args)
            fitness = [r for r in results if r is not None]
            
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
            
            if gen % 20 == 0:
                print(f"  نسل {gen}: بهترین MSE = {best_mse:.6f}")
    
    return best_tree, best_mse

# ═══════════════════════════════════════════════════════════
# ۳. اجرا با هدف نرمال‌شده
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    start_time = time.time()
    
    # بارگذاری داده‌های ۱۰۰۰ صفر
    import mpmath as mp
    mp.mp.dps = 15
    print("[*] تولید ۱۰۰۰ صفر...")
    zeros = []
    for i in range(1, 1001):
        z = mp.zetazero(i)
        zeros.append((i, float(mp.im(z))))
    data = np.array(zeros)
    
    X = data[:, 0].reshape(-1, 1)
    imag = data[:, 1]
    
    # محاسبه‌ی تقریب ریمان برای هر نقطه
    riemann_approx = 2 * np.pi * X[:, 0] / np.log(X[:, 0] / (2 * np.pi) + 1e-9)
    
    # هدف نرمال‌شده: نسبت imag به ریمان (باید حدود 1 باشد)
    y_normalized = imag / (riemann_approx + 1e-9)
    print(f"[*] میانگین نسبت: {np.mean(y_normalized):.6f}")
    print(f"[*] انحراف معیار نسبت: {np.std(y_normalized):.6f}")
    
    features = ['x0']
    print(f"\n[*] آموزش GP برای کشف نسبت (imag/Riemann)...")
    
    best_tree, best_mse = train_gp_fast(X, y_normalized, features, generations=100, pop_size=300)
    
    elapsed = time.time() - start_time
    print(f"\n[✓] زمان اجرا: {elapsed:.1f} ثانیه")
    print(f"[✓] بهترین MSE: {best_mse:.6f}")
    print(f"[✓] درخت: {best_tree}")
    
    # ارزیابی فرمول کشف‌شده
    if best_tree:
        pred_ratio = evaluate_tree(best_tree, X)
        pred_imag = pred_ratio * riemann_approx
        mse_imag = np.mean((imag - pred_imag)**2)
        print(f"\n[✓] MSE نهایی (روی imag): {mse_imag:.2f}")
        print(f"    MSE ریمان خام: {np.mean((imag - riemann_approx)**2):.2f}")
