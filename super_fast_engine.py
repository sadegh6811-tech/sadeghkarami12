"""موتور فوق‌سریع GP با موازی‌سازی و ذخیره‌سازی بهینه"""
import numpy as np
import random
import os
import json
from multiprocessing import Pool, cpu_count

# ═══════════════════════════════════════════════════════════
# ۱. توابع ارزیابی (سریع‌تر از نسخه‌ی قبلی)
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
        if op == 'exp': return np.exp(np.clip(evaluate_tree(tree[1], X), -10, 10))
        if op == 'sqrt': return np.sqrt(np.abs(evaluate_tree(tree[1], X)) + 1e-9)
        if op == 'max': return np.maximum(evaluate_tree(tree[1], X), evaluate_tree(tree[2], X))
        return np.zeros(X.shape[0])
    except:
        return np.zeros(X.shape[0])

def tree_depth(tree):
    if isinstance(tree, str) or (isinstance(tree, tuple) and tree[0] == 'const'): return 0
    return 1 + max(tree_depth(c) for c in tree[1:])

UNARY = ['abs', 'log', 'sin', 'cos', 'exp', 'sqrt']
BINARY = ['add', 'sub', 'mul', 'div', 'max']

def random_tree(depth, features):
    if depth <= 0:
        if random.random() < 0.3:
            return ('const', random.choice([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.14159, 6.28318]))
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

# ═══════════════════════════════════════════════════════════
# ۲. ارزیابی موازی (با multiprocessing)
# ═══════════════════════════════════════════════════════════
def evaluate_single(args):
    """ارزیابی یک درخت (برای multiprocessing)"""
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
# ۳. آموزش GP با موازی‌سازی
# ═══════════════════════════════════════════════════════════
def train_gp_fast(X, y, features, generations=100, pop_size=500, checkpoint_file=None):
    n_cores = cpu_count()
    print(f"[*] استفاده از {n_cores} هسته CPU")
    
    # بارگذاری checkpoint
    start_gen = 0
    best_mse = float('inf')
    best_tree = None
    if checkpoint_file and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                start_gen = data['gen']
                best_mse = data['best_mse']
                print(f"[*] ادامه از نسل {start_gen}، بهترین MSE: {best_mse:.4f}")
        except: pass
    
    # جمعیت اولیه
    population = [random_tree(6, features) for _ in range(pop_size)]
    
    with Pool(n_cores) as pool:
        for gen in range(start_gen, generations):
            # ارزیابی موازی
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
            
            if gen % 10 == 0:
                print(f"  نسل {gen}: بهترین MSE = {best_mse:.4f}")
            
            # ذخیره‌ی checkpoint
            if checkpoint_file and gen % 25 == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump({'gen': gen, 'best_mse': best_mse}, f)
    
    return best_tree, best_mse

# ═══════════════════════════════════════════════════════════
# ۴. تولید صفرها با روش سریع‌تر
# ═══════════════════════════════════════════════════════════
def generate_zeros_fast(N=10000):
    """تولید صفرها با استفاده از mpmath (هنوز سریع‌ترین راه)"""
    import mpmath as mp
    mp.mp.dps = 15  # دقت کمتر = سریع‌تر
    
    print(f"[*] تولید {N} صفر تابع زتا (روش سریع)...")
    zeros = []
    for i in range(1, N + 1):
        z = mp.zetazero(i)
        zeros.append((i, float(mp.im(z))))
        if i % 1000 == 0:
            print(f"    {i} صفر...")
    
    return np.array(zeros)

# ═══════════════════════════════════════════════════════════
# ۵. اجرای اصلی
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import time
    start_time = time.time()
    
    print("[*] شروع آزمایش سریع...")
    data = generate_zeros_fast(N=1000)
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]
    
    features = ['x0']
    print(f"[*] آموزش با {len(X)} نقطه و {cpu_count()} هسته...")
    
    best_tree, best_mse = train_gp_fast(
        X, y, features, 
        generations=100, pop_size=300,
        checkpoint_file='/data/data/com.termux/files/home/gp_fast_checkpoint.json'
    )
    
    elapsed = time.time() - start_time
    print(f"\n[✓] زمان اجرا: {elapsed:.1f} ثانیه")
    print(f"[✓] بهترین MSE: {best_mse:.4f}")
    print(f"[✓] درخت: {best_tree}")
