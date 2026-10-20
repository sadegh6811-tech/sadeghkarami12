import numpy as np
import random
import sympy as sp

# داده‌ها
data = np.loadtxt('/data/data/com.termux/files/home/zeta_data.csv', delimiter=',', skiprows=1)
X = data[:, :3]  # index, real, imag
y = data[:, 3]   # gap

features = ['x0', 'x1', 'x2']

# ═══════════════════════════════════════════════════════════
# کتابخانه‌ی عملگرهای گسترده (بدون راهنمایی)
# ═══════════════════════════════════════════════════════════
def evaluate_tree(tree, X):
    if isinstance(tree, str): return X[:, int(tree[1:])]
    if isinstance(tree, tuple) and tree[0] == 'const': return np.full(X.shape[0], tree[1])
    op = tree[0]
    if op == 'abs': return np.abs(evaluate_tree(tree[1], X))
    if op == 'add': return evaluate_tree(tree[1], X) + evaluate_tree(tree[2], X)
    if op == 'sub': return evaluate_tree(tree[1], X) - evaluate_tree(tree[2], X)
    if op == 'mul': return evaluate_tree(tree[1], X) * evaluate_tree(tree[2], X)
    if op == 'div': return evaluate_tree(tree[1], X) / (evaluate_tree(tree[2], X) + 1e-9)
    if op == 'sin': return np.sin(evaluate_tree(tree[1], X))
    if op == 'cos': return np.cos(evaluate_tree(tree[1], X))
    if op == 'tanh': return np.tanh(evaluate_tree(tree[1], X))
    if op == 'atan': return np.arctan(evaluate_tree(tree[1], X))
    if op == 'exp': return np.exp(np.clip(evaluate_tree(tree[1], X), -20, 20))
    if op == 'log': return np.log(np.abs(evaluate_tree(tree[1], X)) + 1e-9)
    if op == 'sqrt': return np.sqrt(np.abs(evaluate_tree(tree[1], X)) + 1e-9)
    if op == 'pow': return np.sign(evaluate_tree(tree[1], X)) * np.abs(evaluate_tree(tree[1], X)) ** np.clip(evaluate_tree(tree[2], X), -3, 3)
    if op == 'max': return np.maximum(evaluate_tree(tree[1], X), evaluate_tree(tree[2], X))
    if op == 'min': return np.minimum(evaluate_tree(tree[1], X), evaluate_tree(tree[2], X))
    return np.zeros(X.shape[0])

def tree_to_sympy(tree, syms):
    if isinstance(tree, str): return syms[int(tree[1:])]
    if isinstance(tree, tuple) and tree[0] == 'const': return sp.Float(tree[1])
    op = tree[0]
    if op == 'abs': return sp.Abs(tree_to_sympy(tree[1], syms))
    if op == 'add': return tree_to_sympy(tree[1], syms) + tree_to_sympy(tree[2], syms)
    if op == 'sub': return tree_to_sympy(tree[1], syms) - tree_to_sympy(tree[2], syms)
    if op == 'mul': return tree_to_sympy(tree[1], syms) * tree_to_sympy(tree[2], syms)
    if op == 'div': return tree_to_sympy(tree[1], syms) / tree_to_sympy(tree[2], syms)
    if op == 'sin': return sp.sin(tree_to_sympy(tree[1], syms))
    if op == 'cos': return sp.cos(tree_to_sympy(tree[1], syms))
    if op == 'tanh': return sp.tanh(tree_to_sympy(tree[1], syms))
    if op == 'atan': return sp.atan(tree_to_sympy(tree[1], syms))
    if op == 'exp': return sp.exp(tree_to_sympy(tree[1], syms))
    if op == 'log': return sp.log(sp.Abs(tree_to_sympy(tree[1], syms)))
    if op == 'sqrt': return sp.sqrt(sp.Abs(tree_to_sympy(tree[1], syms)))
    if op == 'pow': return tree_to_sympy(tree[1], syms) ** tree_to_sympy(tree[2], syms)
    if op == 'max': return sp.Max(tree_to_sympy(tree[1], syms), tree_to_sympy(tree[2], syms))
    if op == 'min': return sp.Min(tree_to_sympy(tree[1], syms), tree_to_sympy(tree[2], syms))
    return sp.Integer(0)

UNARY = ['abs', 'sin', 'cos', 'tanh', 'atan', 'exp', 'log', 'sqrt']
BINARY = ['add', 'sub', 'mul', 'div', 'pow', 'max', 'min']

def random_tree(depth):
    if depth <= 0:
        if random.random() < 0.3:
            return ('const', random.choice([-3.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.14159, 6.28318]))
        return random.choice(features)
    if random.random() < 0.4:
        return (random.choice(UNARY), random_tree(depth-1))
    return (random.choice(BINARY), random_tree(depth-1), random_tree(depth-1))

def tree_depth(tree):
    if isinstance(tree, str) or (isinstance(tree, tuple) and tree[0] == 'const'): return 0
    return 1 + max(tree_depth(c) for c in tree[1:])

def mutate(tree, prob=0.2):
    if random.random() < prob: return random_tree(3)
    if isinstance(tree, str) or (isinstance(tree, tuple) and tree[0] == 'const'): return tree
    if tree[0] in UNARY: return (tree[0], mutate(tree[1], prob))
    return (tree[0], mutate(tree[1], prob), mutate(tree[2], prob))

def crossover(t1, t2):
    if isinstance(t1, str) or isinstance(t2, str): return random.choice([t1, t2])
    if t1[0] == 'const' or t2[0] == 'const': return random.choice([t1, t2])
    if t1[0] != t2[0]: return random.choice([t1, t2])
    if t1[0] in UNARY: return (t1[0], random.choice([t1[1], t2[1]]))
    return (t1[0], random.choice([t1[1], t2[1]]), random.choice([t1[2], t2[2]]))

# ═══════════════════════════════════════════════════════════
# اجرای GP آزاد (بدون Seed، با کتابخانه‌ی کامل)
# ═══════════════════════════════════════════════════════════
pop_size = 500
generations = 100

print("[*] کشف آزاد فرمول gap صفرهای زتا (بدون راهنمایی)...")
print("[*] کتابخانه: sin, cos, tanh, atan, exp, log, sqrt, pow, +, -, *, /, max, min\n")

population = [random_tree(6) for _ in range(pop_size)]
best_mse = float('inf')
best_tree = None

for gen in range(generations):
    fitness = []
    for tree in population:
        try:
            pred = evaluate_tree(tree, X)
            if not np.all(np.isfinite(pred)): continue
            mse = np.mean((y - pred)**2)
            complexity = tree_depth(tree)
            score = mse + 0.001 * complexity
            fitness.append((score, mse, tree))
        except: pass
    
    if not fitness: break
    fitness.sort(key=lambda x: x[0])
    
    if fitness[0][1] < best_mse:
        best_mse = fitness[0][1]
        best_tree = fitness[0][2]
    
    top = [f[2] for f in fitness[:pop_size//2]]
    new_pop = top.copy()
    while len(new_pop) < pop_size:
        p1, p2 = random.choice(top), random.choice(top)
        new_pop.append(mutate(crossover(p1, p2)))
    population = new_pop
    
    if gen % 20 == 0:
        print(f"  نسل {gen}: بهترین MSE = {best_mse:.6f}")

print(f"\n[✓] بهترین MSE: {best_mse:.6f}")
syms = sp.symbols("x0:3")
expr = tree_to_sympy(best_tree, syms)
print(f"[✓] معادله کشف‌شده: y = {sp.simplify(expr)}")
print(f"[✓] عمق درخت: {tree_depth(best_tree)}")

# تحلیل الگوی خطا
pred = evaluate_tree(best_tree, X)
errors = y - pred
print(f"\n[+] تحلیل خطا:")
print(f"    میانگین خطا: {np.mean(errors):.6f}")
print(f"    انحراف معیار خطا: {np.std(errors):.6f}")
print(f"    خطا در ۵۰ صفر اول: {np.mean(np.abs(errors[:50])):.6f}")
print(f"    خطا در ۵۰ صفر آخر: {np.mean(np.abs(errors[-50:])):.6f}")
