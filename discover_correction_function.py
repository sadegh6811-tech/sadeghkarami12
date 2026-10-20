import numpy as np
import random

# ═══════════════════════════════════════════════════════════
# ۱. داده‌ها
# ═══════════════════════════════════════════════════════════
data = np.loadtxt('/data/data/com.termux/files/home/zeta_data_1000.csv', delimiter=',', skiprows=1)
T_all = data[:, 2]
actual_all = data[:, 3]

# فرمول ریمان-فون مانگولت
def riemann(T):
    return 2 * np.pi / np.log(T / (2 * np.pi) + 1e-9)

# هدف جدید: نسبت gap واقعی به پیش‌بینی ریمان
# این نسبت باید نزدیک به ۱ باشد. اگر تابع تصحیح وجود داشته باشد، این نسبت نوسان می‌کند.
ratio = actual_all / (riemann(T_all) + 1e-9)

# حذف نقاط پرت
mask = np.isfinite(ratio) & (ratio < 10) & (ratio > 0)
T = T_all[mask]
y = ratio[mask]

print(f"[*] داده: {len(T)} نقطه")
print(f"[*] میانگین نسبت: {np.mean(y):.4f} (باید نزدیک 1 باشد)")
print(f"[*] انحراف معیار نسبت: {np.std(y):.4f}")

# ═══════════════════════════════════════════════════════════
# ۲. موتور GP با کتابخانه‌ی گسترده
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
    if op == 'log': return np.log(np.abs(evaluate_tree(tree[1], X)) + 1e-9)
    if op == 'exp': return np.exp(np.clip(evaluate_tree(tree[1], X), -10, 10))
    if op == 'sqrt': return np.sqrt(np.abs(evaluate_tree(tree[1], X)) + 1e-9)
    if op == 'pow': return np.sign(evaluate_tree(tree[1], X)) * np.abs(evaluate_tree(tree[1], X)) ** np.clip(evaluate_tree(tree[2], X), -3, 3)
    return np.zeros(X.shape[0])

def random_tree(depth, features):
    if depth <= 0:
        if random.random() < 0.3:
            return ('const', random.choice([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.14159, 6.28318]))
        return random.choice(features)
    ops = ['abs', 'sin', 'cos', 'log', 'sqrt', 'add', 'sub', 'mul', 'div', 'pow']
    op = random.choice(ops)
    if op in ['abs', 'sin', 'cos', 'log', 'sqrt']:
        return (op, random_tree(depth-1, features))
    return (op, random_tree(depth-1, features), random_tree(depth-1, features))

def tree_depth(tree):
    if isinstance(tree, str) or (isinstance(tree, tuple) and tree[0] == 'const'): return 0
    return 1 + max(tree_depth(c) for c in tree[1:])

def mutate(tree, features, prob=0.2):
    if random.random() < prob: return random_tree(3, features)
    if isinstance(tree, str) or (isinstance(tree, tuple) and tree[0] == 'const'): return tree
    if tree[0] in ['abs', 'sin', 'cos', 'log', 'sqrt']: return (tree[0], mutate(tree[1], features, prob))
    return (tree[0], mutate(tree[1], features, prob), mutate(tree[2], features, prob))

def crossover(t1, t2):
    if isinstance(t1, str) or isinstance(t2, str): return random.choice([t1, t2])
    if t1[0] == 'const' or t2[0] == 'const': return random.choice([t1, t2])
    if t1[0] != t2[0]: return random.choice([t1, t2])
    if t1[0] in ['abs', 'sin', 'cos', 'log', 'sqrt']: return (t1[0], random.choice([t1[1], t2[1]]))
    return (t1[0], random.choice([t1[1], t2[1]]), random.choice([t1[2], t2[2]]))

# ═══════════════════════════════════════════════════════════
# ۳. اجرای GP برای کشف تابع تصحیح
# ═══════════════════════════════════════════════════════════
X = T.reshape(-1, 1)  # فقط T به عنوان ورودی
features = ['x0']
pop_size = 500
generations = 150

print("\n[*] کشف تابع تصحیح (Correction Function)...")
population = [random_tree(6, features) for _ in range(pop_size)]
best_mse = float('inf')
best_tree = None

for gen in range(generations):
    fitness = []
    for tree in population:
        try:
            pred = evaluate_tree(tree, X)
            if not np.all(np.isfinite(pred)): continue
            mse = np.mean((y - pred)**2)
            fitness.append((mse + 0.001 * tree_depth(tree), mse, tree))
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
        new_pop.append(mutate(crossover(p1, p2), features))
    population = new_pop
    if gen % 30 == 0:
        print(f"  نسل {gen}: بهترین MSE = {best_mse:.6f}")

print(f"\n[✓] بهترین MSE تابع تصحیح: {best_mse:.6f}")
print(f"[✓] تابع تصحیح: {best_tree}")
print(f"[✓] عمق: {tree_depth(best_tree)}")

# ═══════════════════════════════════════════════════════════
# ۴. اعمال تابع تصحیح و مقایسه
# ═══════════════════════════════════════════════════════════
correction_pred = evaluate_tree(best_tree, X)
gap_corrected = riemann(T) * correction_pred
mse_corrected = np.mean((actual_all[mask] - gap_corrected)**2)
mse_original = np.mean((actual_all[mask] - riemann(T))**2)

print(f"\n═══ نتیجه ═══")
print(f"MSE ریمان خام:           {mse_original:.6f}")
print(f"MSE ریمان با تصحیح GP:   {mse_corrected:.6f}")
if mse_corrected < mse_original:
    print(f"✅ تابع تصحیح، دقت را {100*(mse_original-mse_corrected)/mse_original:.1f}% بهبود داد!")
else:
    print(f"⚠️ تابع تصحیح، دقت را بهبود نداد.")
