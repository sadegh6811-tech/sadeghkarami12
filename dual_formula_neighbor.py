import numpy as np
import random

# ═══════════════════════════════════════════════════════════
# ۱. بارگذاری داده‌های M5 (۱۰ سال)
# ═══════════════════════════════════════════════════════════
print("[*] بارگذاری داده‌های M5...")
data = np.genfromtxt(
    '/storage/emulated/0/forex_bot/data_m5/EURUSD.csv',
    delimiter=',', skip_header=1, usecols=(1,2,3,4,5),
    dtype=float, invalid_raise=False
)
mask = np.all(np.isfinite(data), axis=1)
data = data[mask]
o, h, l, c, v = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4]
print(f"[*] تعداد کندل: {len(c)}")

# ═══════════════════════════════════════════════════════════
# ۲. ساخت ویژگی‌های پایه (فرمول A)
# ═══════════════════════════════════════════════════════════
range_p = h - l
body = c - o

X_base = np.column_stack([
    np.sign(body),                                    # x0: جهت
    ((h - np.maximum(o, c)) - (np.minimum(o, c) - l)) / (range_p + 1e-9),  # x1: نویز
    range_p / (np.std(range_p) + 1e-9),               # x2: نقدینگی
    v / (np.std(v) + 1e-9),                           # x3: حجم
    np.gradient(c) / (np.std(np.gradient(c)) + 1e-9), # x4: مومنتوم
])

# ═══════════════════════════════════════════════════════════
# ۳. ساخت ویژگی‌های «خبر همسایه» (فرمول B)
# ═══════════════════════════════════════════════════════════
K = 3  # تعداد همسایه‌ها
n = len(c)

neighbor_features = np.zeros((n, 4))
for i in range(K, n):
    # میانگین وزنی کندل‌های قبلی
    weights = np.array([0.5, 0.3, 0.2])
    neighbor_body = np.sum(weights * body[i-K:i][::-1])
    neighbor_direction = np.sum(weights * np.sign(body[i-K:i][::-1]))
    neighbor_range = np.sum(weights * range_p[i-K:i][::-1])
    neighbor_volume = np.sum(weights * v[i-K:i][::-1])
    
    neighbor_features[i] = [neighbor_body, neighbor_direction, neighbor_range, neighbor_volume]

# ═══════════════════════════════════════════════════════════
# ۴. هدف
# ═══════════════════════════════════════════════════════════
future_c = np.roll(c, -1)
y = np.sign(future_c - c)
y[-1] = 0

# پاک‌سازی
mask = np.all(np.isfinite(X_base), axis=1) & np.all(np.isfinite(neighbor_features), axis=1) & np.isfinite(y)
X_base = X_base[mask]
neighbor_features = neighbor_features[mask]
y = y[mask]
n = len(y)
print(f"[*] نمونه‌های معتبر: {n}")

# ═══════════════════════════════════════════════════════════
# ۵. موتور GP ساده
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
    if op == 'max': return np.maximum(evaluate_tree(tree[1], X), evaluate_tree(tree[2], X))
    return np.zeros(X.shape[0])

def tree_depth(tree):
    if isinstance(tree, str) or (isinstance(tree, tuple) and tree[0] == 'const'): return 0
    return 1 + max(tree_depth(c) for c in tree[1:])

def random_tree(depth, features):
    if depth <= 0:
        if random.random() < 0.3:
            return ('const', random.choice([-1.0, -0.5, 0.5, 1.0]))
        return random.choice(features)
    if random.random() < 0.4:
        return ('abs', random_tree(depth-1, features))
    op = random.choice(['add', 'sub', 'mul', 'div', 'max'])
    return (op, random_tree(depth-1, features), random_tree(depth-1, features))

def mutate(tree, features, prob=0.2):
    if random.random() < prob: return random_tree(3, features)
    if isinstance(tree, str) or (isinstance(tree, tuple) and tree[0] == 'const'): return tree
    if tree[0] == 'abs': return ('abs', mutate(tree[1], features, prob))
    return (tree[0], mutate(tree[1], features, prob), mutate(tree[2], features, prob))

def crossover(t1, t2):
    if isinstance(t1, str) or isinstance(t2, str): return random.choice([t1, t2])
    if t1[0] == 'const' or t2[0] == 'const': return random.choice([t1, t2])
    if t1[0] != t2[0]: return random.choice([t1, t2])
    if t1[0] == 'abs': return ('abs', random.choice([t1[1], t2[1]]))
    return (t1[0], random.choice([t1[1], t2[1]]), random.choice([t1[2], t2[2]]))

def train_gp(X, y, features, generations=30, pop_size=200):
    population = [random_tree(4, features) for _ in range(pop_size)]
    best_acc = 0.0
    best_tree = None
    best_th = 0.0
    
    for gen in range(generations):
        fitness = []
        for tree in population:
            try:
                pred = evaluate_tree(tree, X)
                if not np.all(np.isfinite(pred)): continue
                if np.std(pred) < 1e-9: continue
                
                th = np.percentile(pred, 50)
                pred_class = np.zeros_like(pred)
                pred_class[pred > th] = 1
                pred_class[pred < -th] = -1
                acc = np.mean(pred_class == y)
                fitness.append((-acc, acc, tree, th))
            except: pass
        
        if not fitness: break
        fitness.sort(key=lambda x: x[0])
        
        if fitness[0][1] > best_acc:
            best_acc = fitness[0][1]
            best_tree = fitness[0][2]
            best_th = fitness[0][3]
        
        top = [f[2] for f in fitness[:pop_size//2]]
        new_pop = top.copy()
        while len(new_pop) < pop_size:
            p1, p2 = random.choice(top), random.choice(top)
            new_pop.append(mutate(crossover(p1, p2), features))
        population = new_pop
    return best_tree, best_acc, best_th

# ═══════════════════════════════════════════════════════════
# ۶. آزمون Walk-Forward با ترکیب دو فرمول
# ═══════════════════════════════════════════════════════════
print("\n═══ آزمون Walk-Forward (ترکیب ۸۱٪ + ۱۹٪) ═══")

n = len(y)
periods = 5
chunk = n // periods

features_A = [f'a{i}' for i in range(X_base.shape[1])]
features_B = [f'b{i}' for i in range(neighbor_features.shape[1])]

results = []

for i in range(periods - 1):
    train_end = (i + 1) * chunk
    test_start = train_end
    test_end = min(test_start + chunk, n)
    
    X_A_train = X_base[:train_end]
    X_B_train = neighbor_features[:train_end]
    y_train = y[:train_end]
    
    X_A_test = X_base[test_start:test_end]
    X_B_test = neighbor_features[test_start:test_end]
    y_test = y[test_start:test_end]
    
    print(f"\nدوره {i+1}:")
    print(f"  آموزش: {train_end} کندل")
    print(f"  تست:   {test_end - test_start} کندل")
    
    # فرمول A (۸۱٪) — آموزش روی X_base
    tree_A, acc_A_train, th_A = train_gp(X_A_train, y_train, features_A, generations=20, pop_size=150)
    
    # فرمول B (۱۹٪) — آموزش روی neighbor_features
    tree_B, acc_B_train, th_B = train_gp(X_B_train, y_train, features_B, generations=20, pop_size=150)
    
    # پیش‌بینی روی تست
    pred_A_test = evaluate_tree(tree_A, X_A_test) if tree_A else np.zeros(len(y_test))
    pred_B_test = evaluate_tree(tree_B, X_B_test) if tree_B else np.zeros(len(y_test))
    
    # ترکیب دو فرمول با سنسور
    final_pred = np.zeros(len(y_test))
    for j in range(len(y_test)):
        # اگر فرمول A مطمئن است (سیگنال قوی)، از آن استفاده کن
        if abs(pred_A_test[j]) > abs(th_A):
            final_pred[j] = np.sign(pred_A_test[j])
        # در غیر این صورت، از فرمول B (همسایه) استفاده کن
        elif tree_B is not None:
            final_pred[j] = np.sign(pred_B_test[j])
    
    test_acc = np.mean(final_pred == y_test)
    
    results.append((i+1, acc_A_train, acc_B_train, test_acc))
    print(f"  دقت فرمول A (۸۱٪): {acc_A_train*100:.2f}%")
    print(f"  دقت فرمول B (۱۹٪): {acc_B_train*100:.2f}%")
    print(f"  دقت ترکیب (تست):    {test_acc*100:.2f}%")

# ═══════════════════════════════════════════════════════════
# ۷. گزارش نهایی
# ═══════════════════════════════════════════════════════════
print("\n═══ گزارش نهایی ═══")
print(f"{'دوره':>6} | {'A (آموزش)':>12} | {'B (آموزش)':>12} | {'ترکیب (تست)':>15}")
print("-" * 55)
for period, acc_A, acc_B, test_acc in results:
    print(f"{period:>6} | {acc_A*100:>11.2f}% | {acc_B*100:>11.2f}% | {test_acc*100:>14.2f}%")

test_accs = [r[3] for r in results]
print(f"\nمیانگین دقت تست: {np.mean(test_accs)*100:.2f}%")
print(f"بهترین دقت تست: {np.max(test_accs)*100:.2f}%")
print(f"بدترین دقت تست: {np.min(test_accs)*100:.2f}%")
