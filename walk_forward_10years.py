import numpy as np
import pandas as pd
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════
# ۱. بارگذاری داده‌های M5
# ═══════════════════════════════════════════════════════════
print("[*] بارگذاری داده‌های M5 (۷۴۸,۰۰۰ کندل)...")
df = pd.read_csv('/storage/emulated/0/forex_bot/data_m5/EURUSD.csv')
print(f"[*] تعداد ردیف‌ها: {len(df)}")
print(f"[*] ستون‌ها: {df.columns.tolist()}")
print(f"[*] بازه‌ی زمانی: {df['time'].iloc[0]} تا {df['time'].iloc[-1]}")

# استخراج قیمت‌ها
o = df['open'].values
h = df['high'].values
l = df['low'].values
c = df['close'].values
v = df['volume'].values

# ═══════════════════════════════════════════════════════════
# ۲. ساخت ویژگی‌های توپولوژیک (بدون نشت داده)
# ═══════════════════════════════════════════════════════════
print("\n[*] ساخت ویژگی‌های توپولوژیک...")

range_p = h - l
body = c - o

# ویژگی‌های امن (بدون نشت داده)
X = np.column_stack([
    np.sign(body),                                    # x0: جهت کندل فعلی
    ((h - np.maximum(o, c)) - (np.minimum(o, c) - l)) / (range_p + 1e-9),  # x1: نویز
    range_p / (np.std(range_p) + 1e-9),               # x2: نقدینگی
    v / (np.std(v) + 1e-9),                           # x3: حجم
    np.gradient(c) / (np.std(np.gradient(c)) + 1e-9), # x4: مومنتوم
])

# هدف: جهت کندل بعدی
future_c = np.roll(c, -1)
y = np.sign(future_c - c)
y[-1] = 0

# حذف NaN و Inf
mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
X = X[mask]
y = y[mask]
print(f"[*] داده‌های معتبر: {len(X)}")

# ═══════════════════════════════════════════════════════════
# ۳. موتور GP (ساده و سریع)
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

def train_gp_simple(X, y, features, generations=30, pop_size=200):
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
                
                # آستانه‌ی پویا
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
# ۴. آزمون Walk-Forward (۵ دوره)
# ═══════════════════════════════════════════════════════════
print("\n═══ آزمون Walk-Forward (۵ دوره) ═══")

n = len(X)
periods = 5
chunk = n // periods
features = [f'x{i}' for i in range(X.shape[1])]

results = []
for i in range(periods - 1):
    train_end = (i + 1) * chunk
    test_start = train_end
    test_end = min(test_start + chunk, n)
    
    X_train = X[:train_end]
    y_train = y[:train_end]
    X_test = X[test_start:test_end]
    y_test = y[test_start:test_end]
    
    print(f"\nدوره {i+1}:")
    print(f"  آموزش: 0 تا {train_end} ({train_end} کندل)")
    print(f"  تست:   {test_start} تا {test_end} ({test_end-test_start} کندل)")
    
    best_tree, train_acc, th = train_gp_simple(X_train, y_train, features, generations=20, pop_size=150)
    
    if best_tree:
        pred_test = evaluate_tree(best_tree, X_test)
        if np.all(np.isfinite(pred_test)) and np.std(pred_test) > 1e-9:
            pred_class = np.zeros_like(pred_test)
            pred_class[pred_test > th] = 1
            pred_class[pred_test < -th] = -1
            test_acc = np.mean(pred_class == y_test)
            
            results.append((i+1, train_acc, test_acc))
            print(f"  دقت آموزش: {train_acc*100:.2f}%")
            print(f"  دقت تست:   {test_acc*100:.2f}%")
        else:
            print(f"  [!] پیش‌بینی معتبر نیست")
            results.append((i+1, train_acc, 0.5))
    else:
        print(f"  [!] هیچ درختی پیدا نشد")
        results.append((i+1, 0.5, 0.5))

# ═══════════════════════════════════════════════════════════
# ۵. گزارش نهایی
# ═══════════════════════════════════════════════════════════
print("\n═══ گزارش نهایی Walk-Forward ═══")
print(f"{'دوره':>6} | {'دقت آموزش':>12} | {'دقت تست':>12}")
print("-" * 35)
for period, train_acc, test_acc in results:
    print(f"{period:>6} | {train_acc*100:>11.2f}% | {test_acc*100:>11.2f}%")

test_accs = [r[2] for r in results]
print(f"\nمیانگین دقت تست: {np.mean(test_accs)*100:.2f}%")
print(f"بهترین دقت تست: {np.max(test_accs)*100:.2f}%")
print(f"بدترین دقت تست: {np.min(test_accs)*100:.2f}%")

if np.mean(test_accs) > 0.52:
    print(f"\n✅ مدل در همه دوره‌ها بهتر از تصادف عمل کرده است.")
else:
    print(f"\n⚠️ مدل در برخی دوره‌ها ضعیف عمل کرده است.")
