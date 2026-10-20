import numpy as np
import random

# بارگذاری داده
data = np.loadtxt('/storage/emulated/0/forex_bot/golden_points/EURUSD_golden.csv', delimiter=',', skiprows=1)
o, h, l, c = data[:,2], data[:,3], data[:,4], data[:,5]
long_pnl, short_pnl = data[:,8], data[:,9]
range_p = h - l

X = np.column_stack([
    (c - np.mean(c)) / (np.std(c) + 1e-9),
    np.sign(c - o),
    ((h - np.maximum(o, c)) - (np.minimum(o, c) - l)) / (range_p + 1e-9),
    range_p / (np.std(range_p) + 1e-9),
    np.abs(long_pnl + short_pnl) / (np.std(long_pnl + short_pnl) + 1e-9),
    data[:,6], data[:,7],
    np.gradient(c) / (np.std(np.gradient(c)) + 1e-9),
    np.linspace(0, 2*np.pi, len(c))
])

# هدف: جهت (نه PnL) — برای طبقه‌بندی
future_c = np.roll(c, -1)
target = np.sign(future_c - c)
target[-1] = 0

# ═══════════════════════════════════════════════════════════
# موتور GP (ساده)
# ═══════════════════════════════════════════════════════════
def evaluate_tree(tree, X):
    if isinstance(tree, str): return X[:, int(tree[1:])]
    if isinstance(tree, tuple) and tree[0] == 'const': return np.full(X.shape[0], tree[1])
    op = tree[0]
    if op == 'abs': return np.abs(evaluate_tree(tree[1], X))
    if op == 'add': return evaluate_tree(tree[1], X) + evaluate_tree(tree[2], X)
    if op == 'sub': return evaluate_tree(tree[1], X) - evaluate_tree(tree[2], X)
    if op == 'mul': return evaluate_tree(tree[1], X) * evaluate_tree(tree[2], X)
    if op == 'div': return evaluate_tree(tree[1], X) / (evaluate_tree(tree[2], X) + 1e-6)
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

def train_gp_classifier(X, y, features, generations=60, pop_size=300):
    population = [random_tree(5, features) for _ in range(pop_size)]
    best_acc = 0.0
    best_tree = None
    for gen in range(generations):
        fitness = []
        for tree in population:
            try:
                pred = evaluate_tree(tree, X)
                if not np.all(np.isfinite(pred)): continue
                pred_class = np.zeros_like(pred)
                pred_class[pred > 0.1] = 1
                pred_class[pred < -0.1] = -1
                acc = np.mean(pred_class == y)
                fitness.append((acc - 0.001*tree_depth(tree), acc, tree))
            except: pass
        if not fitness: break
        fitness.sort(key=lambda x: x[0], reverse=True)
        if fitness[0][1] > best_acc:
            best_acc = fitness[0][1]
            best_tree = fitness[0][2]
        top = [f[2] for f in fitness[:pop_size//2]]
        new_pop = top.copy()
        while len(new_pop) < pop_size:
            p1, p2 = random.choice(top), random.choice(top)
            new_pop.append(mutate(crossover(p1, p2), features))
        population = new_pop
    return best_tree, best_acc

def evaluate_trading(signals, y_long, y_short):
    total_pnl, wins, losses = 0.0, 0, 0
    for i in range(len(signals) - 1):
        if signals[i] == 1: pnl = y_long[i]
        elif signals[i] == -1: pnl = y_short[i]
        else: continue
        total_pnl += pnl
        if pnl > 0: wins += 1
        else: losses += 1
    return total_pnl, wins, losses

# ═══════════════════════════════════════════════════════════
# آزمون Out-of-Sample
# ═══════════════════════════════════════════════════════════
features = [f'x{i}' for i in range(X.shape[1])]
n = len(X)
split = int(n * 0.7)
X_train, X_test = X[:split], X[split:]
y_train, y_test = target[:split], target[split:]
long_train, long_test = long_pnl[:split], long_pnl[split:]
short_train, short_test = short_pnl[:split], short_pnl[split:]

print(f"[*] آموزش: {split} | تست: {n-split}")
print("[*] آموزش GP Classifier...")

best_tree, train_acc = train_gp_classifier(X_train, y_train, features)

pred_train = evaluate_tree(best_tree, X_train)
pred_test = evaluate_tree(best_tree, X_test)
sig_train = np.zeros_like(pred_train)
sig_train[pred_train > 0.1] = 1
sig_train[pred_train < -0.1] = -1
sig_test = np.zeros_like(pred_test)
sig_test[pred_test > 0.1] = 1
sig_test[pred_test < -0.1] = -1

test_acc = np.mean(sig_test == y_test)

pnl_train, wins_tr, loss_tr = evaluate_trading(sig_train, long_train, short_train)
pnl_test, wins_te, loss_te = evaluate_trading(sig_test, long_test, short_test)

print(f"\n═══ نتیجه ═══")
print(f"دقت آموزش: {100*train_acc:.1f}%")
print(f"دقت تست:   {100*test_acc:.1f}%")
print(f"\nسود آموزش: {pnl_train:+.4f} ({wins_tr+loss_tr} معامله)")
print(f"سود تست:   {pnl_test:+.4f} ({wins_te+loss_te} معامله)")

if pnl_test > 0:
    print(f"\n✅ مدل تعمیم‌پذیر است! سود تست مثبت است.")
else:
    print(f"\n❌ مدل Overfit است! سود تست منفی است.")
