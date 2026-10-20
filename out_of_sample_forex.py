import numpy as np
import random

# ═══════════════════════════════════════════════════════════
# ۱. بارگذاری داده
# ═══════════════════════════════════════════════════════════
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

next_long = np.roll(long_pnl, -1)
next_short = np.roll(short_pnl, -1)
next_long[-1] = 0
next_short[-1] = 0

# ═══════════════════════════════════════════════════════════
# ۲. موتور GP
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

def train_gp(X, y, features, generations=60, pop_size=300):
    population = [random_tree(5, features) for _ in range(pop_size)]
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
    return best_tree, best_mse

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
# ۳. آزمون Out-of-Sample
# ═══════════════════════════════════════════════════════════
features = [f'x{i}' for i in range(X.shape[1])]

n = len(X)
split = int(n * 0.7)
X_train, X_test = X[:split], X[split:]
y_train, y_test = next_long[:split], next_long[split:]
y_short_train, y_short_test = next_short[:split], next_short[split:]

print(f"[*] داده: {n} نقطه")
print(f"[*] آموزش: {split} نقطه (70%)")
print(f"[*] تست:   {n - split} نقطه (30%)")
print()

print("[*] آموزش فرمول A روی داده آموزش...")
best_A, mse_A_train = train_gp(X_train, y_train, features, generations=50, pop_size=300)

pred_A_train = evaluate_tree(best_A, X_train)
pred_A_test = evaluate_tree(best_A, X_test)
mse_A_test = np.mean((y_test - pred_A_test)**2)

print(f"\n═══ فرمول A ═══")
print(f"MSE آموزش: {mse_A_train:.6f}")
print(f"MSE تست:   {mse_A_test:.6f}")

sig_A_train = np.zeros_like(pred_A_train)
sig_A_train[pred_A_train > 0.01] = 1
sig_A_train[pred_A_train < -0.01] = -1
pnl_train, wins_train, losses_train = evaluate_trading(sig_A_train, y_train, y_short_train)

sig_A_test = np.zeros_like(pred_A_test)
sig_A_test[pred_A_test > 0.01] = 1
sig_A_test[pred_A_test < -0.01] = -1
pnl_test, wins_test, losses_test = evaluate_trading(sig_A_test, y_test, y_short_test)

print(f"\n═══ ارزیابی مالی ═══")
print(f"آموزش: معاملات={wins_train+losses_train}, برد={100*wins_train/max(wins_train+losses_train,1):.1f}%, سود={pnl_train:+.4f}")
print(f"تست:   معاملات={wins_test+losses_test}, برد={100*wins_test/max(wins_test+losses_test,1):.1f}%, سود={pnl_test:+.4f}")

ratio = pnl_test / pnl_train if pnl_train != 0 else float('inf')
print(f"\n[★] نسبت (تست/آموزش): {ratio:.4f}")

if pnl_test > 0 and ratio > 0.3:
    print("✅ مدل تعمیم‌پذیر است! سود تست مثبت و قابل توجه است.")
elif pnl_test > 0:
    print("⚠️ مدل تعمیم‌پذیر است اما ضعیف‌تر از آموزش.")
else:
    print("❌ مدل Overfit است! سود تست منفی است.")
