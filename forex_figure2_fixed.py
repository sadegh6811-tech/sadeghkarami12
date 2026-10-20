import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════
# ۱. بارگذاری داده‌ها
# ═══════════════════════════════════════════════════════════
data = np.loadtxt('/storage/emulated/0/forex_bot/golden_points/EURUSD_golden.csv', 
                   delimiter=',', skiprows=1)
o, h, l, c = data[:,2], data[:,3], data[:,4], data[:,5]
long_pnl, short_pnl = data[:,8], data[:,9]
range_p = h - l

# ساخت ویژگی‌های توپولوژیک ۹بعدی
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

# هدف: جهت حرکت (نه PnL)
future_c = np.roll(c, -1)
target = np.sign(future_c - c)
target[-1] = 0

# ═══════════════════════════════════════════════════════════
# ۲. موتور GP (اصلاح‌شده)
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
        
        # 🔧 اصلاح: به‌روزرسانی best_tree با بهترین دقت
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

# ═══════════════════════════════════════════════════════════
# ۳. آزمون Out-of-Sample (۷۰/۳۰)
# ═══════════════════════════════════════════════════════════
features = [f'x{i}' for i in range(X.shape[1])]
n = len(X)
split = int(n * 0.7)

X_train, X_test = X[:split], X[split:]
y_train, y_test = target[:split], target[split:]
long_train, long_test = long_pnl[:split], long_pnl[split:]
short_train, short_test = short_pnl[:split], short_pnl[split:]

print(f"[*] داده: {n} نقطه")
print(f"[*] آموزش: {split} (70%)")
print(f"[*] تست:   {n-split} (30%)")

best_tree, train_acc = train_gp(X_train, y_train, features, generations=50, pop_size=300)
print(f"[*] دقت آموزش: {train_acc*100:.2f}%")

# ═══════════════════════════════════════════════════════════
# ۴. ارزیابی و محاسبه Equity Curve
# ═══════════════════════════════════════════════════════════
def evaluate_trading_equity(signals, y_long, y_short):
    equity = [0.0]
    total = 0.0
    for i in range(len(signals) - 1):
        if signals[i] == 1: pnl = y_long[i]
        elif signals[i] == -1: pnl = y_short[i]
        else: pnl = 0
        total += pnl
        equity.append(total)
    return equity

pred_train = evaluate_tree(best_tree, X_train)
pred_test = evaluate_tree(best_tree, X_test)

sig_train = np.zeros_like(pred_train)
sig_train[pred_train > 0.1] = 1
sig_train[pred_train < -0.1] = -1

sig_test = np.zeros_like(pred_test)
sig_test[pred_test > 0.1] = 1
sig_test[pred_test < -0.1] = -1

test_acc = np.mean(sig_test == y_test)
print(f"[*] دقت تست:   {test_acc*100:.2f}%")

equity_train = evaluate_trading_equity(sig_train, long_train, short_train)
equity_test = evaluate_trading_equity(sig_test, long_test, short_test)

print(f"\n═══ نتایج ═══")
print(f"سود آموزش: {equity_train[-1]:+.4f}")
print(f"سود تست:   {equity_test[-1]:+.4f}")

# ═══════════════════════════════════════════════════════════
# ۵. رسم Equity Curve
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))

x_train_plot = np.arange(len(equity_train))
ax.plot(x_train_plot, equity_train, color='#1f77b4', linewidth=2, 
        label=f'Train (P&L = {equity_train[-1]:+.2f})')

x_test_plot = np.arange(len(equity_test)) + len(equity_train)
ax.plot(x_test_plot, equity_test, color='#ff7f0e', linewidth=2, 
        label=f'Test / Out-of-Sample (P&L = {equity_test[-1]:+.2f})')

ax.axvline(x=len(equity_train), color='red', linestyle='--', 
           linewidth=2, label='Train/Test Split (70%)')

ax.set_xlabel('Time (Candles)', fontsize=13)
ax.set_ylabel('Cumulative P&L', fontsize=13)
ax.set_title('Figure 2: Equity Curve - EURUSD (Out-of-Sample Validation)\n'
             f'Train P&L: {equity_train[-1]:+.2f} | Test P&L: {equity_test[-1]:+.2f}',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='best')

plt.tight_layout()
plt.savefig('/data/data/com.termux/files/home/figure2_forex_equity.png', 
            dpi=150, bbox_inches='tight')
print("\n[+] شکل ۲ ساخته شد: figure2_forex_equity.png")
