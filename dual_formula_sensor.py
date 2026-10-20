import numpy as np
import random
import sympy as sp

# ═══════════════════════════════════════════════════════════
# ۱. آماده‌سازی داده‌ها
# ═══════════════════════════════════════════════════════════
def load_data(path):
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    o, h, l, c = data[:,2], data[:,3], data[:,4], data[:,5]
    long_pnl, short_pnl = data[:,8], data[:,9]
    range_p = h - l
    
    X = np.column_stack([
        (c - np.mean(c)) / (np.std(c) + 1e-9),  # x0: position
        np.sign(c - o),                          # x1: direction
        ((h - np.maximum(o, c)) - (np.minimum(o, c) - l)) / (range_p + 1e-9),  # x2: noise
        range_p / (np.std(range_p) + 1e-9),      # x3: liquidity
        np.abs(long_pnl + short_pnl) / (np.std(long_pnl + short_pnl) + 1e-9),  # x4: volume
        data[:,6],                               # x5: buy
        data[:,7],                               # x6: sell
        np.gradient(c) / (np.std(np.gradient(c)) + 1e-9),  # x7: momentum
        np.linspace(0, 2*np.pi, len(c))          # x8: time
    ])
    
    # هدف: سیگنال واقعی
    future_c = np.roll(c, -1)
    target = np.sign(future_c - c)
    target[-1] = 0
    
    return X, target, long_pnl, short_pnl

# ═══════════════════════════════════════════════════════════
# ۲. فرمول اصلی GP (که ۸۱٪ جهت را درست می‌زند)
# ═══════════════════════════════════════════════════════════
BEST_TREE_1 = ('div',
    ('add', ('add', 'x7', 'x2'), ('const', 0.0)),
    ('add', 
        ('add', ('abs', ('mul', ('sub', ('add', 'x6', 'x1'), ('sub', 'x1', 'x0')), ('div', ('div', 'x1', 'x6'), ('abs', 'x0')))),
            ('div', ('const', -1.0), 'x5')
        ),
        ('abs', ('sub', ('abs', 'x2'), ('div', ('const', 0.5), 'x7')))
    )
)

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

# ═══════════════════════════════════════════════════════════
# ۳. موتور GP (برای فرمول دوم - ۱۹٪)
# ═══════════════════════════════════════════════════════════
def random_tree(depth, features):
    if depth <= 0:
        if random.random() < 0.3:
            return ('const', random.choice([-1.0, -0.5, 0.5, 1.0]))
        return random.choice(features)
    if random.random() < 0.4:
        return ('abs', random_tree(depth-1, features))
    op = random.choice(['add', 'sub', 'mul', 'div', 'max'])
    return (op, random_tree(depth-1, features), random_tree(depth-1, features))

def tree_depth(tree):
    if isinstance(tree, str) or (isinstance(tree, tuple) and tree[0] == 'const'): return 0
    return 1 + max(tree_depth(c) for c in tree[1:])

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

# ═══════════════════════════════════════════════════════════
# ۴. سنسور دوگانه (تشخیص ۸۱٪ یا ۱۹٪)
# ═══════════════════════════════════════════════════════════
def dual_sensor(pred_gp, target):
    """تشخیص اینکه GP درست می‌گوید (۸۱٪) یا اشتباه (۱۹٪)"""
    pred_sig = np.zeros_like(pred_gp)
    pred_sig[pred_gp > 0.1] = 1
    pred_sig[pred_gp < -0.1] = -1
    
    mask_valid = (pred_sig != 0) & (target != 0)
    is_correct = (pred_sig == target) & mask_valid  # ۸۱٪
    is_wrong = (pred_sig != target) & mask_valid    # ۱۹٪
    return pred_sig, is_correct, is_wrong

# ═══════════════════════════════════════════════════════════
# ۵. اجرای نهایی
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    path = '/storage/emulated/0/forex_bot/golden_points/EURUSD_golden.csv'
    X, target, long_pnl, short_pnl = load_data(path)
    features = [f'x{i}' for i in range(X.shape[1])]
    
    print(f"[*] داده: {X.shape[0]} نقطه، {X.shape[1]} بعد")
    
    # ۱. فرمول A (۸۱٪): همان فرمول اصلی
    pred_A = evaluate_tree(BEST_TREE_1, X)
    pred_A_sig, is_correct, is_wrong = dual_sensor(pred_A, target)
    
    n_correct = np.sum(is_correct)
    n_wrong = np.sum(is_wrong)
    print(f"\n[*] فرمول A (۸۱٪): موارد درست = {n_correct}, موارد خطا = {n_wrong}")
    
    # ۲. فرمول B (۱۹٪): آموزش روی داده‌های خطا
    print("\n[*] آموزش فرمول B روی ۱۹٪ داده‌های خطا...")
    X_wrong = X[is_wrong]
    y_wrong = target[is_wrong]
    best_B, mse_B = train_gp(X_wrong, y_wrong, features, generations=60, pop_size=300)
    print(f"    MSE فرمول B: {mse_B:.6f}")
    
    # ۳. سنسور: تعیین اینکه در هر لحظه کدام فرمول فعال باشد
    # برای سادگی، از یک آستانه در فرمول A استفاده می‌کنیم
    pred_B = evaluate_tree(best_B, X)
    
    # سنسور: اگر |pred_A| بزرگ باشد، از A استفاده کن، در غیر این صورت از B
    final_pred = np.zeros_like(pred_A)
    for i in range(len(pred_A)):
        if abs(pred_A[i]) > 0.3:
            final_pred[i] = pred_A[i]  # ۸۱٪: فرمول اصلی
        else:
            final_pred[i] = pred_B[i]  # ۱۹٪: فرمول جایگزین
    
    # ۴. ارزیابی
    final_sig = np.zeros_like(final_pred)
    final_sig[final_pred > 0.1] = 1
    final_sig[final_pred < -0.1] = -1
    
    total_pnl, wins, losses = 0.0, 0, 0
    for i in range(len(final_sig) - 1):
        if final_sig[i] == 1: pnl = long_pnl[i+1]
        elif final_sig[i] == -1: pnl = short_pnl[i+1]
        else: continue
        total_pnl += pnl
        if pnl > 0: wins += 1
        else: losses += 1
    
    print(f"\n═══ نتیجه ربات دوگانه (۸۱٪ + ۱۹٪) ═══")
    print(f"معاملات: {wins+losses}, برد: {100*wins/max(wins+losses,1):.1f}%")
    print(f"سود کل: {total_pnl:+.4f}")
    
    # ۵. مقایسه با حالت تک‌فرمولی
    total_pnl_A, wins_A, losses_A = 0.0, 0, 0
    for i in range(len(pred_A_sig) - 1):
        if pred_A_sig[i] == 1: pnl = long_pnl[i+1]
        elif pred_A_sig[i] == -1: pnl = short_pnl[i+1]
        else: continue
        total_pnl_A += pnl
        if pnl > 0: wins_A += 1
        else: losses_A += 1
    
    print(f"\n═══ مقایسه ═══")
    print(f"فرمول A تنها:  برد={100*wins_A/max(wins_A+losses_A,1):.1f}%, سود={total_pnl_A:+.4f}")
    print(f"ربات دوگانه:  برد={100*wins/max(wins+losses,1):.1f}%, سود={total_pnl:+.4f}")
    print(f"بهبود سود: {total_pnl - total_pnl_A:+.4f}")
