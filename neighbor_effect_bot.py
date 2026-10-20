import numpy as np
import random
import sympy as sp

# ------------------ ۱. ساخت ویژگی‌های «خبر همسایه» ------------------
def add_neighbor_features(data):
    o, h, l, c = data[:,2], data[:,3], data[:,4], data[:,5]
    long_pnl, short_pnl = data[:,8], data[:,9]
    
    n = len(data)
    body = c - o
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    range_p = h - l
    
    # ۱. «خبر همسایه»: میانگین وزنی کندل‌های قبلی (K=3)
    neighbor_body = np.zeros(n)
    neighbor_range = np.zeros(n)
    neighbor_direction = np.zeros(n)
    
    for i in range(3, n):
        # وزن‌های همسایگی: کندل نزدیک‌تر، وزن بیشتر
        w = np.array([0.5, 0.3, 0.2])
        neighbor_body[i] = np.sum(w * body[i-3:i][::-1])
        neighbor_range[i] = np.sum(w * range_p[i-3:i][::-1])
        neighbor_direction[i] = np.sum(w * np.sign(body[i-3:i][::-1]))
    
    # ۲. مومنتوم همسایگی (تفاوت کندل فعلی با میانگین همسایه‌ها)
    local_momentum = body - neighbor_body
    
    # ۳. انرژی همسایگی (مجموع قدرمطلق بدنه‌های قبلی = «خبر»)
    neighbor_energy = np.zeros(n)
    for i in range(3, n):
        neighbor_energy[i] = np.sum(np.abs(body[i-3:i]))
    
    # ۴. نویز همسایگی
    neighbor_noise = np.zeros(n)
    for i in range(3, n):
        neighbor_noise[i] = np.std(range_p[i-3:i])
    
    # ساخت دیتاست جدید
    X = np.column_stack([
        body,              # x0
        upper_wick,        # x1
        lower_wick,        # x2
        range_p,           # x3
        neighbor_body,     # x4 ← خبر همسایه
        neighbor_range,    # x5 ← خبر همسایه
        neighbor_direction,# x6 ← خبر همسایه
        local_momentum,    # x7 ← مومنتوم محلی
        neighbor_energy,   # x8 ← انرژی همسایه
        neighbor_noise     # x9 ← نویز همسایه
    ])
    
    # هدف: PnL کندل بعدی
    y_long = np.roll(long_pnl, -1)
    y_short = np.roll(short_pnl, -1)
    y_long[-1] = 0
    y_short[-1] = 0
    
    return X, y_long, y_short

# ------------------ ۲. موتور GP ------------------
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

# ------------------ ۳. اجرا ------------------
if __name__ == "__main__":
    path = '/storage/emulated/0/forex_bot/golden_points/EURUSD_golden.csv'
    raw_data = np.loadtxt(path, delimiter=',', skiprows=1)
    
    print(f"[*] داده اصلی: {raw_data.shape}")
    X, y_long, y_short = add_neighbor_features(raw_data)
    
    # حذف ۳ کندل اول (چون ویژگی‌های همسایگی ندارند)
    X = X[3:]
    y_long = y_long[3:]
    y_short = y_short[3:]
    
    print(f"[*] ویژگی‌ها: {X.shape[1]} (شامل ۴ ویژگی همسایگی)")
    print(f"[*] نقاط: {X.shape[0]}")
    
    features = [f'x{i}' for i in range(X.shape[1])]
    
    # آموزش GP برای پیش‌بینی سود
    print("\n[*] آموزش GP برای کشف «خبر همسایه»...")
    
    pop_size = 500
    generations = 100
    population = [random_tree(5, features) for _ in range(pop_size)]
    best_mse = float('inf')
    best_tree = None
    
    for gen in range(generations):
        fitness = []
        for tree in population:
            try:
                pred = evaluate_tree(tree, X)
                if not np.all(np.isfinite(pred)): continue
                # هدف: اگر سود long مثبت بود، سیگنال خرید
                target = (y_long > 0).astype(float) - (y_short > 0).astype(float)
                mse = np.mean((target - pred)**2)
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
        if gen % 20 == 0:
            print(f"  نسل {gen}: بهترین MSE = {best_mse:.6f}")
    
    print(f"\n[✓] بهترین MSE: {best_mse:.6f}")
    print(f"[✓] درخت کشف‌شده: {best_tree}")
    
    # ارزیابی مالی
    pred = evaluate_tree(best_tree, X)
    signal = np.zeros_like(pred)
    signal[pred > 0.1] = 1
    signal[pred < -0.1] = -1
    
    total_pnl, wins, losses = 0.0, 0, 0
    for i in range(len(signal) - 1):
        if signal[i] == 1: pnl = y_long[i]
        elif signal[i] == -1: pnl = y_short[i]
        else: continue
        total_pnl += pnl
        if pnl > 0: wins += 1
        else: losses += 1
    
    print(f"\n═══ نتیجه با مدل «خبر همسایه» ═══")
    print(f"معاملات: {wins+losses}, برد: {100*wins/max(wins+losses,1):.1f}%")
    print(f"سود کل: {total_pnl:+.4f}")
