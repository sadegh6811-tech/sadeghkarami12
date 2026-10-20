import numpy as np
import random
import sys
sys.path.insert(0, '/data/data/com.termux/files/home/mathbot')
from extended_operators import evaluate_tree_extended, UNARY_OPERATORS, BINARY_OPERATORS

SAFE_UNARY = {k: v for k, v in UNARY_OPERATORS.items() if k not in ['gamma']}
SAFE_BINARY = {k: v for k, v in BINARY_OPERATORS.items() if k not in ['pow', 'mod']}

def tree_depth(tree):
    if isinstance(tree, str) or (isinstance(tree, tuple) and tree[0] == 'const'):
        return 0
    return 1 + max(tree_depth(c) for c in tree[1:])

def random_tree(depth, features, use_safe=False):
    if depth <= 0:
        if random.random() < 0.3:
            return ('const', random.choice([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]))
        return random.choice(features)
    unary = SAFE_UNARY if use_safe else UNARY_OPERATORS
    binary = SAFE_BINARY if use_safe else BINARY_OPERATORS
    if random.random() < 0.4:
        op = random.choice(list(unary.keys()))
        return (op, random_tree(depth-1, features, use_safe))
    else:
        op = random.choice(list(binary.keys()))
        return (op, random_tree(depth-1, features, use_safe), random_tree(depth-1, features, use_safe))

def mutate(tree, features, prob=0.2, use_safe=False):
    if random.random() < prob:
        return random_tree(3, features, use_safe)
    if isinstance(tree, str) or (isinstance(tree, tuple) and tree[0] == 'const'):
        return tree
    if tree[0] in UNARY_OPERATORS:
        return (tree[0], mutate(tree[1], features, prob, use_safe))
    return (tree[0], mutate(tree[1], features, prob, use_safe), mutate(tree[2], features, prob, use_safe))

def crossover(t1, t2):
    if isinstance(t1, str) or isinstance(t2, str):
        return random.choice([t1, t2])
    if t1[0] == 'const' or t2[0] == 'const':
        return random.choice([t1, t2])
    if t1[0] != t2[0]:
        return random.choice([t1, t2])
    if t1[0] in UNARY_OPERATORS:
        return (t1[0], random.choice([t1[1], t2[1]]))
    return (t1[0], random.choice([t1[1], t2[1]]), random.choice([t1[2], t2[2]]))

def train_gp(X, y, features, generations=50, pop_size=300):
    population = [random_tree(5, features) for _ in range(pop_size)]
    best_acc = 0.0
    best_tree = None
    best_threshold = 0.0
    
    for gen in range(generations):
        fitness = []
        for tree in population:
            try:
                pred = evaluate_tree_extended(tree, X)
                if not np.all(np.isfinite(pred)):
                    continue
                if np.std(pred) < 1e-9:
                    continue
                
                # 🔧 جستجوی بهترین آستانه برای classification
                best_tree_acc = 0.0
                best_th = 0.0
                for th_pct in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
                    th = np.percentile(pred, th_pct)
                    pred_class = np.zeros_like(pred)
                    pred_class[pred > th] = 1
                    pred_class[pred < -th] = -1
                    acc = np.mean(pred_class == y)
                    if acc > best_tree_acc:
                        best_tree_acc = acc
                        best_th = th
                
                score = best_tree_acc - 0.001 * tree_depth(tree)
                fitness.append((-score, best_tree_acc, tree, best_th))
            except:
                pass
        
        if not fitness:
            break
        fitness.sort(key=lambda x: x[0])
        
        if fitness[0][1] > best_acc:
            best_acc = fitness[0][1]
            best_tree = fitness[0][2]
            best_threshold = fitness[0][3]
        
        top = [f[2] for f in fitness[:pop_size//2]]
        new_pop = top.copy()
        while len(new_pop) < pop_size:
            p1, p2 = random.choice(top), random.choice(top)
            new_pop.append(mutate(crossover(p1, p2), features))
        population = new_pop
        
        if gen % 10 == 0:
            print(f"  نسل {gen}: بهترین دقت = {best_acc:.4f}")
    
    return best_tree, best_acc, best_threshold

if __name__ == "__main__":
    path = sys.argv[1]
    target_col = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    print(f"[*] تعداد ستون‌های داده اصلی: {data.shape[1]}")
    
    if target_col < 0:
        target_col = data.shape[1] + target_col
    
    y = data[:, target_col]
    X = np.delete(data, target_col, axis=1)
    
    # 🔧 حذف ستون‌های نشت‌دهنده (x0, x5, x6, x8)
    if X.shape[1] == 9:
        keep_idx = [1, 2, 3, 4, 7]
        X = X[:, keep_idx]
        print("[*] x0، x5، x6، x8 حذف شدند.")
        print(f"[*] ویژگی‌های باقی‌مانده: {X.shape[1]}")
    
    features = [f'x{i}' for i in range(X.shape[1])]
    print(f"[*] داده: {X.shape[0]} نقطه، {X.shape[1]} بعد")
    
    # تقسیم 70/30
    n = len(X)
    split = int(n * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"[*] آموزش: {split} (70%)")
    print(f"[*] تست:   {n-split} (30%)")
    print()
    
    best_tree, train_acc, train_th = train_gp(X_train, y_train, features, generations=50, pop_size=300)
    
    if best_tree:
        print(f"\n[✓] دقت آموزش: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"[✓] آستانه آموزش: {train_th:.4f}")
        
        # ارزیابی روی تست با همان آستانه
        pred_test = evaluate_tree_extended(best_tree, X_test)
        if np.all(np.isfinite(pred_test)):
            pred_class_test = np.zeros_like(pred_test)
            pred_class_test[pred_test > train_th] = 1
            pred_class_test[pred_test < -train_th] = -1
            test_acc = np.mean(pred_class_test == y_test)
            print(f"[✓] دقت تست:   {test_acc:.4f} ({test_acc*100:.2f}%)")
            
            if test_acc > 0.55:
                print(f"\n✅ GP یک قانون واقعی کشف کرده (دقت تست > 55%).")
            elif test_acc > 0.50:
                print(f"\n⚠️ GP یک سیگنال ضعیف کشف کرده (50% < دقت تست < 55%).")
            else:
                print(f"\n❌ GP دچار Overfitting شده (دقت تست < 50%).")
                print("   این یعنی همبستگی‌های 99.6٪ قبلی، نشت داده بوده‌اند.")
        else:
            print("[!] پیش‌بینی روی تست معتبر نیست.")
        
        print(f"\n[✓] درخت: {best_tree}")
        print(f"[✓] عمق: {tree_depth(best_tree)}")
    else:
        print("[!] هیچ معادله‌ای پیدا نشد.")
