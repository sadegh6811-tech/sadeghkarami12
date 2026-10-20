"""
موتور GP با کتابخانه‌ی عملگرهای گسترده
"""
import numpy as np
import random
import sys
sys.path.insert(0, '/data/data/com.termux/files/home/mathbot')
from extended_operators import evaluate_tree_extended, UNARY_OPERATORS, BINARY_OPERATORS

# ═══════════════════════════════════════════════════════════
# ۱. توابع درخت
# ═══════════════════════════════════════════════════════════
def tree_depth(tree):
    if isinstance(tree, str) or (isinstance(tree, tuple) and tree[0] == 'const'):
        return 0
    return 1 + max(tree_depth(c) for c in tree[1:])

def random_tree(depth, features):
    if depth <= 0:
        if random.random() < 0.3:
            return ('const', random.choice([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.14159, 6.28318]))
        return random.choice(features)
    
    # انتخاب تصادفی از عملگرهای یکانی یا دودویی
    if random.random() < 0.4:
        op = random.choice(list(UNARY_OPERATORS.keys()))
        return (op, random_tree(depth-1, features))
    else:
        op = random.choice(list(BINARY_OPERATORS.keys()))
        return (op, random_tree(depth-1, features), random_tree(depth-1, features))

def mutate(tree, features, prob=0.2):
    if random.random() < prob:
        return random_tree(3, features)
    if isinstance(tree, str) or (isinstance(tree, tuple) and tree[0] == 'const'):
        return tree
    if tree[0] in UNARY_OPERATORS:
        return (tree[0], mutate(tree[1], features, prob))
    return (tree[0], mutate(tree[1], features, prob), mutate(tree[2], features, prob))

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

# ═══════════════════════════════════════════════════════════
# ۲. موتور GP
# ═══════════════════════════════════════════════════════════
def train_gp(X, y, features, generations=50, pop_size=300, mode='regression'):
    population = [random_tree(5, features) for _ in range(pop_size)]
    best_score = float('inf') if mode == 'regression' else 0.0
    best_tree = None
    
    for gen in range(generations):
        fitness = []
        for tree in population:
            try:
                pred = evaluate_tree_extended(tree, X)
                if not np.all(np.isfinite(pred)):
                    continue
                
                if mode == 'regression':
                    mse = np.mean((y - pred)**2)
                    score = mse + 0.001 * tree_depth(tree)
                    fitness.append((score, mse, tree))
                else:
                    pred_class = np.zeros_like(pred)
                    pred_class[pred > 0.1] = 1
                    pred_class[pred < -0.1] = -1
                    acc = np.mean(pred_class == y)
                    score = -acc + 0.001 * tree_depth(tree)
                    fitness.append((score, acc, tree))
            except:
                pass
        
        if not fitness:
            break
        fitness.sort(key=lambda x: x[0])
        
        if mode == 'regression':
            if fitness[0][1] < best_score:
                best_score = fitness[0][1]
                best_tree = fitness[0][2]
        else:
            if -fitness[0][1] > best_score:
                best_score = -fitness[0][1]
                best_tree = fitness[0][2]
        
        top = [f[2] for f in fitness[:pop_size//2]]
        new_pop = top.copy()
        while len(new_pop) < pop_size:
            p1, p2 = random.choice(top), random.choice(top)
            new_pop.append(mutate(crossover(p1, p2), features))
        population = new_pop
        
        if gen % 10 == 0:
            print(f"  نسل {gen}: بهترین {'MSE' if mode=='regression' else 'دقت'} = {best_score:.6f}")
    
    return best_tree, best_score

# ═══════════════════════════════════════════════════════════
# ۳. اجرا
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python genetic_engine_extended.py <file.csv> [target_col] [mode]")
        sys.exit(1)
    
    path = sys.argv[1]
    target_col = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    mode = sys.argv[3] if len(sys.argv) > 3 else 'regression'
    
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    if target_col < 0:
        target_col = data.shape[1] + target_col
    
    y = data[:, target_col]
    X = np.delete(data, target_col, axis=1)
    features = [f'x{i}' for i in range(X.shape[1])]
    
    print(f"[*] داده: {X.shape[0]} نقطه، {X.shape[1]} بعد")
    print(f"[*] حالت: {mode}")
    print(f"[*] کتابخانه: {len(UNARY_OPERATORS)} یکانی + {len(BINARY_OPERATORS)} دودویی")
    
    best_tree, best_score = train_gp(X, y, features, generations=50, pop_size=300, mode=mode)
    
    if best_tree:
        print(f"\n[✓] بهترین {'MSE' if mode=='regression' else 'دقت'}: {best_score:.6f}")
        print(f"[✓] درخت: {best_tree}")
        print(f"[✓] عمق: {tree_depth(best_tree)}")
    else:
        print("[!] هیچ معادله‌ای پیدا نشد.")
