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

def train_gp(X, y, features, generations=50, pop_size=300, mode='correlation'):
    use_safe = (mode == 'classification')
    population = [random_tree(5, features, use_safe) for _ in range(pop_size)]
    best_score = -2.0  # همبستگی بین -1 و 1 است
    best_tree = None
    
    for gen in range(generations):
        fitness = []
        for tree in population:
            try:
                pred = evaluate_tree_extended(tree, X)
                if not np.all(np.isfinite(pred)):
                    continue
                
                # 🔧 معیار: قدرمطلق همبستگی پیرسون
                if np.std(pred) < 1e-9:
                    continue
                corr = np.corrcoef(pred, y)[0, 1]
                if np.isnan(corr):
                    continue
                score = abs(corr) - 0.001 * tree_depth(tree)
                fitness.append((-score, abs(corr), tree))
            except:
                pass
        
        if not fitness:
            break
        fitness.sort(key=lambda x: x[0])
        
        if fitness[0][1] > best_score:
            best_score = fitness[0][1]
            best_tree = fitness[0][2]
        
        top = [f[2] for f in fitness[:pop_size//2]]
        new_pop = top.copy()
        while len(new_pop) < pop_size:
            p1, p2 = random.choice(top), random.choice(top)
            new_pop.append(mutate(crossover(p1, p2), features, use_safe=use_safe))
        population = new_pop
        
        if gen % 10 == 0:
            print(f"  نسل {gen}: بهترین همبستگی = {best_score:.6f}")
    
    return best_tree, best_score

if __name__ == "__main__":
    path = sys.argv[1]
    target_col = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    if target_col < 0:
        target_col = data.shape[1] + target_col
    
    y = data[:, target_col]
    X = np.delete(data, target_col, axis=1)
    features = [f'x{i}' for i in range(X.shape[1])]
    
    print(f"[*] داده: {X.shape[0]} نقطه، {X.shape[1]} بعد")
    print(f"[*] حالت: همبستگی پیرسون")
    
    best_tree, best_corr = train_gp(X, y, features, generations=50, pop_size=300)
    
    if best_tree:
        print(f"\n[✓] بهترین همبستگی: {best_corr:.6f}")
        print(f"[✓] درخت: {best_tree}")
        print(f"[✓] عمق: {tree_depth(best_tree)}")
    else:
        print("[!] هیچ معادله‌ای پیدا نشد.")
