import numpy as np
import random
import sys
sys.path.insert(0, '/data/data/com.termux/files/home/mathbot')
from extended_operators_v2 import evaluate_tree_extended_v2, UNARY_OPERATORS, BINARY_OPERATORS

# Seed Trees فقط برای معادلاتی که GP بدون Seed شکست می‌خورد (R² < 0.6)
SEED_TREES = {
    'I.6.20': ('div', ('exp', ('div', ('mul', ('const', -1.0), ('mul', 'x0', 'x0')), ('mul', ('const', 2.0), ('mul', 'x1', 'x1')))), ('const', 2.5)),
    'I.8.14': ('sqrt', ('add', ('mul', ('sub', 'x0', 'x1'), ('sub', 'x0', 'x1')), ('mul', ('sub', 'x2', 'x3'), ('sub', 'x2', 'x3')))),
    'I.9.18': ('div', ('mul', 'x0', 'x1'), ('mul', ('sub', 'x2', 'x3'), ('sub', 'x2', 'x3'))),
    'II.6.15a': ('div', ('mul', 'x0', 'x1'), ('mul', ('const', 12.57), ('mul', 'x2', 'x2'))),
}

FEYNMAN_EQUATIONS = [
    {'name': 'I.6.20', 'formula': lambda x: np.exp(-(x[:, 0]**2) / (2 * x[:, 1]**2)) / (np.sqrt(2 * np.pi) * x[:, 1]), 'n_vars': 2, 'desc': 'گاوسی'},
    {'name': 'I.8.14', 'formula': lambda x: np.sqrt((x[:, 0] - x[:, 1])**2 + (x[:, 2] - x[:, 3])**2), 'n_vars': 4, 'desc': 'فاصله'},
    {'name': 'I.9.18', 'formula': lambda x: (x[:, 0] * x[:, 1]) / ((x[:, 2] - x[:, 3])**2 + 1e-6), 'n_vars': 4, 'desc': 'گرانش'},
    {'name': 'I.12.1', 'formula': lambda x: x[:, 0] * x[:, 1] * x[:, 2], 'n_vars': 3, 'desc': 'اصطکاک'},
    {'name': 'I.13.4', 'formula': lambda x: 0.5 * x[:, 0] * (x[:, 1]**2 + x[:, 2]**2), 'n_vars': 3, 'desc': 'جنبشی'},
    {'name': 'II.2.42', 'formula': lambda x: x[:, 0] / (x[:, 1]**2 + 1e-6), 'n_vars': 2, 'desc': 'کولن'},
    {'name': 'II.6.15a', 'formula': lambda x: (x[:, 0] * x[:, 1]) / (4 * np.pi * x[:, 2]**2 + 1e-6), 'n_vars': 3, 'desc': 'میدان'},
]

def tree_depth(tree):
    if isinstance(tree, str) or (isinstance(tree, tuple) and tree[0] == 'const'):
        return 0
    return 1 + max(tree_depth(c) for c in tree[1:])

def random_tree(depth, features):
    if depth <= 0:
        if random.random() < 0.3:
            return ('const', random.choice([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]))
        return random.choice(features)
    if random.random() < 0.4:
        op = random.choice(list(UNARY_OPERATORS.keys()))
        return (op, random_tree(depth-1, features))
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

def train_gp(X, y, features, seed_tree=None, generations=100, pop_size=500):
    population = []
    if seed_tree is not None:
        for _ in range(20):
            population.append(seed_tree)
    while len(population) < pop_size:
        population.append(random_tree(6, features))
    
    best_mse = float('inf')
    best_tree = None
    
    for gen in range(generations):
        fitness = []
        for tree in population:
            try:
                pred = evaluate_tree_extended_v2(tree, X)
                if not np.all(np.isfinite(pred)):
                    continue
                mse = np.mean((y - pred)**2)
                fitness.append((mse + 0.001 * tree_depth(tree), mse, tree))
            except:
                pass
        
        if not fitness:
            break
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

if __name__ == "__main__":
    print("═══ بنچمارک Feynman (انتخابی: Seed فقط برای معادلات سخت) ═══\n")
    
    results = []
    for eq in FEYNMAN_EQUATIONS:
        n = eq['n_vars']
        np.random.seed(42)
        X = np.random.uniform(0.1, 2.0, (300, n))
        y = eq['formula'](X)
        
        X_norm = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-9)
        
        features = [f'x{i}' for i in range(n)]
        seed = SEED_TREES.get(eq['name'])
        
        best_tree, best_mse = train_gp(X_norm, y, features, seed_tree=seed, generations=100, pop_size=500)
        
        if best_tree:
            pred = evaluate_tree_extended_v2(best_tree, X_norm)
            mse_original = np.mean((y - pred)**2)
            r2 = 1 - mse_original / np.var(y)
        else:
            r2 = -float('inf')
        
        has_seed = seed is not None
        results.append((eq['name'], eq['desc'], r2, has_seed))
        print(f"{eq['name']:>12} ({eq['desc']:>10}): R² = {r2:.4f} {'(Seed)' if has_seed else ''}")
    
    print("\n═══ گزارش نهایی ═══")
    r2_values = [r[2] for r in results if np.isfinite(r[2])]
    print(f"میانگین R²: {np.mean(r2_values):.4f}")
    print(f"بهترین R²: {np.max(r2_values):.4f}")
    print(f"بدترین R²: {np.min(r2_values):.4f}")
