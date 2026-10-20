"""موتور ژنتیک با سنسور همبستگی چندلایه (نسخه نهایی)."""
import sys
import numpy as np
import random
import sympy as sp
from scipy.stats import spearmanr, pearsonr

# ═══════════════════════════════════════════════════════════
# ۱. سنسور همبستگی چندلایه
# ═══════════════════════════════════════════════════════════
def correlation_sensor(X, y, threshold=0.05):
    n, d = X.shape
    importances = np.zeros(d)
    details = []
    
    for i in range(d):
        if np.std(X[:, i]) < 1e-6:
            details.append((i, 0, 0, 0, 0, 0))
            continue
        
        try:
            r_pearson, _ = pearsonr(X[:, i], y)
        except: r_pearson = 0.0
        
        try:
            r_spearman, _ = spearmanr(X[:, i], y)
        except: r_spearman = 0.0
        
        try:
            r_abs, _ = pearsonr(np.abs(X[:, i]), np.abs(y))
        except: r_abs = 0.0
        
        try:
            r_sq, _ = pearsonr(X[:, i]**2, y**2)
        except: r_sq = 0.0
        
        max_corr = max(abs(r_pearson), abs(r_spearman), abs(r_abs), abs(r_sq))
        importances[i] = max_corr
        details.append((i, r_pearson, r_spearman, r_abs, r_sq, max_corr))
    
    if importances.sum() > 0:
        importances = importances / importances.sum()
    
    mandatory = [f'x{i}' for i in range(d) if importances[i] > threshold]
    
    print(f"[*] سنسور همبستگی چندلایه (threshold={threshold}):")
    print(f"    {'متغیر':<6} {'پیرسون':<10} {'اسپیرمن':<10} {'قدر مطلق':<10} {'مربع':<10} {'نهایی':<10}")
    for (i, rp, rs, ra, rsq, rm) in details:
        marker = "✅" if f'x{i}' in mandatory else "  "
        print(f"    x{i}   {rp:>8.3f}  {rs:>8.3f}  {ra:>8.3f}  {rsq:>8.3f}  {rm:>8.3f}  {marker}")
    
    if not mandatory:
        mandatory = [f'x{i}' for i in range(d)]
        print("    [!] هیچ متغیری اجباری نشد. همه اجباری شدند.")
    
    return mandatory

# ═══════════════════════════════════════════════════════════
# ۲. توابع GP
# ═══════════════════════════════════════════════════════════
def evaluate_node(node, X):
    if isinstance(node, str):
        return X[:, int(node[1:])]
    op = node[0]
    if op == 'abs':
        return np.abs(evaluate_node(node[1], X))
    return np.maximum(evaluate_node(node[1], X), evaluate_node(node[2], X))

def tree_to_sympy(node, syms):
    if isinstance(node, str):
        return syms[int(node[1:])]
    op = node[0]
    if op == 'abs':
        return sp.Abs(tree_to_sympy(node[1], syms))
    return sp.Max(tree_to_sympy(node[1], syms), tree_to_sympy(node[2], syms))

def get_vars_used(node):
    if isinstance(node, str):
        return [node]
    result = []
    for child in node[1:]:
        if isinstance(child, (tuple, str)):
            result.extend(get_vars_used(child))
    return result

def create_random_tree(depth, features, mandatory):
    if depth == 0:
        if random.random() < 0.7 and mandatory:
            return str(random.choice(mandatory))
        return str(np.random.choice(features))
    op = random.choice(['abs', 'max'])
    if op == 'abs':
        return ('abs', create_random_tree(depth-1, features, mandatory))
    return ('max', create_random_tree(depth-1, features, mandatory), 
                   create_random_tree(depth-1, features, mandatory))

def crossover(parent1, parent2):
    if isinstance(parent1, str) or isinstance(parent2, str):
        return random.choice([parent1, parent2])
    if parent1[0] != parent2[0]:
        return random.choice([parent1, parent2])
    if parent1[0] == 'abs':
        return ('abs', random.choice([parent1[1], parent2[1]]))
    return ('max', random.choice([parent1[1], parent2[1]]), random.choice([parent1[2], parent2[2]]))

def mutate(node, features, mandatory):
    if isinstance(node, str):
        if random.random() < 0.7 and mandatory:
            return str(random.choice(mandatory))
        return str(np.random.choice(features))
    if random.random() < 0.3:
        return create_random_tree(2, features, mandatory)
    return node

# ═══════════════════════════════════════════════════════════
# ۳. موتور ژنتیک
# ═══════════════════════════════════════════════════════════
def genetic_programming(X, y, features, mandatory, generations=100, population_size=500):
    population = [create_random_tree(4, features, mandatory) for _ in range(population_size)]
    
    for gen in range(generations):
        fitness = []
        for tree in population:
            try:
                pred = evaluate_node(tree, X)
                mse = np.mean((y - pred) ** 2)
                
                vars_used = set(get_vars_used(tree))
                missing = set(mandatory) - vars_used
                penalty_missing = len(missing) * 10.0
                
                extra = vars_used - set(mandatory)
                penalty_extra = len(extra) * 0.5
                
                depth = str(tree).count('max') + str(tree).count('abs')
                reward = depth * 0.05
                
                score = mse + penalty_missing + penalty_extra - reward
                
                if np.isfinite(score):
                    fitness.append((score, tree, mse))
            except:
                pass
        
        if not fitness:
            break
        
        fitness.sort(key=lambda x: x[0])
        best = [f[1] for f in fitness[:population_size // 2]]
        
        new_population = best.copy()
        while len(new_population) < population_size:
            p1, p2 = random.choice(best), random.choice(best)
            child = crossover(p1, p2)
            child = mutate(child, features, mandatory)
            new_population.append(child)
        
        population = new_population
        
        if gen % 20 == 0:
            print(f"  نسل {gen}: بهترین MSE = {fitness[0][2]:.4f}")
    
    return fitness[0][1], fitness[0][2]

def main():
    if len(sys.argv) < 2:
        print("Usage: python genetic_engine_sensor.py <file.csv> <target_col>")
        return
    
    path = sys.argv[1]
    target_col = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    X = np.loadtxt(path, delimiter=',', skiprows=1)
    if target_col < 0: target_col = X.shape[1] + target_col
    y = X[:, target_col]
    X_feat = np.delete(X, target_col, axis=1)
    
    print(f"[*] داده: {X.shape[0]} نقطه، {X.shape[1]} ستون")
    
    features = [f'x{i}' for i in range(X_feat.shape[1])]
    mandatory = correlation_sensor(X_feat, y, threshold=0.05)
    
    print(f"\n[*] اجرای موتور ژنتیک هدایت‌شده...")
    best_tree, best_mse = genetic_programming(X_feat, y, features, mandatory)
    
    print(f"\n[✓] بهترین معادله (MSE={best_mse:.4f}):")
    print(f"    {best_tree}")
    
    n_feats = X_feat.shape[1]
    syms = sp.symbols(f"x0:{n_feats}")
    expr = tree_to_sympy(best_tree, syms)
    print(f"\n[✓] معادله ریاضی:")
    print(f"    y = {expr}")

if __name__ == "__main__":
    main()
