"""موتور ژنتیک تقویت‌شده (Crossover + افزایش نسل + جریمه هوشمند)."""
import sys
import numpy as np
import random
import sympy as sp

def evaluate_node(node, X):
    if isinstance(node, str):
        idx = int(node[1:])
        return X[:, idx]
    op = node[0]
    if op == 'abs':
        return np.abs(evaluate_node(node[1], X))
    elif op == 'max':
        return np.maximum(evaluate_node(node[1], X), evaluate_node(node[2], X))

def tree_to_sympy(node, syms):
    if isinstance(node, str):
        idx = int(node[1:])
        return syms[idx]
    op = node[0]
    if op == 'abs':
        return sp.Abs(tree_to_sympy(node[1], syms))
    elif op == 'max':
        return sp.Max(tree_to_sympy(node[1], syms), tree_to_sympy(node[2], syms))

def get_vars_used(node):
    if isinstance(node, str):
        return [node]
    result = []
    for child in node[1:]:
        if isinstance(child, tuple):
            result.extend(get_vars_used(child))
        elif isinstance(child, str):
            result.append(child)
    return result

def create_random_tree(depth, features):
    if depth == 0:
        return str(np.random.choice(features))
    op = random.choice(['abs', 'max'])
    if op == 'abs':
        return ('abs', create_random_tree(depth-1, features))
    elif op == 'max':
        return ('max', create_random_tree(depth-1, features), create_random_tree(depth-1, features))

def crossover(parent1, parent2):
    """ترکیب دو والد (Crossover) - نسخه امن بدون خطا."""
    # اگر یکی از والدین رشته (متغیر) است
    if isinstance(parent1, str) and isinstance(parent2, str):
        return random.choice([parent1, parent2])
    if isinstance(parent1, str):
        return parent2
    if isinstance(parent2, str):
        return parent1
    
    # هر دو tuple هستند
    op1, op2 = parent1[0], parent2[0]
    
    # اگر یکی abs است و دیگری max، یکی را تصادفی انتخاب کن
    if op1 != op2:
        return random.choice([parent1, parent2])
    
    if op1 == 'abs':
        # هر دو abs دارند، فرزند از ترکیب فرزندانشان
        return ('abs', random.choice([parent1[1], parent2[1]]))
    elif op1 == 'max':
        # هر دو max دارند (دارای ۳ عضو هستند)
        return ('max', 
                random.choice([parent1[1], parent2[1]]),
                random.choice([parent1[2], parent2[2]]))
    
    return parent1

def genetic_programming(X, y, features, generations=50, population_size=300):
    # جمعیت اولیه
    population = [create_random_tree(4, features) for _ in range(population_size)]
    
    for gen in range(generations):
        fitness = []
        for tree in population:
            try:
                pred = evaluate_node(tree, X)
                mse = np.mean((y - pred) ** 2)
                
                # جریمه هوشمند: استفاده از همه متغیرها + عمق درخت
                vars_used = set(get_vars_used(tree))
                penalty_vars = (len(features) - len(vars_used)) * 5.0  # جریمه شدید
                
                # پاداش برای استفاده از max تو در تو (ساختار مکعب)
                depth = str(tree).count('max')
                reward_depth = depth * 0.1
                
                score = mse + penalty_vars - reward_depth
                
                if np.isfinite(score):
                    fitness.append((score, tree, mse))
            except:
                pass
        
        if not fitness:
            break
        
        fitness.sort(key=lambda x: x[0])
        best = [f[1] for f in fitness[:population_size // 2]]
        
        # تولید نسل جدید با Crossover
        new_population = best.copy()
        while len(new_population) < population_size:
            parent1 = random.choice(best)
            parent2 = random.choice(best)
            child = crossover(parent1, parent2)
            new_population.append(child)
        
        population = new_population
        
        # چاپ پیشرفت
        if gen % 10 == 0:
            print(f"  نسل {gen}: بهترین MSE = {fitness[0][2]:.4f}")
    
    return fitness[0][1], fitness[0][2]

def main():
    if len(sys.argv) < 2:
        print("Usage: python genetic_engine_boosted.py <file.csv> <target_col>")
        return
    
    path = sys.argv[1]
    target_col = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    
    X = np.loadtxt(path, delimiter=',', skiprows=1)
    if target_col < 0:
        target_col = X.shape[1] + target_col
    
    y = X[:, target_col]
    X_feat = np.delete(X, target_col, axis=1)
    
    print(f"[*] داده: {X.shape[0]} نقطه، {X.shape[1]} ستون")
    print("[*] اجرای موتور ژنتیک تقویت‌شده (Crossover + نسل بالا)...")
    
    features = [f'x{i}' for i in range(X_feat.shape[1])]
    best_tree, best_mse = genetic_programming(X_feat, y, features, generations=50, population_size=300)
    
    print(f"\n[✓] بهترین معادله (MSE={best_mse:.4f}):")
    print(f"    {best_tree}")
    
    n_feats = X_feat.shape[1]
    syms = sp.symbols(f"x0:{n_feats}")
    expr = tree_to_sympy(best_tree, syms)
    print(f"\n[✓] معادله ریاضی:")
    print(f"    y = {expr}")

if __name__ == "__main__":
    main()
