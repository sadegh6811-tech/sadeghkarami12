import re

with open('genetic_engine_golden.py', 'r') as f:
    content = f.read()

# پیدا کردن کل تابع genetic_programming و جایگزینی آن
pattern = r'def genetic_programming\(X, y, features, generations=100, population_size=500\):.*?return valid_best\[0\]\[1\], valid_best\[0\]\[2\] if valid_best else \(None, float\(\'inf\'\)\)'

new_func = '''def genetic_programming(X, y, features, generations=100, population_size=500):
    population = [create_random_tree(5, features) for _ in range(population_size)]
    
    best_overall_tree = None
    best_overall_mse = float('inf')
    
    for gen in range(generations):
        fitness = []
        for tree in population:
            try:
                vars_used = set(get_vars_used(tree))
                missing = set(features) - vars_used
                
                # قانون طلایی: جریمه مرگبار برای متغیرهای غایب
                if missing:
                    score = 1e10
                    mse = float('inf')
                else:
                    pred = evaluate_node(tree, X)
                    mse = np.mean((y - pred) ** 2)
                    depth = str(tree).count('max') + str(tree).count('abs')
                    score = mse - depth * 0.01
                    
                if np.isfinite(score):
                    fitness.append((score, tree, mse))
            except:
                pass
        
        if not fitness:
            break
            
        fitness.sort(key=lambda x: x[0])
        
        # پیدا کردن بهترین درخت معتبر
        valid_trees = [f for f in fitness if f[2] != float('inf')]
        
        if valid_trees:
            current_best_mse = valid_trees[0][2]
            if current_best_mse < best_overall_mse:
                best_overall_mse = current_best_mse
                best_overall_tree = valid_trees[0][1]
        
        # اگر هیچ درخت معتبری نبود، به نسل بعد برو
        if not valid_trees:
            # ادامه با بهترین‌های نامعتبر برای بقا
            best = [f[1] for f in fitness[:population_size // 2]]
        else:
            best = [f[1] for f in valid_trees[:population_size // 2]]
            
        new_population = best.copy()
        while len(new_population) < population_size:
            p1, p2 = random.choice(best), random.choice(best)
            child = mutate(crossover(p1, p2), features)
            new_population.append(child)
        
        population = new_population
        
        if gen % 20 == 0:
            print(f"  نسل {gen}: بهترین MSE = {best_overall_mse if best_overall_tree else 'N/A'} (تعداد معتبر: {len(valid_trees)})")
    
    if best_overall_tree:
        return best_overall_tree, best_overall_mse
    else:
        return None, float('inf')'''

content = re.sub(pattern, new_func, content, flags=re.DOTALL)

with open('genetic_engine_golden.py', 'w') as f:
    f.write(content)
print("[+] تابع genetic_programming با موفقیت بازنویسی شد.")
