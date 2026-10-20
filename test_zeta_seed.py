import numpy as np
import sys
sys.path.insert(0, '/data/data/com.termux/files/home/mathbot')
from unified_engine import evaluate_tree, tree_to_sympy, tree_depth
import sympy as sp

# داده‌ها
data = np.loadtxt('/data/data/com.termux/files/home/zeta_data.csv', delimiter=',', skiprows=1)
X = data[:, :3]  # index, real, imag
y = data[:, 3]   # gap

# Seed Tree برای ریمان-فون مانگولت: gap ≈ 2π / log(T/2π)
# ما از x2 (imag) به عنوان T استفاده می‌کنیم
# ساخت درخت: 2π / log(x2 / 2π)
# اما درخت ما فقط با اعداد ثابت کار می‌کند، پس تقریب ساده‌تر: 6.28 / log(x2 / 6.28)

def seed_tree_riemann():
    # 2π ≈ 6.28
    two_pi = ('const', 6.283185307)
    # log(x2 / 2π) = log(x2) - log(2π)
    # ما نمی‌توانیم log(2π) را دقیق بسازیم، پس از یک تقریب استفاده می‌کنیم
    # gap ≈ 2π / log(T)
    return ('div', two_pi, ('log', 'x2'))

seed = seed_tree_riemann()
pred = evaluate_tree(seed, X)

print("═══ نتیجه Seed Tree ریمان-فون مانگولت ═══")
print(f"MSE: {np.mean((y - pred)**2):.6f}")

syms = sp.symbols("x0:3")
expr = tree_to_sympy(seed, syms)
print(f"معادله: y = {expr}")

# حالا با GP سعی می‌کنیم این را بهبود دهیم
from unified_engine import genetic_programming, random_tree, mutate, crossover
import random

# جمعیت اولیه با Seed
features = ['x0', 'x1', 'x2']
pop_size = 500

# تغییر موتور GP برای شروع با Seed
# (در نسخه فعلی، genetic_programming فقط seed_mode را پشتیبانی می‌کند)
# ما یک نسخه سفارشی می‌سازیم
print("\n[*] اجرای GP با Seed Tree ریمان...")

population = [seed] + [random_tree(4, features) for _ in range(pop_size - 1)]
best_mse = np.mean((y - pred)**2)
best_tree = seed

for gen in range(100):
    fitness = []
    for tree in population:
        try:
            p = evaluate_tree(tree, X)
            if not np.all(np.isfinite(p)): continue
            mse = np.mean((y - p)**2)
            fitness.append((mse, tree))
        except: pass
    
    if not fitness: break
    fitness.sort(key=lambda x: x[0])
    
    if fitness[0][0] < best_mse:
        best_mse = fitness[0][0]
        best_tree = fitness[0][1]
    
    top = [f[1] for f in fitness[:pop_size//2]]
    new_pop = top.copy()
    while len(new_pop) < pop_size:
        p1, p2 = random.choice(top), random.choice(top)
        new_pop.append(mutate(crossover(p1, p2), features))
    population = new_pop
    
    if gen % 20 == 0:
        print(f"  نسل {gen}: بهترین MSE = {best_mse:.6f}")

print(f"\n[✓] بهترین MSE: {best_mse:.6f}")
expr = tree_to_sympy(best_tree, syms)
print(f"[✓] معادله: y = {expr}")
print(f"[✓] عمق: {tree_depth(best_tree)}")
