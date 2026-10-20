import numpy as np
import random
from genetic_engine_classifier import evaluate_tree, random_tree, mutate, crossover, tree_depth

def genetic_programming(X, y, features, generations=100, pop_size=500):
    population = [random_tree(5, features) for _ in range(pop_size)]
    best_overall, best_acc = None, 0.0
    
    for gen in range(generations):
        fitness = []
        for tree in population:
            try:
                pred = evaluate_tree(tree, X)
                pred_class = np.zeros_like(pred)
                pred_class[pred > 0.1] = 1
                pred_class[pred < -0.1] = -1
                accuracy = np.mean(pred_class == y)
                size_penalty = 0.001 * tree_depth(tree)
                fitness.append((accuracy - size_penalty, accuracy, tree))
            except: pass
        if not fitness: break
        fitness.sort(key=lambda x: x[0], reverse=True)
        if fitness[0][1] > best_acc:
            best_acc = fitness[0][1]
            best_overall = fitness[0][2]
        top = [f[2] for f in fitness[:pop_size//2]]
        new_pop = top.copy()
        while len(new_pop) < pop_size:
            p1, p2 = random.choice(top), random.choice(top)
            new_pop.append(mutate(crossover(p1, p2), features))
        population = new_pop
    return best_overall, best_acc

# خواندن داده
data = np.loadtxt('/data/data/com.termux/files/home/forex_topology.csv', delimiter=',', skiprows=1)
X, y = data[:, :-1], data[:, -1]
features = [f'x{i}' for i in range(X.shape[1])]

# ۱. آموزش روی ۸۰٪
n_train = int(len(X) * 0.8)
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

print(f"[*] آموزش: {len(X_train)}، تست: {len(X_test)}")
print("[*] در حال آموزش GP...")

best_tree, train_acc = genetic_programming(X_train, y_train, features)
print(f"\n[+] دقت آموزش: {train_acc*100:.2f}%")
print(f"\n[+] 🌟 درخت کشف‌شده:\n{best_tree}")

# ۲. ارزیابی روی تست
pred_test = evaluate_tree(best_tree, X_test)
pred_class_test = np.zeros_like(pred_test)
pred_class_test[pred_test > 0.1] = 1
pred_class_test[pred_test < -0.1] = -1
test_acc = np.mean(pred_class_test == y_test)

print(f"\n[+] دقت تست: {test_acc*100:.2f}%")
print(f"[+] اختلاف: {(train_acc - test_acc)*100:.2f}%")

# ۳. ماتریس درهم‌ریختگی
print(f"\n[+] ماتریس درهم‌ریختگی (تست):")
print(f"    پیش‌بینی صعودی درست: {np.sum((pred_class_test==1) & (y_test==1))}")
print(f"    پیش‌بینی نزولی درست:  {np.sum((pred_class_test==-1) & (y_test==-1))}")
print(f"    کل صعودی‌های واقعی:   {np.sum(y_test==1)}")
print(f"    کل نزولی‌های واقعی:   {np.sum(y_test==-1)}")
