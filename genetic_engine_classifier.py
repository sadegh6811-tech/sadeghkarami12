"""موتور GP Classifier - برای مدل توپولوژیک بازار"""
import sys
import numpy as np
import random

def evaluate_tree(tree, X):
    if isinstance(tree, str):
        return X[:, int(tree[1:])]
    if isinstance(tree, tuple) and tree[0] == 'const':
        return np.full(X.shape[0], tree[1])
    op = tree[0]
    if op == 'abs': return np.abs(evaluate_tree(tree[1], X))
    if op == 'add': return evaluate_tree(tree[1], X) + evaluate_tree(tree[2], X)
    if op == 'sub': return evaluate_tree(tree[1], X) - evaluate_tree(tree[2], X)
    if op == 'mul': return evaluate_tree(tree[1], X) * evaluate_tree(tree[2], X)
    if op == 'div': return evaluate_tree(tree[1], X) / (evaluate_tree(tree[2], X) + 1e-6)
    return np.maximum(evaluate_tree(tree[1], X), evaluate_tree(tree[2], X))

def random_tree(depth, features):
    if depth <= 0:
        if random.random() < 0.3:
            return ('const', random.choice([-1.0, -0.5, 0.0, 0.5, 1.0]))
        return random.choice(features)
    op = random.choice(['add', 'sub', 'mul', 'div', 'abs', 'max'])
    if op == 'abs':
        return ('abs', random_tree(depth-1, features))
    return (op, random_tree(depth-1, features), random_tree(depth-1, features))

def tree_depth(tree):
    if isinstance(tree, str) or (isinstance(tree, tuple) and tree[0] == 'const'):
        return 0
    return 1 + max(tree_depth(c) for c in tree[1:])

def mutate(tree, features, prob=0.2):
    if random.random() < prob:
        return random_tree(3, features)
    if isinstance(tree, str) or (isinstance(tree, tuple) and tree[0] == 'const'):
        return tree
    if tree[0] == 'abs':
        return ('abs', mutate(tree[1], features, prob))
    return (tree[0], mutate(tree[1], features, prob), mutate(tree[2], features, prob))

def crossover(t1, t2):
    if isinstance(t1, str) or isinstance(t2, str) or t1[0]=='const' or t2[0]=='const':
        return random.choice([t1, t2])
    if t1[0] != t2[0]:
        return random.choice([t1, t2])
    if t1[0] == 'abs':
        return ('abs', random.choice([t1[1], t2[1]]))
    return (t1[0], random.choice([t1[1], t2[1]]), random.choice([t1[2], t2[2]]))

def genetic_programming(X, y, features, generations=100, pop_size=500):
    print(f"[*] GP Classifier - هدف: طبقه‌بندی گسسته")
    population = [random_tree(5, features) for _ in range(pop_size)]
    best_overall = None
    best_acc = 0.0
    
    for gen in range(generations):
        fitness = []
        for tree in population:
            try:
                pred = evaluate_tree(tree, X)
                # تبدیل پیوسته به کلاس با آستانه 0.1
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
            child = mutate(crossover(p1, p2), features)
            new_pop.append(child)
        population = new_pop
        
        if gen % 20 == 0:
            print(f"  نسل {gen}: بهترین دقت = {best_acc*100:.2f}%")
    
    return best_overall, best_acc

if __name__ == "__main__":
    data = np.loadtxt(sys.argv[1], delimiter=',', skiprows=1)
    X, y = data[:, :-1], data[:, -1]
    features = [f'x{i}' for i in range(X.shape[1])]
    best_tree, best_acc = genetic_programming(X, y, features, generations=100, pop_size=500)
    print(f"\n[+] بهترین دقت طبقه‌بندی: {best_acc*100:.2f}%")
