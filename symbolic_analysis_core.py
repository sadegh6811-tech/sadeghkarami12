"""
هسته چهارم: تحلیل نمادین
- تشخیص ساختار اولیه از داده‌ها
- ساخت Seed Tree هوشمند
"""
import numpy as np

def analyze_structure(X, y):
    """
    تحلیل ساختار داده‌ها و پیشنهاد Seed Tree.
    """
    n, d = X.shape
    suggestions = []
    
    # ۱. تحلیل همبستگی ساده
    for i in range(d):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        if abs(corr) > 0.3:
            suggestions.append(('linear', i, corr))
    
    # ۲. تحلیل همبستگی با توان‌های ۲، ۳
    for i in range(d):
        for power in [2, 3]:
            corr = np.corrcoef(X[:, i]**power, y)[0, 1]
            if abs(corr) > 0.5:
                suggestions.append((f'power{power}', i, corr))
    
    # ۳. تحلیل همبستگی با exp, log
    for i in range(d):
        # exp
        exp_x = np.exp(np.clip(X[:, i], -5, 5))
        corr = np.corrcoef(exp_x, y)[0, 1]
        if abs(corr) > 0.5:
            suggestions.append(('exp', i, corr))
        
        # log
        log_x = np.log(np.abs(X[:, i]) + 1e-6)
        corr = np.corrcoef(log_x, y)[0, 1]
        if abs(corr) > 0.5:
            suggestions.append(('log', i, corr))
    
    # مرتب‌سازی بر اساس قدرت همبستگی
    suggestions.sort(key=lambda x: abs(x[2]), reverse=True)
    
    return suggestions

def build_seed_tree(suggestions, features, max_depth=3):
    """
    ساخت Seed Tree بر اساس پیشنهادات.
    """
    if not suggestions:
        return None
    
    # انتخاب بهترین پیشنهاد
    best = suggestions[0]
    kind, idx, corr = best
    feature = features[idx]
    
    if kind == 'linear':
        return feature
    elif kind == 'power2':
        return ('mul', feature, feature)
    elif kind == 'power3':
        return ('mul', ('mul', feature, feature), feature)
    elif kind == 'exp':
        return ('exp', feature)
    elif kind == 'log':
        return ('log', feature)
    
    return feature

if __name__ == "__main__":
    # تست روی معادلات Feynman
    print("═══ تست هسته چهارم (تحلیل نمادین) ═══\n")
    
    equations = [
        {'name': 'I.8.14', 'formula': lambda x: np.sqrt((x[:,0]-x[:,1])**2 + (x[:,2]-x[:,3])**2), 'n': 4},
        {'name': 'I.9.18', 'formula': lambda x: (x[:,0]*x[:,1])/((x[:,2]-x[:,3])**2 + 1e-6), 'n': 4},
        {'name': 'II.2.42', 'formula': lambda x: x[:,0]/(x[:,1]**2 + 1e-6), 'n': 2},
    ]
    
    for eq in equations:
        np.random.seed(42)
        X = np.random.uniform(0.1, 2.0, (300, eq['n']))
        y = eq['formula'](X)
        X_norm = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-9)
        
        features = [f'x{i}' for i in range(eq['n'])]
        suggestions = analyze_structure(X_norm, y)
        seed = build_seed_tree(suggestions, features)
        
        print(f"معادله {eq['name']}:")
        print(f"  پیشنهادات: {[(s[0], s[1], f'{s[2]:.2f}') for s in suggestions[:3]]}")
        print(f"  Seed Tree: {seed}")
        print()
