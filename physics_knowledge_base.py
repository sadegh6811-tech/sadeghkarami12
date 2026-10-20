"""
هسته چهارم: کتابخانه دانش فیزیک و کوانتوم
"""
import numpy as np

# ═══════════════════════════════════════════════════════════
# ۱. الگوهای فیزیک
# ═══════════════════════════════════════════════════════════
PHYSICS_PATTERNS = {
    'gravity': {
        'template': lambda v: ('div', ('mul', v[0], v[1]), ('mul', ('sub', v[2], v[3]), ('sub', v[2], v[3]))),
        'vars': 4, 'desc': 'گرانش نیوتن'
    },
    'coulomb': {
        'template': lambda v: ('div', ('mul', v[0], v[1]), ('mul', v[2], v[2])),
        'vars': 3, 'desc': 'کولن'
    },
    'distance': {
        'template': lambda v: ('sqrt', ('add', ('mul', ('sub', v[0], v[1]), ('sub', v[0], v[1])), ('mul', ('sub', v[2], v[3]), ('sub', v[2], v[3])))),
        'vars': 4, 'desc': 'فاصله اقلیدسی'
    },
    'kinetic': {
        'template': lambda v: ('mul', ('const', 0.5), ('mul', v[0], ('mul', v[1], v[1]))),
        'vars': 2, 'desc': 'انرژی جنبشی'
    },
    'potential': {
        'template': lambda v: ('mul', ('mul', v[0], v[1]), v[2]),
        'vars': 3, 'desc': 'انرژی پتانسیل'
    },
    'gaussian': {
        'template': lambda v: ('div', ('exp', ('div', ('mul', ('const', -1.0), ('mul', v[0], v[0])), ('mul', ('const', 2.0), ('mul', v[1], v[1])))), ('mul', ('const', 2.5), v[1])),
        'vars': 2, 'desc': 'توزیع گاوسی'
    },
    'photon': {
        'template': lambda v: ('mul', v[0], v[1]),
        'vars': 2, 'desc': 'فوتون (E=hf)'
    },
    'relativity': {
        'template': lambda v: ('mul', ('mul', v[0], v[1]), ('mul', ('const', 3e8), ('const', 3e8))),
        'vars': 2, 'desc': 'نسبیت (E=mc²)'
    },
    'schrodinger': {
        'template': lambda v: ('div', ('mul', ('mul', v[0], v[0]), ('const', 9.87)), ('mul', v[1], ('mul', v[2], v[2]))),
        'vars': 3, 'desc': 'شرودینگر'
    },
    'harmonic': {
        'template': lambda v: ('mul', ('sub', v[0], ('const', 0.5)), v[1]),
        'vars': 2, 'desc': 'نوسانگر هارمونیک'
    },
    'boltzmann': {
        'template': lambda v: ('exp', ('div', ('const', -1.0), v[0])),
        'vars': 1, 'desc': 'بولتزمان'
    },
    'tunneling': {
        'template': lambda v: ('exp', ('mul', ('const', -2.0), v[0])),
        'vars': 1, 'desc': 'تونل‌زنی'
    },
}

# ═══════════════════════════════════════════════════════════
# ۲. ارزیابی الگو
# ═══════════════════════════════════════════════════════════
def evaluate_pattern(pattern, X):
    def eval_tree(tree):
        if isinstance(tree, str): return X[:, int(tree[1:])]
        if isinstance(tree, tuple) and tree[0] == 'const': return np.full(X.shape[0], tree[1])
        op = tree[0]
        if op == 'abs': return np.abs(eval_tree(tree[1]))
        if op == 'add': return eval_tree(tree[1]) + eval_tree(tree[2])
        if op == 'sub': return eval_tree(tree[1]) - eval_tree(tree[2])
        if op == 'mul': return eval_tree(tree[1]) * eval_tree(tree[2])
        if op == 'div': return eval_tree(tree[1]) / (eval_tree(tree[2]) + 1e-9)
        if op == 'sqrt': return np.sqrt(np.abs(eval_tree(tree[1])) + 1e-9)
        if op == 'exp': return np.exp(np.clip(eval_tree(tree[1]), -20, 20))
        if op == 'sin': return np.sin(eval_tree(tree[1]))
        if op == 'pow': return np.sign(eval_tree(tree[1])) * np.abs(eval_tree(tree[1])) ** np.clip(eval_tree(tree[2]), -5, 5)
        return np.zeros(X.shape[0])
    
    try:
        return eval_tree(pattern)
    except:
        return None

# ═══════════════════════════════════════════════════════════
# ۳. تشخیص قانون فیزیک
# ═══════════════════════════════════════════════════════════
def detect_physics_law(X, y):
    n, d = X.shape
    best_match = None
    best_corr = 0.0
    
    for name, pattern in PHYSICS_PATTERNS.items():
        if pattern['vars'] != d:
            continue
        
        features = [f'x{i}' for i in range(d)]
        
        try:
            tree = pattern['template'](features)
            pred = evaluate_pattern(tree, X)
            
            if pred is None or not np.all(np.isfinite(pred)):
                continue
            
            corr = abs(np.corrcoef(pred, y)[0, 1])
            if corr > best_corr:
                best_corr = corr
                best_match = (name, tree, corr, pattern['desc'])
        except:
            pass
    
    return best_match
