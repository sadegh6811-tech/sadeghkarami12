"""
کتابخانه‌ی عملگرهای گسترده برای MathBot
شامل ۲۰+ عملگر جدید با مدیریت دامنه
"""
import numpy as np

# ═══════════════════════════════════════════════════════════
# ۱. توابع پایه (با مدیریت دامنه)
# ═══════════════════════════════════════════════════════════
def safe_log(x):
    return np.log(np.abs(x) + 1e-9)

def safe_sqrt(x):
    return np.sqrt(np.abs(x) + 1e-9)

def safe_exp(x):
    return np.exp(np.clip(x, -50, 50))

def safe_div(a, b):
    return a / (b + 1e-9 * np.sign(b + 1e-30))

def safe_pow(a, b):
    return np.sign(a) * np.abs(a) ** np.clip(b, -5, 5)

def safe_asin(x):
    return np.arcsin(np.clip(x, -1, 1))

def safe_acos(x):
    return np.arccos(np.clip(x, -1, 1))

def safe_atan(x):
    return np.arctan(np.clip(x, -1e6, 1e6))

def safe_tanh(x):
    return np.tanh(np.clip(x, -20, 20))

def safe_sinh(x):
    return np.sinh(np.clip(x, -20, 20))

def safe_cosh(x):
    return np.cosh(np.clip(x, -20, 20))

def safe_erf(x):
    from scipy.special import erf
    return erf(np.clip(x, -10, 10))

def safe_gamma(x):
    from scipy.special import gamma
    return gamma(np.clip(x, 0.1, 10))

def safe_mod(a, b):
    return np.mod(a, np.abs(b) + 1e-9)

# ═══════════════════════════════════════════════════════════
# ۲. دیکشنری عملگرها
# ═══════════════════════════════════════════════════════════
UNARY_OPERATORS = {
    'abs': np.abs,
    'log': safe_log,
    'sin': np.sin,
    'cos': np.cos,
    'sqrt': safe_sqrt,
    'exp': safe_exp,
    'tanh': safe_tanh,
    'sinh': safe_sinh,
    'cosh': safe_cosh,
    'asin': safe_asin,
    'acos': safe_acos,
    'atan': safe_atan,
    'erf': safe_erf,
    'gamma': safe_gamma,
}

BINARY_OPERATORS = {
    'add': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'div': safe_div,
    'pow': safe_pow,
    'max': np.maximum,
    'min': np.minimum,
    'mod': safe_mod,
}

# ═══════════════════════════════════════════════════════════
# ۳. تابع ارزیابی
# ═══════════════════════════════════════════════════════════
def evaluate_tree_extended(tree, X):
    """ارزیابی درخت با عملگرهای گسترده"""
    if isinstance(tree, str): 
        return X[:, int(tree[1:])]
    if isinstance(tree, tuple) and tree[0] == 'const': 
        return np.full(X.shape[0], tree[1])
    
    op = tree[0]
    
    # عملگرهای یکانی
    if op in UNARY_OPERATORS:
        return UNARY_OPERATORS[op](evaluate_tree_extended(tree[1], X))
    
    # عملگرهای دودویی
    if op in BINARY_OPERATORS:
        return BINARY_OPERATORS[op](
            evaluate_tree_extended(tree[1], X),
            evaluate_tree_extended(tree[2], X)
        )
    
    return np.zeros(X.shape[0])

# ═══════════════════════════════════════════════════════════
# ۴. تست خودکار
# ═══════════════════════════════════════════════════════════
def test_operators():
    """تست همه‌ی عملگرها برای اطمینان از صحت"""
    print("═══ تست عملگرهای گسترده ═══")
    
    X = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    
    # تست عملگرهای یکانی
    for op, func in UNARY_OPERATORS.items():
        try:
            result = func(X[:, 0])
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                print(f"  ⚠️ {op}: NaN/Inf detected")
            else:
                print(f"  ✅ {op}: OK (range: [{np.min(result):.2f}, {np.max(result):.2f}])")
        except Exception as e:
            print(f"  ❌ {op}: {e}")
    
    # تست عملگرهای دودویی
    A = np.array([1.0, 2.0, 3.0])
    B = np.array([2.0, 0.0, -1.0])
    
    for op, func in BINARY_OPERATORS.items():
        try:
            result = func(A, B)
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                print(f"  ⚠️ {op}: NaN/Inf detected")
            else:
                print(f"  ✅ {op}: OK")
        except Exception as e:
            print(f"  ❌ {op}: {e}")
    
    print(f"\n[+] تعداد عملگرهای یکانی: {len(UNARY_OPERATORS)}")
    print(f"[+] تعداد عملگرهای دودویی: {len(BINARY_OPERATORS)}")
    print(f"[+] مجموع: {len(UNARY_OPERATORS) + len(BINARY_OPERATORS)}")

# ═══════════════════════════════════════════════════════════
# ۵. اجرا
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    test_operators()
