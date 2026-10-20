"""
کتابخانه‌ی عملگرهای گسترده v2 - شامل توابع خاص فیزیک
"""
import numpy as np
from scipy.special import gamma, erf, zeta, jv, legendre

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
    return erf(np.clip(x, -10, 10))

def safe_gamma(x):
    return gamma(np.clip(x, 0.1, 10))

def safe_mod(a, b):
    return np.mod(a, np.abs(b) + 1e-9)

def safe_bessel(x):
    """تابع بسل مرتبه‌ی صفر (J0)"""
    return jv(0, np.clip(x, -100, 100))

def safe_legendre(x):
    """چندجمله‌ای لژاندر مرتبه‌ی 2 (P2)"""
    return legendre(2)(np.clip(x, -1, 1))

def safe_zeta(x):
    """تابع زتای ریمان (فقط برای x > 1)"""
    return zeta(np.clip(x, 1.1, 10))

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
    'bessel': safe_bessel,
    'legendre': safe_legendre,
    'zeta': safe_zeta,
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
def evaluate_tree_extended_v2(tree, X):
    if isinstance(tree, str): 
        return X[:, int(tree[1:])]
    if isinstance(tree, tuple) and tree[0] == 'const': 
        return np.full(X.shape[0], tree[1])
    
    op = tree[0]
    if op in UNARY_OPERATORS:
        return UNARY_OPERATORS[op](evaluate_tree_extended_v2(tree[1], X))
    if op in BINARY_OPERATORS:
        return BINARY_OPERATORS[op](
            evaluate_tree_extended_v2(tree[1], X),
            evaluate_tree_extended_v2(tree[2], X)
        )
    return np.zeros(X.shape[0])

# ═══════════════════════════════════════════════════════════
# ۴. تست خودکار
# ═══════════════════════════════════════════════════════════
def test_operators():
    print("═══ تست عملگرهای گسترده v2 ═══")
    X = np.array([[-2.0], [-1.0], [0.5], [1.0], [2.0]])
    
    for op, func in UNARY_OPERATORS.items():
        try:
            result = func(X[:, 0])
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                print(f"  ⚠️ {op}: NaN/Inf")
            else:
                print(f"  ✅ {op}: OK (range: [{np.min(result):.2f}, {np.max(result):.2f}])")
        except Exception as e:
            print(f"  ❌ {op}: {e}")
    
    A = np.array([1.0, 2.0, 3.0])
    B = np.array([2.0, 0.5, -1.0])
    for op, func in BINARY_OPERATORS.items():
        try:
            result = func(A, B)
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                print(f"  ⚠️ {op}: NaN/Inf")
            else:
                print(f"  ✅ {op}: OK")
        except Exception as e:
            print(f"  ❌ {op}: {e}")
    
    print(f"\n[+] عملگرهای یکانی: {len(UNARY_OPERATORS)}")
    print(f"[+] عملگرهای دودویی: {len(BINARY_OPERATORS)}")
    print(f"[+] مجموع: {len(UNARY_OPERATORS) + len(BINARY_OPERATORS)}")

if __name__ == "__main__":
    test_operators()
