with open('extended_lib.py', 'r') as f:
    content = f.read()

new_funcs = '''

def abs_library(X):
    """|x| — قدر مطلق، برای کشف مرزهای هایپرکیوب."""
    n, d = X.shape
    cols, specs = [], []
    for i in range(d):
        cols.append(np.abs(X[:, i]))
        specs.append(("abs", i, 0.0))
    return np.column_stack(cols), specs


def max_abs_library(X, pairs):
    """max(|xi|, |xj|) — برای کشف هایپرکیوب‌های چندبعدی."""
    n, d = X.shape
    cols, specs = [], []
    for i, j in pairs:
        val = np.maximum(np.abs(X[:, i]), np.abs(X[:, j]))
        cols.append(val)
        specs.append(("maxabs", i, j))
    return np.column_stack(cols), specs
'''

if "def abs_library" not in content:
    content += new_funcs
    with open('extended_lib.py', 'w') as f:
        f.write(content)
    print("[+] abs_library و max_abs_library به extended_lib.py اضافه شدند.")
else:
    print("[i] توابع abs از قبل وجود دارند.")

# اضافه کردن maxabs به تولید معادله در discover.py
with open('discover.py', 'r') as f:
    content = f.read()

# اضافه کردن abs به import
old_imp = 'from extended_lib import (exp_library, log_library,\n                                       inverse_library, inverse_square_library,\n                                       sqrt_library)'
new_imp = 'from extended_lib import (exp_library, log_library,\n                                       inverse_library, inverse_square_library,\n                                       sqrt_library, abs_library)'
content = content.replace(old_imp, new_imp)

# اضافه کردن abs به _build_library
old_build = '''            Inv2, inv2_specs = inverse_square_library(Xn)
            Sqrt, sqrt_specs = sqrt_library(Xn)
            
            for arr, specs in [(Log, log_specs), (Inv, inv_specs), 
                                (Inv2, inv2_specs), (Sqrt, sqrt_specs)]:'''
new_build = '''            Inv2, inv2_specs = inverse_square_library(Xn)
            Sqrt, sqrt_specs = sqrt_library(Xn)
            Abs, abs_specs = abs_library(Xn)
            
            for arr, specs in [(Log, log_specs), (Inv, inv_specs), 
                                (Inv2, inv2_specs), (Sqrt, sqrt_specs), (Abs, abs_specs)]:'''
content = content.replace(old_build, new_build)

# اضافه کردن abs به تولید معادله
old_expr = '''                elif kind == "sqrt":
                    expr += sp.Float(c) * sp.sqrt(xi)'''
new_expr = '''                elif kind == "sqrt":
                    expr += sp.Float(c) * sp.sqrt(xi)
                elif kind == "abs":
                    expr += sp.Float(c) * sp.Abs(xi)'''
content = content.replace(old_expr, new_expr)

with open('discover.py', 'w') as f:
    f.write(content)
print("[+] discover.py برای پشتیبانی از |x| به‌روزرسانی شد.")
