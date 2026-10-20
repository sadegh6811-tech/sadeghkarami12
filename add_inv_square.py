"""افزودن 1/x² به کتابخانه و discover.py"""

# ۱. اضافه کردن تابع به extended_lib.py
with open('extended_lib.py', 'r') as f:
    content = f.read()

new_func = '''

def inverse_square_library(X, offset=0.1):
    """1/x² — برای کشف معادلات گرانش و کولن."""
    n, d = X.shape
    cols, specs = [], []
    for i in range(d):
        xi = np.where(np.abs(X[:, i]) < offset, offset, X[:, i])
        cols.append(1.0 / (xi ** 2))
        specs.append(("inv2", i, 0.0))
    return np.column_stack(cols), specs
'''

if "def inverse_square_library" not in content:
    content += new_func
    with open('extended_lib.py', 'w') as f:
        f.write(content)
    print("[+] inverse_square_library به extended_lib.py اضافه شد")
else:
    print("[i] inverse_square_library از قبل وجود دارد")

# ۲. به‌روزرسانی discover.py
with open('discover.py', 'r') as f:
    content = f.read()

# اضافه کردن به _build_library
old_build = '''            Log, log_specs = log_library(Xn)
            Inv, inv_specs = inverse_library(Xn)
            Hyp, hyp_specs = hyperbolic_library(Xn)
            Sqrt, sqrt_specs = sqrt_library(Xn)'''

new_build = '''            Log, log_specs = log_library(Xn)
            Inv, inv_specs = inverse_library(Xn)
            Inv2, inv2_specs = inverse_square_library(Xn)
            Hyp, hyp_specs = hyperbolic_library(Xn)
            Sqrt, sqrt_specs = sqrt_library(Xn)'''

content = content.replace(old_build, new_build)

old_list = '''            for arr, specs in [(Log, log_specs), (Inv, inv_specs), (Hyp, hyp_specs), (Sqrt, sqrt_specs)]:'''

new_list = '''            for arr, specs in [(Log, log_specs), (Inv, inv_specs), (Inv2, inv2_specs), (Hyp, hyp_specs), (Sqrt, sqrt_specs)]:'''

content = content.replace(old_list, new_list)

# اضافه کردن به import
old_import = 'from extended_lib import (exp_library, log_library,\n                                       inverse_library, hyperbolic_library, sqrt_library)'
new_import = 'from extended_lib import (exp_library, log_library,\n                                       inverse_library, inverse_square_library, hyperbolic_library, sqrt_library)'
content = content.replace(old_import, new_import)

# اضافه کردن به تولید معادله
old_expr = '''                elif kind == "inv":
                    expr += sp.Float(c) / xi'''
new_expr = '''                elif kind == "inv":
                    expr += sp.Float(c) / xi
                elif kind == "inv2":
                    expr += sp.Float(c) / (xi ** 2)'''
content = content.replace(old_expr, new_expr)

with open('discover.py', 'w') as f:
    f.write(content)
print("[+] discover.py برای پشتیبانی از 1/x² به‌روزرسانی شد")
