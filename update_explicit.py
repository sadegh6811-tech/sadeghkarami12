"""به‌روزرسانی mathbot_explicit.py برای پشتیبانی از 1/x²"""
with open('mathbot_explicit.py', 'r') as f:
    content = f.read()

# اضافه کردن به extended
old = '''        from extended_lib import sqrt_library, log_library, inverse_library, exp_library
        Sqrt, sqrt_specs = sqrt_library(Xn)
        Log, log_specs = log_library(Xn)
        Inv, inv_specs = inverse_library(Xn)
        Exp, exp_specs = exp_library(Xn)
        
        for arr, specs in [(Sqrt, sqrt_specs), (Log, log_specs), (Inv, inv_specs), (Exp, exp_specs)]:'''

new = '''        from extended_lib import sqrt_library, log_library, inverse_library, inverse_square_library, exp_library
        Sqrt, sqrt_specs = sqrt_library(Xn)
        Log, log_specs = log_library(Xn)
        Inv, inv_specs = inverse_library(Xn)
        Inv2, inv2_specs = inverse_square_library(Xn)
        Exp, exp_specs = exp_library(Xn)
        
        for arr, specs in [(Sqrt, sqrt_specs), (Log, log_specs), (Inv, inv_specs), (Inv2, inv2_specs), (Exp, exp_specs)]:'''

content = content.replace(old, new)

# اضافه کردن به تولید معادله
old_expr = '''        elif kind == "inv":
            expr += sp.Float(c) / xi
        elif kind == "exp":'''
new_expr = '''        elif kind == "inv":
            expr += sp.Float(c) / xi
        elif kind == "inv2":
            expr += sp.Float(c) / (xi ** 2)
        elif kind == "exp":'''
content = content.replace(old_expr, new_expr)

with open('mathbot_explicit.py', 'w') as f:
    f.write(content)
print("[+] mathbot_explicit.py برای پشتیبانی از 1/x² به‌روزرسانی شد")
