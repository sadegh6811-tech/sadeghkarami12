with open('discover.py', 'r') as f:
    content = f.read()

# ۱. اضافه کردن abs_library به import
old_imp = 'from extended_lib import (exp_library, log_library,\n                                       inverse_library, inverse_square_library,\n                                       sqrt_library)'
new_imp = 'from extended_lib import (exp_library, log_library,\n                                       inverse_library, inverse_square_library,\n                                       sqrt_library, abs_library)'
content = content.replace(old_imp, new_imp)

# ۲. اضافه کردن abs به _build_library
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

# ۳. اضافه کردن تولید عبارت abs در معادله
old_expr = '''                elif kind == "sqrt":
                    expr += sp.Float(c) * sp.sqrt(xi)'''
new_expr = '''                elif kind == "sqrt":
                    expr += sp.Float(c) * sp.sqrt(xi)
                elif kind == "abs":
                    expr += sp.Float(c) * sp.Abs(xi)'''
content = content.replace(old_expr, new_expr)

with open('discover.py', 'w') as f:
    f.write(content)
print("[+] discover.py برای پشتیبانی از |x| و اولویت‌دهی به آن به‌روزرسانی شد.")
