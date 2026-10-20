import re

with open('genetic_engine_golden.py', 'r') as f:
    content = f.read()

# ۱. افزایش عمق درخت از ۵ به ۶
content = content.replace('create_random_tree(5, features)', 'create_random_tree(6, features)')

# ۲. کاهش جریمه از 1e10 به 100.0 (قانون طلایی نرم‌تر)
content = content.replace('score = 1e10', 'score = 100.0 + (len(missing) * 20.0)')

# ۳. افزایش تعداد نسل به ۲۰۰ و جمعیت به ۸۰۰
content = content.replace('generations=100, population_size=500', 'generations=200, population_size=800')

with open('genetic_engine_golden.py', 'w') as f:
    f.write(content)
print("[+] اصلاحات نهایی اعمال شد:")
print("  - عمق درخت: ۶")
print("  - جریمه: 100 + (تعداد غایب × 20)")
print("  - نسل: ۲۰۰، جمعیت: ۸۰۰")
