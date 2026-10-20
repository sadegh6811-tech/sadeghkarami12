with open('genetic_engine_golden.py', 'r') as f:
    content = f.read()

# ۱. اضافه کردن مقدار اولیه به valid_best قبل از حلقه
old_line = '''        valid_best = [f for f in fitness if f[0] < 1e9]'''
new_line = '''        valid_best = [f for f in fitness if f[0] < 1e9] if fitness else []'''

# جایگزینی
content = content.replace(old_line, new_line)

# ۲. اصلاح خط return
old_return = '''    return valid_best[0][1], valid_best[0][2] if valid_best else (None, float('inf'))'''
new_return = '''    if valid_best:
        return valid_best[0][1], valid_best[0][2]
    else:
        return None, float('inf')'''

content = content.replace(old_return, new_return)

with open('genetic_engine_golden.py', 'w') as f:
    f.write(content)
print("[+] خطای UnboundLocalError برطرف شد.")
