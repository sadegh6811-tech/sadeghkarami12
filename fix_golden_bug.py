with open('genetic_engine_golden.py', 'r') as f:
    content = f.read()

# باگ اصلی: features باید بر اساس عرض اولیه X باشد، نه X_feat
old_code = '''    features = [f'x{i}' for i in range(X_feat.shape[1])]
    print(f"[*] متغیرهای اجباری: {features}")'''

new_code = '''    # باگ رفع شد: استفاده از X.shape[1] به جای X_feat.shape[1]
    features = [f'x{i}' for i in range(X.shape[1])]
    print(f"[*] متغیرهای اجباری (همه ۵ متغیر): {features}")'''

content = content.replace(old_code, new_code)

with open('genetic_engine_golden.py', 'w') as f:
    f.write(content)
print("[+] باگ باگ range برطرف شد.")
