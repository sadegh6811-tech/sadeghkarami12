import re

with open('mathbot_hybrid.py', 'r') as f:
    content = f.read()

# اضافه کردن سنسور هندسی به diagnostic_sensor
old_sensor = '''    # ۴. بعد ذاتی
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-12)
    intrinsic_dim = estimate_dimension_twonn(Xn)
    
    return {"noise_ratio": noise_ratio, "max_corr": max_corr,
            "complexity": complexity, "intrinsic_dim": intrinsic_dim,
            "n_samples": n, "n_features": d}'''

new_sensor = '''    # ۴. بعد ذاتی
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-12)
    intrinsic_dim = estimate_dimension_twonn(Xn)
    
    # ۵. سنسور هندسی (تشخیص مکعب)
    from geometric_sensor import geometric_sensor
    topology = geometric_sensor(X)
    
    return {"noise_ratio": noise_ratio, "max_corr": max_corr,
            "complexity": complexity, "intrinsic_dim": intrinsic_dim,
            "n_samples": n, "n_features": d, "topology": topology}'''

content = content.replace(old_sensor, new_sensor)

# به‌روزرسانی decision_core برای استفاده از سنسور هندسی
old_decision = '''    # اگر داده با مدل درجه ۲ خوب فیت شد (نویز کم) → رگرسیون
    if noise < 0.1:
        return "regression", None, 0.1'''
new_decision = '''    # اگر سنسور هندسی تشخیص داد داده مکعب است، فقط از abs استفاده کن
    if sensor_data.get("topology") == "cube":
        return "regression", None, 0.1  # (در ادامه در discover.py فیلتر می‌کنیم)
        
    # اگر داده با مدل درجه ۲ خوب فیت شد (نویز کم) → رگرسیون
    if noise < 0.1:
        return "regression", None, 0.1'''
content = content.replace(old_decision, new_decision)

with open('mathbot_hybrid.py', 'w') as f:
    f.write(content)
print("[+] mathbot_hybrid.py به‌روزرسانی شد.")
