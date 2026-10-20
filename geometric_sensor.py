import numpy as np

def geometric_sensor(X):
    """
    تشخیص می‌دهد که آیا داده در یک هایپرکیوب (مکعب) قرار دارد یا خیر.
    خروجی: 'cube' یا 'smooth'
    """
    # نرمال‌سازی
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-12)
    
    # ۱. بررسی بازه‌ی داده‌ها
    x_min, x_max = Xn.min(), Xn.max()
    range_ok = (-2.5 < x_min < -1.0) and (1.0 < x_max < 2.5)
    
    # ۲. بررسی تمرکز نقاط روی مرز
    # در هایپرکیوب، نقاط روی مقادیر حدی متمرکز هستند
    boundary_ratio = np.mean((np.abs(Xn) > 1.5))
    
    if range_ok and boundary_ratio > 0.3:
        return "cube"
    else:
        return "smooth"
