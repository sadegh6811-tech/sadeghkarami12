"""تست AI روی داده‌های کوانتومی."""
import numpy as np
from math_ai_core import frequency_validator

np.random.seed(42)
N = 500

print("=" * 70)
print("تست AI روی داده‌های کوانتومی")
print("=" * 70)

# ۱. کره بلاخ (نباید فرکانس داشته باشه)
print("\n[۱] کره بلاخ — انتظار: هیچ فرکانسی")
theta = np.random.uniform(0, np.pi, N)
phi = np.random.uniform(0, 2*np.pi, N)
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
for omega in [0.5, 1.0, 3.0]:
    res = frequency_validator(x, y, omega)
    print(f"  ω={omega}: verdict={res['verdict']} score={res['score']:.3f}")

# ۲. نوسانگر هارمونیک (نباید فرکانس داشته باشه)
print("\n[۲] نوسانگر هارمونیک — انتظار: هیچ فرکانسی")
x2 = np.random.uniform(-3, 3, N)
y2 = x2**2 / 2
for omega in [0.5, 1.0, 2.0]:
    res = frequency_validator(x2, y2, omega)
    print(f"  ω={omega}: verdict={res['verdict']} score={res['score']:.3f}")

# ۳. ذره در جعبه (نباید فرکانس داشته باشه)
print("\n[۳] ذره در جعبه — انتظار: هیچ فرکانسی")
n = np.random.uniform(1, 10, N)
E = n**2
for omega in [0.5, 1.0, 3.0]:
    res = frequency_validator(n, E, omega)
    print(f"  ω={omega}: verdict={res['verdict']} score={res['score']:.3f}")

# ۴. داده‌ی مثلثاتی واقعی (باید فرکانس 5 پیدا بشه)
print("\n[۴] داده‌ی مثلثاتی مصنوعی sin(5x) — انتظار: تأیید ω=5")
x4 = np.random.uniform(0, 10, N)
y4 = np.sin(5 * x4) + 0.05 * np.random.randn(N)
for omega in [3.0, 5.0, 7.0]:
    res = frequency_validator(x4, y4, omega)
    print(f"  ω={omega}: verdict={res['verdict']} score={res['score']:.3f}")

# ۵. تابع نمایی (نباید فرکانس داشته باشه)
print("\n[۵] تابع نمایی exp(-x²) — انتظار: هیچ فرکانسی")
x5 = np.random.uniform(-2, 2, N)
y5 = np.exp(-x5**2)
for omega in [1.0, 3.0, 5.0]:
    res = frequency_validator(x5, y5, omega)
    print(f"  ω={omega}: verdict={res['verdict']} score={res['score']:.3f}")

print("\n" + "=" * 70)
print("تست کامل شد")
print("=" * 70)
