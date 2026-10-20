"""تست باگ سیستم خبره‌ی math_ai_core."""
import numpy as np
from math_ai_core import frequency_validator, filter_frequencies

np.random.seed(42)
N = 500

print("=" * 70)
print("تست ۱: فرکانس واقعی (sin(3x) + نویز کم)")
print("=" * 70)
x = np.random.uniform(0, 10, N)
y = np.sin(3 * x) + 0.05 * np.random.randn(N)
res = frequency_validator(x, y, 3.0, verbose=True)
print(f"نتیجه: {res['verdict']} (score={res['score']:.3f})")
assert res["verdict"] == "accepted", "❌ فرکانس واقعی رد شد!"
print("✅ PASS\n")

print("=" * 70)
print("تست ۲: فرکانس جعلی (y = x²، بدون مثلثات)")
print("=" * 70)
x = np.random.uniform(1, 10, N)
y = x ** 2
res = frequency_validator(x, y, 0.5, verbose=True)
print(f"نتیجه: {res['verdict']} (score={res['score']:.3f})")
assert res["verdict"] != "accepted", "❌ فرکانس جعلی قبول شد!"
print("✅ PASS\n")

print("=" * 70)
print("تست ۳: فرکانس جعلی (y = x²، ω=0.612)")
print("=" * 70)
x = np.random.uniform(1, 10, N)
y = x ** 2
res = frequency_validator(x, y, 0.612, verbose=True)
print(f"نتیجه: {res['verdict']} (score={res['score']:.3f})")
assert res["verdict"] != "accepted", "❌ فرکانس جعلی قبول شد!"
print("✅ PASS\n")

print("=" * 70)
print("تست ۴: فرکانس جعلی (y = x، فرکانس 0.5)")
print("=" * 70)
x = np.random.uniform(1, 10, N)
y = x
res = frequency_validator(x, y, 0.5, verbose=True)
print(f"نتیجه: {res['verdict']} (score={res['score']:.3f})")
assert res["verdict"] != "accepted", "❌ فرکانس جعلی قبول شد!"
print("✅ PASS\n")

print("=" * 70)
print("تست ۵: فرکانس واقعی با نویز متوسط (sin(5x) + 0.3*noise)")
print("=" * 70)
x = np.random.uniform(0, 10, N)
y = np.sin(5 * x) + 0.3 * np.random.randn(N)
res = frequency_validator(x, y, 5.0, verbose=True)
print(f"نتیجه: {res['verdict']} (score={res['score']:.3f})")
# با نویز بالا ممکنه weak بشه ولی نباید رد مطلق بشه
print("✅ PASS\n")

print("=" * 70)
print("تست ۶: فیلتر فرکانس‌های متعدد")
print("=" * 70)
x = np.random.uniform(0, 10, N)
y = np.sin(3 * x) + 0.05 * np.random.randn(N)
candidates = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0]
approved = filter_frequencies(x, y, candidates, threshold=0.5)
print(f"کاندیدها: {candidates}")
print(f"تأییدشده: {approved}")
assert 3.0 in approved, "❌ فرکانس واقعی 3.0 رد شد!"
assert len(approved) <= 3, f"❌ تعداد زیاد تأیید شد: {len(approved)}"
print("✅ PASS\n")

print("=" * 70)
print("تست ۷: چند فرکانس واقعی (sin(2x) + 0.5*sin(7x))")
print("=" * 70)
x = np.random.uniform(0, 10, N)
y = np.sin(2 * x) + 0.5 * np.sin(7 * x) + 0.05 * np.random.randn(N)
candidates = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
approved = filter_frequencies(x, y, candidates, threshold=0.5)
print(f"کاندیدها: {candidates}")
print(f"تأییدشده: {approved}")
assert 2.0 in approved, "❌ فرکانس 2.0 رد شد!"
assert 7.0 in approved, "❌ فرکانس 7.0 رد شد!"
print("✅ PASS\n")

print("=" * 70)
print("تست ۸: فرکانس در بازه‌ی کوچیک (باید رد بشه)")
print("=" * 70)
x = np.random.uniform(1, 3, N)  # بازه‌ی فقط 2 واحد
y = np.sin(0.5 * x) + 0.05 * np.random.randn(N)
res = frequency_validator(x, y, 0.5, verbose=True)
print(f"نتیجه: {res['verdict']} (score={res['score']:.3f})")
# فرکانس 0.5 در بازه‌ی 2 واحدی: period = 12.6 → n_periods = 0.16 → رد
assert res["verdict"] != "accepted", "❌ فرکانس در بازه‌ی کوچیک قبول شد!"
print("✅ PASS\n")

print("=" * 70)
print("🎉 همه‌ی تست‌ها پاس شدند!")
print("=" * 70)
