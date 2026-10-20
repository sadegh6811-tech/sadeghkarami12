"""ساخت داده‌ی هایپرکیوب ۵بعدی استاندارد."""
import numpy as np

np.random.seed(42)
N = 500
d = 5

# توزیع یکنواخت روی سطح هایپرکیوب
# هر نقطه حداقل یک مختصات با قدر مطلق 1 دارد
X = np.random.uniform(-1, 1, (N, d))
for i in range(N):
    # انتخاب تصادفی یک یا چند بعد برای قرار دادن روی مرز
    n_dims = np.random.randint(1, d+1)  # حداقل یکی
    dims = np.random.choice(d, n_dims, replace=False)
    for dim in dims:
        X[i, dim] = np.random.choice([-1.0, 1.0])

np.savetxt('/data/data/com.termux/files/home/hypercube5d_real.csv', X, delimiter=',', comments="")
print(f"[+] داده‌ی استاندارد ساخته شد: {X.shape}")
print(f"    (میانگین تعداد مرز در هر نقطه: {np.mean(np.sum(np.abs(X)==1, axis=1)):.2f})")
