"""اسکریپت عیب‌یابی برای topology.py"""
import os
import sys
import importlib
import numpy as np

print("=" * 60)
print("۱. چک وجود فایل topology.py")
print("=" * 60)

if os.path.exists("topology.py"):
    with open("topology.py") as f:
        content = f.read()
    lines = content.split("\n")
    print(f"✓ فایل وجود دارد: {len(lines)} خط")
    if "_auto_radius" in content:
        print("✓ تابع _auto_radius در فایل هست")
    else:
        print("✗ تابع _auto_radius در فایل نیست → فایل قدیمی است!")
        sys.exit(1)
    if "_count_triangles" in content:
        print("✓ تابع _count_triangles در فایل هست")
    else:
        print("✗ تابع _count_triangles در فایل نیست")
else:
    print("✗ فایل topology.py وجود ندارد!")
    sys.exit(1)

print()
print("=" * 60)
print("۲. چک کش پایتون")
print("=" * 60)

pycache = "__pycache__"
if os.path.exists(pycache):
    files = os.listdir(pycache)
    print(f"پوشه __pycache__ شامل {len(files)} فایل:")
    for f in files:
        if "topology" in f:
            print(f"  ⚠️ {f} — کش topology! احتمال مشکل")
else:
    print("✓ پوشه __pycache__ وجود ندارد (تمیز است)")

print()
print("=" * 60)
print("۳. import کردن topology")
print("=" * 60)

try:
    if "topology" in sys.modules:
        del sys.modules["topology"]
    import topology
    print("✓ import موفق")
    print(f"  فایل: {topology.__file__}")
    print(f"  توابع موجود: {[x for x in dir(topology) if not x.startswith('_')]}")
    if hasattr(topology, "_auto_radius"):
        print("✓ تابع _auto_radius در ماژول هست")
    else:
        print("✗ تابع _auto_radius در ماژول نیست!")
except Exception as e:
    print(f"✗ خطا در import: {type(e).__name__}: {e}")
    sys.exit(1)

print()
print("=" * 60)
print("۴. تست _auto_radius با داده‌ی کره")
print("=" * 60)

try:
    # ساخت کره‌ی زیرنمونه
    np.random.seed(42)
    u = np.random.randn(150, 3)
    u /= np.linalg.norm(u, axis=1, keepdims=True)

    D = topology._pairwise_distances(u)
    r = topology._auto_radius(D)
    print(f"✓ شعاع محاسبه‌شده: {r:.4f}")

    A = (D <= r).astype(int)
    np.fill_diagonal(A, 0)
    b0 = topology._count_components(A)
    print(f"✓ Betti-0 با این شعاع: {b0}")

    if b0 == 1:
        print("✓ همه‌ی نقاط در یک مؤلفه‌ی همبند — درست")
    else:
        print(f"✗ Betti-0 = {b0} — باید 1 باشد")

    E = int(A.sum() // 2)
    F = topology._count_triangles(A)
    chi = 150 - E + F
    print(f"  E={E}  F={F}  chi={chi}")
except Exception as e:
    print(f"✗ خطا: {type(e).__name__}: {e}")

print()
print("=" * 60)
print("۵. تست analyze_topology روی کره")
print("=" * 60)

try:
    result = topology.analyze_topology(u, d_intrinsic=2)
    print()
    if result["betti_0"] == 1 and result["chi"] == 2:
        print("✓✓✓ نتیجه درست است!")
    else:
        print(f"✗ نتیجه غلط: b0={result['betti_0']}, chi={result['chi']}")
except Exception as e:
    print(f"✗ خطا: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("پایان تست")
print("=" * 60)
