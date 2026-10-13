import numpy as np
import csv
import os
import sys

print("=" * 70)
print("TEST: Discover wick formula from OHLC data")
print("=" * 70)

# 1. اضافه کردن مسیر
MATHBOT_DIR = '/data/data/com.termux/files/home/mathbot'
sys.path.insert(0, MATHBOT_DIR)
os.chdir(MATHBOT_DIR)

# 2. بررسی فایل ها
print("\n[1/5] Checking files...")
if not os.path.exists('discover.py'):
    print("  X discover.py not found")
    sys.exit(1)
print("  OK discover.py")

# 3. بارگذاری داده
print("\n[2/5] Loading data...")
csv_path = '/storage/emulated/0/forex_bot/golden_points/AUDCAD_golden.csv'
if not os.path.exists(csv_path):
    print(f"  X CSV not found: {csv_path}")
    sys.exit(1)

H, L, O, C = [], [], [], []
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for r in reader:
        try:
            h = float(r['high'])
            l = float(r['low'])
            o = float(r['open'])
            c = float(r['close'])
            if h > l and h > 0 and l > 0:
                H.append(h)
                L.append(l)
                O.append(o)
                C.append(c)
        except (ValueError, KeyError):
            continue

H = np.array(H)
L = np.array(L)
O = np.array(O)
C = np.array(C)
n = len(H)
print(f"  OK Loaded {n} candles")

if n < 100:
    print("  X Not enough data")
    sys.exit(1)

# 4. محاسبه wick
print("\n[3/5] Computing wick...")
wick = (H + L - O - C) / (H - L)
# حذف outliers
mask = np.abs(wick) < 5
H, L, O, C, wick = H[mask], L[mask], O[mask], C[mask], wick[mask]
n = len(H)
print(f"  After cleanup: {n} candles")
print(f"  wick mean: {wick.mean():+.4f}")
print(f"  wick std:  {wick.std():.4f}")
print(f"  wick range: [{wick.min():+.3f}, {wick.max():+.3f}]")

# 5. آماده سازی X
# شامل wick به عنوان ستون پنجم
X = np.column_stack([H, L, O, C, wick])
print(f"\n[4/5] X shape: {X.shape}")
print(f"  Columns: [H, L, O, C, wick]")

# 6. تست ساده اول: X خودش باید همبستگی داخلی داشته باشد
print("\n[5/5] Running discover_equations...")
try:
    from discover import discover_equations
    print("  OK Imported discover_equations")
    
    # تست با degree پایین برای سرعت
    result = discover_equations(X, n_equations=2, degree=2, use_trig=False)
    print(f"  OK Discovery completed")
    
    # بررسی خروجی
    print(f"\n  Result type: {type(result)}")
    if isinstance(result, tuple):
        print(f"  Tuple length: {len(result)}")
        for i, item in enumerate(result):
            print(f"    [{i}] type={type(item)}")
            if hasattr(item, '__len__') and not isinstance(item, str):
                try:
                    print(f"        len={len(item)}")
                except:
                    pass
            print(f"        value (first 200 chars): {str(item)[:200]}")
    else:
        print(f"  Result: {str(result)[:500]}")
    
except Exception as e:
    print(f"  X Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
