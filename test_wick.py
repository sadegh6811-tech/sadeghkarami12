import numpy as np
import csv
import os
import sys

# اضافه کردن مسیر mathbot
sys.path.insert(0, '/data/data/com.termux/files/home/mathbot')

# بررسی وجود discover.py
if not os.path.exists('discover.py'):
    print("❌ discover.py not found")
    sys.exit(1)
print("✅ discover.py found")

# خواندن داده ی فارکس
csv_path = '/storage/emulated/0/forex_bot/golden_points/AUDCAD_golden.csv'
if not os.path.exists(csv_path):
    print(f"❌ CSV not found: {csv_path}")
    sys.exit(1)
print(f"✅ CSV found")

# بارگذاری داده
H, L, O, C = [], [], [], []
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for r in reader:
        try:
            H.append(float(r['high']))
            L.append(float(r['low']))
            O.append(float(r['open']))
            C.append(float(r['close']))
        except (ValueError, KeyError):
            continue

print(f"✅ Loaded {len(H)} candles")

H = np.array(H)
L = np.array(L)
O = np.array(O)
C = np.array(C)

# هدف: فرمول wick
y = (H + L - O - C) / (H - L + 1e-10)

# ورودی: OHLC
X = np.column_stack([H, L, O, C])
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"y mean: {y.mean():.4f}, y std: {y.std():.4f}")

# اجرای کشف
try:
    from discover import discover_equations
    print("\n[*] Running discover_equations...")
    eqs, coefs, residual, history = discover_equations(X, n_equations=1, degree=3)
    print(f"\n✅ Discovered equations:")
    print(f"   eqs: {eqs}")
    print(f"   residual: {residual}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
