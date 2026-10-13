import numpy as np
import csv, os, sys

MATHBOT = '/data/data/com.termux/files/home/mathbot'
sys.path.insert(0, MATHBOT); os.chdir(MATHBOT)

# ---- 1. Load data ----
print("[1] Loading data...")
H, L, O, C, pnl = [], [], [], [], []
csv_path = '/storage/emulated/0/forex_bot/golden_points/AUDCAD_golden.csv'

with open(csv_path) as f:
    for r in csv.DictReader(f):
        try:
            h = float(r['high'])
            l = float(r['low'])
            o = float(r['open'])
            c = float(r['close'])
            p = float(r['long_pnl'])
            if h > l and h > 0 and l > 0:
                H.append(h); L.append(l); O.append(o); C.append(c); pnl.append(p)
        except (ValueError, KeyError):
            continue

H = np.array(H); L = np.array(L); O = np.array(O); C = np.array(C); pnl = np.array(pnl)
n = len(H)
print(f"    Loaded {n} candles")

if n < 100:
    print("    ERROR: not enough data")
    sys.exit(1)

# ---- 2. Feature engineering (بدون wick) ----
print("[2] Engineering features...")
num = H + L - O - C          # = (H+L) - (O+C)
den = H - L                  # = Range
abs_body = np.abs(C - O)
body_dir = np.where(C > O, 1.0, -1.0)  # +1 if bullish candle
upper_wick = H - np.maximum(O, C)
lower_wick = np.minimum(O, C) - L
wick_total = upper_wick + lower_wick
cpos = (C - L) / den         # close position
range_pct = den / C          # normalized range

# ستون‌های X (ویژگی ها)
X = np.column_stack([
    H, L, O, C,              # OHLC
    num, den,                # ratio components
    upper_wick, lower_wick,  # wicks
    cpos, range_pct          # ratios
])
print(f"    X shape: {X.shape}")
print(f"    Columns: [H, L, O, C, num, den, upper_w, lower_w, cpos, range_pct]")

# ---- 3. Target: بازده آینده ----
# ما می خواهیم ببینیم آیا بازده با فرمولی از ویژگی ها پیش بینی می شود
# اضافه کردن pnl به X برای کشف رابطه
X_with_y = np.column_stack([X, pnl])
print(f"    X_with_y shape: {X_with_y.shape}")

# ---- 4. Discovery ----
print("[3] Running discover_equations...")
try:
    from discover import discover_equations
except ImportError:
    print("    ERROR: cannot import discover_equations")
    sys.exit(1)

result = discover_equations(X_with_y, n_equations=3, degree=2, 
                             tol_ratio=0.01, use_trig=False)

# ---- 5. بررسی خروجی ----
print(f"\n[4] Result:")
if result is None:
    print("    None")
    sys.exit(1)

print(f"    type: {type(result)}")
if isinstance(result, tuple):
    print(f"    len: {len(result)}")
    eqs = result[0]
    print(f"    Equations: {len(eqs)}")
    for i, eq in enumerate(eqs[:5]):
        print(f"      [{i}] {eq}")
else:
    print(f"    value: {result}")

print("\nDONE")
