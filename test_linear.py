import numpy as np, csv, sys, os
sys.path.insert(0, '/data/data/com.termux/files/home/mathbot')
os.chdir('/data/data/com.termux/files/home/mathbot')

# ---- Load ----
H, L, O, C = [], [], [], []
with open('/storage/emulated/0/forex_bot/golden_points/AUDCAD_golden.csv') as f:
    for r in csv.DictReader(f):
        try:
            h,l,o,c = float(r['high']),float(r['low']),float(r['open']),float(r['close'])
            if h > l: H.append(h); L.append(l); O.append(o); C.append(c)
        except (ValueError, KeyError): pass

H,L,O,C = map(np.array, [H,L,O,C])
num = H + L - O - C          # numerator
den = H - L                  # denominator
wick = num / den             # target

# حذف outliers
m = np.abs(wick) < 3
num, den, wick = num[m], den[m], wick[m]
print(f"n = {len(num)}")

# ---- X: شامل ستون ضرب ----
den_x_wick = den * wick      # = num (باید کشف شود)
X = np.column_stack([num, den, wick, den_x_wick])
print(f"X shape: {X.shape}")
print(f"Columns: [num, den, wick, den*wick]")

# ---- Discover ----
from discover import discover_equations
print("\n[*] Running...")
result = discover_equations(X, n_equations=3, degree=2, 
                             tol_ratio=0.02, use_trig=False)

eqs = result[0]
print(f"\nEquations found: {len(eqs)}")
for i, eq in enumerate(eqs):
    print(f"  [{i}] {eq}")

# بررسی: آیا eq حاوی (x0 - x3) هست؟
print(f"\nانتظار: x0 - x3 = 0  یعنی  num - den*wick = 0")
