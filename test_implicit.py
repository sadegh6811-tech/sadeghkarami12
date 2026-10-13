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
num = H + L - O - C
den = H - L
wick = num / den

# حذف outliers
m = np.abs(wick) < 3
num, den, wick = num[m], den[m], wick[m]
print(f"n = {len(num)}")

# ---- X: فقط سه ستون مستقل ----
# قرار است پیدا کنیم: den*wick - num = 0
X = np.column_stack([num, den, wick])
print(f"X shape: {X.shape}, columns: [num, den, wick]")

# ---- Discover ----
from discover import discover_equations
print("\n[*] Running discover_equations (implicit search)...")
result = discover_equations(X, n_equations=3, degree=2, 
                             tol_ratio=0.02, use_trig=False)

eqs = result[0]
print(f"\nEquations found: {len(eqs)}")
for i, eq in enumerate(eqs):
    print(f"  [{i}] {eq}")
