import numpy as np, csv, sys, os
sys.path.insert(0, '/data/data/com.termux/files/home/mathbot')
os.chdir('/data/data/com.termux/files/home/mathbot')

# ---- Load ----
H, L, O, C, pnl = [], [], [], [], []
with open('/storage/emulated/0/forex_bot/golden_points/AUDCAD_golden.csv') as f:
    for r in csv.DictReader(f):
        try:
            h,l,o,c = float(r['high']),float(r['low']),float(r['open']),float(r['close'])
            p = float(r['long_pnl'])
            if h > l and h > 0:
                H.append(h); L.append(l); O.append(o); C.append(c); pnl.append(p)
        except (ValueError, KeyError): pass

H,L,O,C,pnl = map(np.array, [H,L,O,C,pnl])
n = len(H)
print(f"n = {n}")

wick = (H + L - O - C) / (H - L)
cpos = (C - L) / (H - L)
body = (C - O) / (H - L)

# فیلتر outlier ها
m = (np.abs(wick) < 3) & (np.abs(pnl) < 0.05)
H,L,O,C,pnl,wick,cpos,body = H[m],L[m],O[m],C[m],pnl[m],wick[m],cpos[m],body[m]
print(f"After filter: {len(wick)}")

# ---- X: ترکیب features ----
# می خواهیم ببینیم آیا pnl را می توان از wick و cpos بازسازی کرد
wick_x_pnl = wick * pnl
cpos_x_pnl = cpos * pnl

X = np.column_stack([
    wick,        # x0
    cpos,        # x1
    pnl,         # x2
    wick_x_pnl,  # x3 = wick × pnl
    cpos_x_pnl,  # x4 = cpos × pnl
])
print(f"X shape: {X.shape}")
print(f"Columns: [wick, cpos, pnl, wick*pnl, cpos*pnl]")

# ---- Discover ----
from discover import discover_equations
result = discover_equations(X, n_equations=5, degree=2, 
                             tol_ratio=0.02, use_trig=False)

eqs = result[0]
print(f"\nEquations ({len(eqs)}):")
for i, eq in enumerate(eqs[:5]):
    print(f"  [{i}] {eq}")

print(f"\nنتایج زیر ساده سازی:")
print(f"  اگر eq[i] ≈ x2 + a*x0 + b*x1 = 0 → pnl = -a*wick - b*cpos")
print(f"  یعنی pnl را می توان از wick و cpos پیش بینی کرد!")
