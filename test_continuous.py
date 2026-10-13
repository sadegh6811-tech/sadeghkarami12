import numpy as np, csv, sys, os
sys.path.insert(0, '/data/data/com.termux/files/home/mathbot')
os.chdir('/data/data/com.termux/files/home/mathbot')

# ---- Load ----
H, L, O, C = [], [], [], []
with open('/storage/emulated/0/forex_bot/golden_points/AUDCAD_golden.csv') as f:
    for r in csv.DictReader(f):
        try:
            h,l,o,c = float(r['high']),float(r['low']),float(r['open']),float(r['close'])
            if h > l and h > 0:
                H.append(h); L.append(l); O.append(o); C.append(c)
        except (ValueError, KeyError): pass

H,L,O,C = map(np.array, [H,L,O,C])
n = len(H)
print(f"n = {n}")

# محاسبه ی بازده آینده (continuous)
HORIZON = 5
future_ret = np.zeros(n)
for i in range(n - HORIZON):
    future_ret[i] = (C[i+HORIZON] - C[i]) / C[i]

# حذف آخرین HORIZON کندل
H, L, O, C, future_ret = H[:-HORIZON], L[:-HORIZON], O[:-HORIZON], C[:-HORIZON], future_ret[:-HORIZON]
n = len(H)

# محاسبه ی features
wick = (H + L - O - C) / (H - L)
cpos = (C - L) / (H - L)
body = (C - O) / (H - L)
range_pct = (H - L) / C

print(f"After horizon filter: {n}")
print(f"wick     mean={wick.mean():+.4f}  std={wick.std():.4f}")
print(f"future_ret mean={future_ret.mean():+.6f}  std={future_ret.std():.6f}")

# فیلتر outlier
m = (np.abs(wick) < 3) & (np.abs(future_ret) < 0.1)
wick, cpos, body, range_pct, future_ret = wick[m], cpos[m], body[m], range_pct[m], future_ret[m]
print(f"After filter: {len(wick)}")

# ---- X: ترکیب features با future_ret ----
wick_x_ret = wick * future_ret
cpos_x_ret = cpos * future_ret

X = np.column_stack([
    wick,           # x0
    cpos,           # x1
    body,           # x2
    range_pct,      # x3
    future_ret,     # x4  (پیوسته!)
    wick_x_ret,     # x5
    cpos_x_ret,     # x6
])
print(f"\nX shape: {X.shape}")
print(f"Columns: [wick, cpos, body, range_pct, future_ret, wick*ret, cpos*ret]")

# ---- Discover ----
from discover import discover_equations
print("\n[*] Searching for wick ↔ future_ret relationship...")
result = discover_equations(X, n_equations=5, degree=2, 
                             tol_ratio=0.05, use_trig=False)

eqs = result[0]
print(f"\nEquations found ({len(eqs)}):")
for i, eq in enumerate(eqs[:5]):
    print(f"  [{i}] {eq}")

# ---- بررسی مهمترین جمله ----
print("\n" + "="*70)
print("تحلیل:")
print("="*70)
print(f"x4 = future_ret")
print(f"x0 = wick")
print(f"x5 = wick * future_ret")
print(f"x6 = cpos * future_ret")
print()
print("اگر در eq ها ضریب x4 یا x5 بزرگ باشد → رابطه ی wick↔ret کشف شد")
