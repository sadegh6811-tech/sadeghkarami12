import numpy as np, csv, sys, os
sys.path.insert(0, '/data/data/com.termux/files/home/mathbot')
os.chdir('/data/data/com.termux/files/home/mathbot')

print("=" * 75)
print("5D DESIGN v3 — Fixed Display")
print("=" * 75)

# ---- Load ----
H, L, O, C = [], [], [], []
with open('/storage/emulated/0/forex_bot/golden_points/AUDCAD_golden.csv') as f:
    for r in csv.DictReader(f):
        try:
            h,l,o,c = float(r['high']),float(r['low']),float(r['open']),float(r['close'])
            if h > l and h > 0 and l > 0:
                H.append(h); L.append(l); O.append(o); C.append(c)
        except (ValueError, KeyError): pass

H,L,O,C = map(np.array, [H,L,O,C])
n_full = len(H)

HORIZON = 5
future_ret = np.zeros(n_full)
for i in range(n_full - HORIZON):
    if C[i] > 0:
        future_ret[i] = (C[i+HORIZON] - C[i]) / C[i]

H,L,O,C = H[:-HORIZON], L[:-HORIZON], O[:-HORIZON], C[:-HORIZON]
future_ret = future_ret[:-HORIZON]

den = H - L
wick = (H + L - O - C) / den
cpos = (C - L) / den
body = (C - O) / den
range_pct = den / C

m = ((np.abs(wick) < 3) & (np.abs(future_ret) < 0.1) & 
     (den > 1e-8) & np.isfinite(wick) & np.isfinite(future_ret))

wick, cpos, body, range_pct = wick[m], cpos[m], body[m], range_pct[m]
future_ret = future_ret[m]

def zscore(x):
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    s = x.std()
    if s < 1e-10: return x - x.mean()
    return (x - x.mean()) / s

wick_z = zscore(wick)
cpos_z = zscore(cpos)
body_z = zscore(body)
range_pct_z = zscore(range_pct)
ret_z = zscore(future_ret)

print(f"n = {len(wick)}")

# ---- طرح برنده: D (Wick-focused) ----
print("\n" + "=" * 75)
print("DESIGN D (Wick-focused): [wick_z, cpos_z, range_pct_z, ret_z, wick_z*cpos_z]")
print("=" * 75)

X = np.column_stack([wick_z, cpos_z, range_pct_z, ret_z, wick_z * cpos_z])

from discover import discover_equations

# با degree=2
res = discover_equations(X, n_equations=5, degree=2, 
                          tol_ratio=0.02, use_trig=False)

eqs = res[0]
print(f"\n✅ Equations found: {len(eqs)}\n")

# ---- FIX: نمایش صحیح ----
print("Equations (converted to string):")
print("-" * 75)

# تعریف اسم ها برای خوانایی
names = ["wick_z", "cpos_z", "range_pct_z", "ret_z", "wick*cpos"]

for i, eq in enumerate(eqs):
    # FIX: تبدیل به string
    eq_str = str(eq)
    print(f"\n[{i}] {eq_str}")
    
    # ساده سازی: نمایش ضرایب اصلی
    print(f"    (length: {len(eq_str)} chars)")

# ---- بدون استفاده از [i][:180] روی sympy ----
print("\n" + "=" * 75)
print("SIMPLE INTERPRETATION:")
print("=" * 75)
print("Each equation is: c0*wick + c1*cpos + c2*range + c3*ret + c4*(wick*cpos) + ... = 0")
print("Looking for equations where ret_z (x3) has a significant coefficient.")
