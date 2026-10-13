import numpy as np, csv, sys, os
sys.path.insert(0, '/data/data/com.termux/files/home/mathbot')
os.chdir('/data/data/com.termux/files/home/mathbot')

print("=" * 75)
print("DESIGNING 5-DIMENSIONAL FEATURE SPACES (v2)")
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
print(f"\nRaw candles: {n_full}")

# ---- Future return ----
HORIZON = 5
future_ret = np.zeros(n_full)
for i in range(n_full - HORIZON):
    if C[i] > 0:
        future_ret[i] = (C[i+HORIZON] - C[i]) / C[i]

# Slicing
H,L,O,C = H[:-HORIZON], L[:-HORIZON], O[:-HORIZON], C[:-HORIZON]
future_ret = future_ret[:-HORIZON]

# ---- Features ----
den = H - L
wick = (H + L - O - C) / den
cpos = (C - L) / den
body = (C - O) / den
range_pct = den / C
# جایگزین برای vol: مقدار absolute حرکت body
abs_body = np.abs(C - O)

# ---- Filter ----
m = ((np.abs(wick) < 3) & 
     (np.abs(future_ret) < 0.1) & 
     (den > 1e-8) & 
     np.isfinite(wick) & 
     np.isfinite(future_ret))

wick, cpos, body, range_pct, abs_body = wick[m], cpos[m], body[m], range_pct[m], abs_body[m]
future_ret = future_ret[m]
H,L,O,C = H[m],L[m],O[m],C[m]

n = len(wick)
print(f"After filter: {n}")

# ---- Safe z-score ----
def zscore(x):
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    s = x.std()
    if s < 1e-10: return x - x.mean()
    return (x - x.mean()) / s

wick_z = zscore(wick)
cpos_z = zscore(cpos)
body_z = zscore(body)
range_pct_z = zscore(range_pct)
abs_body_z = zscore(abs_body)
ret_z = zscore(future_ret)

# برای طرح C: تفاضل ها به جای مقادیر مطلق
range_abs_z = zscore(H - L)          # همان range
body_signed_z = zscore(C - O)        # body با علامت
c_minus_l_z = zscore(C - L)          # CPos numerator
o_minus_l_z = zscore(O - L)          # Open position
h_minus_l_z = zscore(H - L)          # Range (same as above)

print(f"\nAll features ready (safe z-score)")

# ============================================================
# 5 طرح مستقل
# ============================================================
designs = {
    "A. Geometric": np.column_stack([wick_z, cpos_z, body_z, range_pct_z, ret_z]),
    
    "B. Abs-Body": np.column_stack([wick_z, cpos_z, abs_body_z, range_pct_z, ret_z]),
    
    "C. Differences": np.column_stack([
        range_abs_z,      # Range
        body_signed_z,    # Body
        c_minus_l_z,      # C-L
        o_minus_l_z,      # O-L
        ret_z
    ]),
    
    "D. Wick-focused": np.column_stack([
        wick_z, 
        cpos_z, 
        range_pct_z, 
        ret_z, 
        wick_z * cpos_z  # ← تعامل (ناخواسته dependent)
    ]),
    
    "E. Pure Ret": np.column_stack([
        wick_z, cpos_z, body_z, ret_z, 
        ret_z * wick_z   # تعامل
    ]),
}

# ============================================================
# Discovery
# ============================================================
from discover import discover_equations

results = {}

for name, X in designs.items():
    print("\n" + "=" * 75)
    print(f"DESIGN: {name}")
    print(f"X shape: {X.shape}")
    print("=" * 75)
    
    if X.shape[1] != 5:
        print(f"  SKIP: not 5D")
        continue
    
    # چک NaN/Inf
    if not np.all(np.isfinite(X)):
        print(f"  WARNING: X contains NaN/Inf — cleaning")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # چک همبستگی
    corr = np.corrcoef(X.T)
    max_corr = 0
    for i in range(5):
        for j in range(i+1, 5):
            if abs(corr[i,j]) > max_corr:
                max_corr = abs(corr[i,j])
    print(f"  Max feature correlation: {max_corr:.3f}")
    
    try:
        res = discover_equations(X, n_equations=3, degree=2, 
                                  tol_ratio=0.03, use_trig=False)
        eqs = res[0]
        print(f"  Equations: {len(eqs)}")
        for i, eq in enumerate(eqs[:3]):
            print(f"    [{i}] {eq[:180]}")
        results[name] = eqs
    except Exception as e:
        print(f"  ERROR: {e}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 75)
print("SUMMARY")
print("=" * 75)
for name in designs:
    if name in results:
        print(f"  OK  {name}: {len(results[name])} eq")
    else:
        print(f"  FAIL {name}")
