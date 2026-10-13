import sys, csv
import numpy as np
sys.path.insert(0, '/data/data/com.termux/files/home/mathbot')
import mathbot_plus2 as M
from rational_lib import build_rational_library

PATH = '/storage/emulated/0/forex_bot/golden_points/EURUSD_golden.csv'
H, L, O, C, LG = [], [], [], [], []
with open(PATH) as f:
    for r in csv.DictReader(f):
        try:
            H.append(float(r['high'])); L.append(float(r['low']))
            O.append(float(r['open'])); C.append(float(r['close']))
            LG.append(int(r['long_golden']))
        except (ValueError, KeyError):
            continue

H, L, O, C = map(np.array, (H, L, O, C))
LG = np.array(LG, dtype=float)
X = np.column_stack([H, L, O, C])

print("[*] Building rational library...")
A, names = build_rational_library(X, base_names=['H','L','O','C'])
print(f"    library size: {A.shape}")

# پیدا کردن Wick
wick_names = ["(H+L-O-C)/(H-L)", "(-H-L+O+C)/(H-L)"]
wick_idx = None
for i, nm in enumerate(names):
    if nm in wick_names:
        wick_idx = i
        print(f"    Wick found at index {i}: {nm}")
        break
if wick_idx is None:
    print("    ⚠️ Wick NOT found in library")

# split
n = len(H)
tr = int(n * 0.7)
A_tr, A_te = A[:tr], A[tr:]
y_tr, y_te = LG[:tr], LG[tr:]

# نرمال‌سازی با safe
A_tr_n, norms = M.normalize_columns(A_tr)
# norms ممکن است صفر داشته باشد
norms = np.where(norms == 0, 1.0, norms)
A_te_n = np.nan_to_num(A_te / norms, nan=0.0, posinf=0.0, neginf=0.0)
yn_tr = (y_tr - y_tr.mean()) / (y_tr.std() + 1e-10)

# baseline Wick
if wick_idx is not None and np.std(y_te) > 1e-9:
    w_te = A_te_n[:, wick_idx]
    if np.std(w_te) > 1e-9:
        c_wick = np.corrcoef(w_te, y_te)[0,1]
        print(f"\n[Baseline] corr(Wick, golden) test = {c_wick:+.4f}")

# OMP
print("\n[*] Running OMP-BIC on rational library...")
try:
    out = M.omp_bic(A_tr_n, yn_tr, max_terms=8, min_rel_improve=0.0005)
    sel = out[0]
    coefs = out[1]
    print(f"    selected: {len(sel)} features")
    
    yhat_tr = A_tr_n @ coefs
    yhat_te = A_te_n @ coefs
    yhat_tr = np.nan_to_num(yhat_tr, nan=0.0, posinf=0.0, neginf=0.0)
    yhat_te = np.nan_to_num(yhat_te, nan=0.0, posinf=0.0, neginf=0.0)
    
    ss_res = np.sum((yn_tr - yhat_tr)**2)
    ss_tot = np.sum((yn_tr - yn_tr.mean())**2) + 1e-10
    r2_tr = 1 - ss_res / ss_tot
    
    if np.std(yhat_te) > 1e-9 and np.std(y_te) > 1e-9:
        c_mb = np.corrcoef(yhat_te, y_te)[0,1]
    else:
        c_mb = 0.0
    print(f"\n[Mathbot] R²_train = {r2_tr:.4f}")
    print(f"[Mathbot] corr(test) = {c_mb:+.4f}")
    
    thr_best, acc_best = 0, 0
    for thr in np.linspace(yhat_tr.min(), yhat_tr.max(), 100):
        acc = ((yhat_tr > thr).astype(int) == y_tr).mean()
        if acc > acc_best:
            acc_best, thr_best = acc, thr
    acc_te = ((yhat_te > thr_best).astype(int) == y_te).mean()
    print(f"[Mathbot] acc_train={acc_best:.4f}  acc_test={acc_te:.4f}")
    
    print("\n[*] Top features:")
    order = np.argsort(-np.abs(coefs))
    for rank, i in enumerate(order[:10], 1):
        if abs(coefs[i]) > 1e-8:
            marker = "  <-- Wick!" if i == wick_idx else ""
            print(f"  {rank:>2}. {names[i]:<45}  {coefs[i]:+.4f}{marker}")
except Exception as e:
    import traceback
    print("ERROR:"); traceback.print_exc()
