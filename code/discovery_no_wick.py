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

A, names = build_rational_library(X, base_names=['H','L','O','C'])
print(f"original library: {A.shape}")

# === FIX 1: حذف دقیق فقط ۴ معادل Wick ===
WICK_EXACT = {
    "(H+L-O-C)/(H-L)",
    "(-H-L+O+C)/(H-L)",
    "(H+L-C-O)/(H-L)",
    "(-H-L+C+O)/(H-L)",
}
keep = [i for i, nm in enumerate(names) if nm not in WICK_EXACT]
removed = [names[i] for i, nm in enumerate(names) if nm in WICK_EXACT]

print(f"removed {len(removed)} exact Wick variants:")
for r in removed:
    print(f"  - {r}")
A2 = A[:, keep]
names2 = [names[i] for i in keep]
print(f"new library: {A2.shape}")

# === FIX 2: بررسی توازن test ===
n = len(H); tr = int(n * 0.7)
y_tr, y_te = LG[:tr], LG[tr:]
print(f"\ny_train: mean={y_tr.mean():.3f},  y_test: mean={y_te.mean():.3f}")
if y_te.std() < 1e-6:
    print("❌ y_test is constant — split point bad")
    sys.exit(1)

A_tr, A_te = A2[:tr], A2[tr:]
A_tr_n, norms = M.normalize_columns(A_tr)
norms = np.where(norms == 0, 1.0, norms)
A_te_n = np.nan_to_num(A_te / norms, nan=0.0, posinf=0.0, neginf=0.0)
yn_tr = (y_tr - y_tr.mean()) / (y_tr.std() + 1e-10)

print(f"\n[*] Running OMP-BIC WITHOUT Wick...")
print(f"    library: {A_tr_n.shape},  max_terms=10")
try:
    out = M.omp_bic(A_tr_n, yn_tr, max_terms=10, min_rel_improve=0.0001)
    sel = out[0]; coefs = out[1]
except Exception as e:
    import traceback
    print("OMP ERROR:"); traceback.print_exc(); sys.exit(1)

yhat_tr = np.nan_to_num(A_tr_n @ coefs)
yhat_te = np.nan_to_num(A_te_n @ coefs)

ss_res = np.sum((yn_tr - yhat_tr)**2)
ss_tot = np.sum((yn_tr - yn_tr.mean())**2) + 1e-10
r2_tr = 1 - ss_res / ss_tot

if np.std(yhat_te) > 1e-9 and np.std(y_te) > 1e-9:
    c_mb = np.corrcoef(yhat_te, y_te)[0,1]
else:
    c_mb = 0.0

thr_best, acc_best = 0, 0
for thr in np.linspace(yhat_tr.min(), yhat_tr.max(), 100):
    acc = ((yhat_tr > thr).astype(int) == y_tr).mean()
    if acc > acc_best:
        acc_best, thr_best = acc, thr
acc_te = ((yhat_te > thr_best).astype(int) == y_te).mean()

print(f"\n[Mathbot without Wick]")
print(f"  n features selected: {len(sel)}")
print(f"  R²_train = {r2_tr:.4f}")
print(f"  corr(test) = {c_mb:+.4f}")
print(f"  acc_test   = {acc_te:.4f}")
print(f"\n  baseline (Wick in library): corr=+0.5065, acc=0.7376")

print("\nSelected features:")
order = np.argsort(-np.abs(coefs))
for rank, i in enumerate(order[:10], 1):
    if abs(coefs[i]) > 1e-6:
        print(f"  {rank:>2}. {names2[i]:<45}  {coefs[i]:+.4f}")
