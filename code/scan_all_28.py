import sys, csv, glob, os
import numpy as np
sys.path.insert(0, '/data/data/com.termux/files/home/mathbot')
import mathbot_plus2 as M
from rational_lib import build_rational_library

# فقط ۲ معادل واقعی Wick (بقیه در کتابخانه ساخته نمی‌شوند)
WICK_EXACT = {
    "(H+L-O-C)/(H-L)",
    "(-H-L+O+C)/(H-L)",
}


def load_pair(path):
    H, L, O, C, LG = [], [], [], [], []
    with open(path, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            try:
                H.append(float(r['high'])); L.append(float(r['low']))
                O.append(float(r['open'])); C.append(float(r['close']))
                LG.append(int(r['long_golden']))
            except (ValueError, KeyError):
                continue
    return (np.array(H), np.array(L), np.array(O), np.array(C),
            np.array(LG, dtype=float))


def run_omp(A_tr_n, yn_tr, A_te_n, y_tr, y_te, max_terms=8, tol=0.0005):
    try:
        out = M.omp_bic(A_tr_n, yn_tr, max_terms=max_terms, min_rel_improve=tol)
        sel, coefs = out[0], out[1]
    except Exception as e:
        return {'error': str(e)}
    
    if len(sel) == 0:
        return {'error': 'no features selected'}
    
    yhat_tr = np.nan_to_num(A_tr_n @ coefs)
    yhat_te = np.nan_to_num(A_te_n @ coefs)
    ss_res = np.sum((yn_tr - yhat_tr)**2)
    ss_tot = np.sum((yn_tr - yn_tr.mean())**2) + 1e-10
    r2 = 1 - ss_res / ss_tot
    
    if np.std(yhat_te) > 1e-9 and np.std(y_te) > 1e-9:
        c = float(np.corrcoef(yhat_te, y_te)[0, 1])
    else:
        c = 0.0
    
    thr_b, acc_b = 0, 0
    for thr in np.linspace(yhat_tr.min(), yhat_tr.max(), 60):
        a = ((yhat_tr > thr).astype(int) == y_tr).mean()
        if a > acc_b:
            acc_b, thr_b = a, thr
    acc_te = float(((yhat_te > thr_b).astype(int) == y_te).mean())
    
    return {
        'n_sel': len(sel), 'r2': float(r2), 'corr': c, 'acc': acc_te,
        'coefs': coefs, 'sel': sel
    }


def process_pair(name, path):
    try:
        H, L, O, C, LG = load_pair(path)
    except Exception as e:
        return {'name': name, 'skip': f'load error: {e}'}
    
    n = len(H)
    if n < 1000:
        return {'name': name, 'skip': f'only {n} rows'}
    
    X = np.column_stack([H, L, O, C])
    A, names = build_rational_library(X, base_names=['H', 'L', 'O', 'C'])
    
    tr = int(n * 0.7)
    y_tr, y_te = LG[:tr], LG[tr:]
    if y_te.std() < 1e-6 or y_tr.std() < 1e-6:
        return {'name': name, 'skip': 'constant y in split'}
    
    # نرمال‌سازی
    A_tr, A_te = A[:tr], A[tr:]
    A_tr_n, norms = M.normalize_columns(A_tr)
    norms = np.where(norms == 0, 1.0, norms)
    A_te_n = np.nan_to_num(A_te / norms, nan=0.0, posinf=0.0, neginf=0.0)
    yn_tr = (y_tr - y_tr.mean()) / (y_tr.std() + 1e-10)
    
    # با Wick
    res_w = run_omp(A_tr_n, yn_tr, A_te_n, y_tr, y_te, max_terms=8)
    
    # بدون Wick
    keep = [i for i, nm in enumerate(names) if nm not in WICK_EXACT]
    A2 = A[:, keep]
    A2_tr, A2_te = A2[:tr], A2[tr:]
    A2_tr_n, n2 = M.normalize_columns(A2_tr)
    n2 = np.where(n2 == 0, 1.0, n2)
    A2_te_n = np.nan_to_num(A2_te / n2, nan=0.0, posinf=0.0, neginf=0.0)
    res_nw = run_omp(A2_tr_n, yn_tr, A2_te_n, y_tr, y_te, max_terms=10)
    
    return {'name': name, 'n': n, 'with_w': res_w, 'no_w': res_nw}


# === اجرا ===
paths = sorted(glob.glob('/storage/emulated/0/forex_bot/golden_points/*_golden.csv'))
print(f"Found {len(paths)} pairs\n")

results = []
for i, p in enumerate(paths, 1):
    name = os.path.basename(p).replace('_golden.csv', '')
    r = process_pair(name, p)
    
    if 'skip' in r:
        print(f"[{i:>2}/{len(paths)}] {name:<8} SKIP: {r['skip']}")
        continue
    
    rw, rn = r['with_w'], r['no_w']
    if 'error' in rw or 'error' in rn:
        err = rw.get('error') or rn.get('error')
        print(f"[{i:>2}/{len(paths)}] {name:<8} OMP error: {err}")
        continue
    
    # فقط اگر هر دو موفق بودند اضافه کن
    results.append(r)
    print(f"[{i:>2}/{len(paths)}] {name:<8}  "
          f"W: n={rw['n_sel']} c={rw['corr']:+.3f} a={rw['acc']:.3f}  |  "
          f"noW: n={rn['n_sel']} c={rn['corr']:+.3f} a={rn['acc']:.3f}")

# === جدول ===
if not results:
    print("\n❌ No successful pairs")
    sys.exit(1)

print("\n" + "="*78)
print(f"{'Pair':<9} {'W_corr':>8} {'W_acc':>7} {'nW_corr':>8} {'nW_acc':>7} "
      f"{'dCorr':>8} {'dAcc':>7}")
print("-"*78)

deltas_c, deltas_a = [], []
for r in results:
    rw, rn = r['with_w'], r['no_w']
    dc = rw['corr'] - rn['corr']
    da = rw['acc'] - rn['acc']
    deltas_c.append(dc); deltas_a.append(da)
    print(f"{r['name']:<9} {rw['corr']:>+8.3f} {rw['acc']:>7.3f} "
          f"{rn['corr']:>+8.3f} {rn['acc']:>7.3f} {dc:>+8.3f} {da:>+7.3f}")

print("-"*78)
mean_wc = np.mean([r['with_w']['corr'] for r in results])
mean_wa = np.mean([r['with_w']['acc'] for r in results])
mean_nc = np.mean([r['no_w']['corr'] for r in results])
mean_na = np.mean([r['no_w']['acc'] for r in results])
print(f"{'MEAN':<9} {mean_wc:>+8.3f} {mean_wa:>7.3f} "
      f"{mean_nc:>+8.3f} {mean_na:>7.3f} {np.mean(deltas_c):>+8.3f} {np.mean(deltas_a):>+7.3f}")

n_pairs = len(results)
wins_corr = sum(1 for d in deltas_c if d > 0)
wins_acc = sum(1 for d in deltas_a if d > 0)
print(f"\nWick wins on corr: {wins_corr}/{n_pairs}")
print(f"Wick wins on acc:  {wins_acc}/{n_pairs}")

# === آیا Wick انتخاب شده؟ ===
print(f"\n[*] Did Mathbot select Wick?")
wick_picked = 0
for r in results:
    rw = r['with_w']
    # اندیس Wick در کتابخانه‌ی اصلی چیه؟ از names می‌فهمیم
    # اما run_omp اندیس رو برمی‌گردونه. باید names رو نگه‌داریم
    pass

# بهتر: چک کنیم n_sel==1 در حالت با Wick
one_feat = sum(1 for r in results if r['with_w']['n_sel'] == 1)
print(f"  pairs where only 1 feature selected: {one_feat}/{n_pairs}")
