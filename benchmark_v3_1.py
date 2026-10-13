#!/usr/bin/env python3
"""Benchmark v3.1 — fixed eval allowing subset of variables."""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.expanduser("~/mathbot"))
from discover import discover_equations
from sympy import lambdify, Symbol

def safe_sort_symbols(symbols):
    def keyfn(s):
        try: return int(s[1:])
        except: return 999
    return sorted([str(s) for s in symbols], key=keyfn)

# ---- Samplers ----
def sample_circle(n):
    t = np.random.uniform(0, 2*np.pi, n)
    return np.column_stack([np.cos(t), np.sin(t)])

def sample_sphere3(n):
    u = np.random.randn(n, 3); return u / np.linalg.norm(u, axis=1, keepdims=True)

def sample_sphere4(n):
    u = np.random.randn(n, 4); return u / np.linalg.norm(u, axis=1, keepdims=True)

def sample_cylinder(n):
    t = np.random.uniform(0, 2*np.pi, n)
    z = np.random.randn(n) * 0.5
    return np.column_stack([np.cos(t), np.sin(t), z])

def sample_paraboloid(n):
    u = np.random.randn(n) * 1.5; v = np.random.randn(n) * 1.5
    return np.column_stack([u, v, u*u + v*v])

def sample_hyperbolic(n):
    u = np.random.randn(n) * 1.5; v = np.random.randn(n) * 1.5
    return np.column_stack([np.sqrt(1 + u*u + v*v), u, v])

def sample_cone(n):
    t = np.random.uniform(0, 2*np.pi, n)
    r = np.random.uniform(0, 2, n)
    return np.column_stack([r*np.cos(t), r*np.sin(t), r])

def sample_helix(n):
    t = np.random.uniform(0, 4*np.pi, n)
    return np.column_stack([np.cos(t), np.sin(t), t/(4*np.pi)])

def sample_torus4(n):
    u = np.random.uniform(0, 2*np.pi, n); v = np.random.uniform(0, 2*np.pi, n)
    return np.column_stack([np.cos(u), np.sin(u), np.cos(v), np.sin(v)])

# (name, sampler, n, d) — ellipsoid حذف شد چون تکراریه
TESTS = [
    ("circle_2d",   sample_circle,     2, 1),
    ("sphere_3d",   sample_sphere3,    3, 2),
    ("sphere_4d",   sample_sphere4,    4, 3),
    ("cylinder",    sample_cylinder,   3, 2),
    ("paraboloid",  sample_paraboloid, 3, 2),
    ("hyperbolic",  sample_hyperbolic, 3, 2),
    ("cone",        sample_cone,       3, 2),
    ("helix",       sample_helix,      3, 1),
    ("torus_4d",    sample_torus4,     4, 2),
]

def evaluate_relaxed(eq, sampler, mu, sd, n_vars, n_test=500, seed_eval=999):
    """
    FIXED: allow equations using SUBSET of variables.
    Evaluate g on the columns actually present in the equation.
    """
    try:
        free = safe_sort_symbols(eq.free_symbols)
        if not free:
            return None, None
        # Check symbols are all x<i>
        col_idx = []
        for s in free:
            if not s.startswith('x'):
                return None, None
            try:
                col_idx.append(int(s[1:]))
            except ValueError:
                return None, None
        # col_idx must be < n_vars
        if max(col_idx) >= n_vars:
            return None, None
        
        syms = [Symbol(s) for s in free]
        np.random.seed(seed_eval)
        X_raw = sampler(n_test)
        X_std = (X_raw - mu) / sd
        # Evaluate on the corresponding columns only
        f = lambdify(syms, eq, 'numpy')
        vals = f(*[X_std[:, i] for i in col_idx])
        vals = np.asarray(vals, dtype=float).ravel()
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return None, None
        return float(np.median(np.abs(vals))), float(np.percentile(np.abs(vals), 95))
    except Exception:
        return None, None

def run_test(name, sampler, n_vars, d, seed_train, noise_frac=0.0):
    np.random.seed(seed_train)
    n_pts = 800 if n_vars <= 3 else 1500
    X_raw = sampler(n_pts)
    mu = X_raw.mean(0); sd = X_raw.std(0); sd[sd < 1e-12] = 1.0
    Xs = (X_raw - mu) / sd
    if noise_frac > 0:
        Xs = Xs + np.random.randn(*Xs.shape) * noise_frac

    n_eqs = n_vars - d
    t0 = time.time()
    try:
        res = discover_equations(Xs, n_equations=n_eqs, degree=2,
                                 tol_ratio=0.05, use_trig=True)
        eqs = res[0]
    except Exception as e:
        return {"name": name, "noise": noise_frac, "seed": seed_train,
                "status": "error", "resid": None, "time": time.time()-t0}
    elapsed = time.time() - t0
    if not eqs:
        return {"name": name, "noise": noise_frac, "seed": seed_train,
                "status": "no_eq", "resid": None, "time": elapsed}

    evals = []
    for eq in eqs:
        m, p = evaluate_relaxed(eq, sampler, mu, sd, n_vars)
        if m is not None:
            evals.append(m)

    if not evals:
        return {"name": name, "noise": noise_frac, "seed": seed_train,
                "status": "eval_fail", "resid": None, "time": elapsed}

    bm = min(evals)
    if bm < 0.05:    st = "excellent"
    elif bm < 0.15:  st = "good"
    elif bm < 0.5:   st = "ok"
    else:            st = "poor"
    return {"name": name, "noise": noise_frac, "seed": seed_train,
            "status": st, "resid": bm, "time": elapsed}

# ============================================================
SEEDS = [42, 7, 123, 2024, 999]
NOISE_LEVELS = [0.00, 0.05, 0.10, 0.15]

print("=" * 100)
print("  BENCHMARK v3.1 — RELAXED EVAL (subset of variables allowed)")
print("=" * 100)
print()

all_results = []
for noise in NOISE_LEVELS:
    print(f"\n{'='*100}")
    print(f"  NOISE = {100*noise:.0f}%   (aggregated over {len(SEEDS)} seeds)")
    print(f"{'='*100}")
    print(f"{'test':<12} {'mean_resid':>11} {'std_resid':>10} "
          f"{'status_counts':<30}")
    print("-" * 70)
    for name, func, nv, d in TESTS:
        rows = []
        for seed in SEEDS:
            r = run_test(name, func, nv, d, seed_train=seed, noise_frac=noise)
            rows.append(r)
            all_results.append(r)
        resids = [r["resid"] for r in rows if r["resid"] is not None]
        mr = np.mean(resids) if resids else float('nan')
        sr = np.std(resids) if resids else float('nan')
        from collections import Counter
        c = Counter(r["status"] for r in rows)
        sc = ",".join(f"{k}:{v}" for k, v in c.items())
        print(f"{name:<12} {mr:>11.4f} {sr:>10.4f} {sc:<30}")
    print()

print()
print("=" * 100)
print("  OVERALL STATISTICS")
print("=" * 100)
print()
from collections import Counter
statuses_all = [r["status"] for r in all_results]
c = Counter(statuses_all)
total = len(statuses_all)
for k in ["excellent", "good", "ok", "poor", "eval_fail", "no_eq", "error"]:
    if c.get(k, 0) > 0:
        print(f"  {k:<12} {c[k]:>4}  ({100*c[k]/total:.1f}%)")
success = c.get('excellent', 0) + c.get('good', 0)
print()
print(f"SUCCESS (excellent+good): {success}/{total} = {100*success/total:.1f}%")
