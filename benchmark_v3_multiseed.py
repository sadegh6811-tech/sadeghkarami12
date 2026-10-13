#!/usr/bin/env python3
"""Benchmark v3 with multiple seeds — verify stability of the fix."""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.expanduser("~/mathbot"))
from discover import discover_equations
from sympy import lambdify, Symbol

def safe_sort_symbols(symbols):
    def keyfn(s):
        try:
            return int(s[1:])
        except (ValueError, IndexError):
            return 999
    return sorted([str(s) for s in symbols], key=keyfn)

# ---- Samplers ----
def sample_circle(n):
    t = np.random.uniform(0, 2*np.pi, n)
    return np.column_stack([np.cos(t), np.sin(t)])

def sample_sphere3(n):
    u = np.random.randn(n, 3); return u / np.linalg.norm(u, axis=1, keepdims=True)

def sample_sphere4(n):
    u = np.random.randn(n, 4); return u / np.linalg.norm(u, axis=1, keepdims=True)

def sample_ellipsoid(n):
    u = np.random.randn(n, 3)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    return u * np.array([1.0, 2.0, 3.0])

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

TESTS = [
    ("circle_2d",   sample_circle,     2, 1),
    ("sphere_3d",   sample_sphere3,    3, 2),
    ("sphere_4d",   sample_sphere4,    4, 3),
    ("ellipsoid",   sample_ellipsoid,  3, 2),
    ("cylinder",    sample_cylinder,   3, 2),
    ("paraboloid",  sample_paraboloid, 3, 2),
    ("hyperbolic",  sample_hyperbolic, 3, 2),
    ("cone",        sample_cone,       3, 2),
    ("helix",       sample_helix,      3, 1),
    ("torus_4d",    sample_torus4,     4, 2),
]

def evaluate(eq, sampler, mu, sd, n_vars, n_test=500, seed_eval=999):
    try:
        free = safe_sort_symbols(eq.free_symbols)
        if not free or len(free) != n_vars: return None, None
        syms = [Symbol(s) for s in free]
        np.random.seed(seed_eval)
        X_raw = sampler(n_test)
        X_std = (X_raw - mu) / sd
        f = lambdify(syms, eq, 'numpy')
        vals = np.asarray(f(*[X_std[:,i] for i in range(len(syms))]), dtype=float).ravel()
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0: return None, None
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
        m, p = evaluate(eq, sampler, mu, sd, n_vars)
        if m is not None: evals.append(m)
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
NOISE = 0.0  # start with clean data

print("=" * 100)
print(f"  MULTI-SEED STABILITY TEST — noise=0%, seeds={SEEDS}")
print("=" * 100)
print()

# Collect results per seed
all_by_seed = {}
for seed in SEEDS:
    print(f"--- seed={seed} ---")
    print(f"{'test':<12} {'resid':>10} {'status':>11} {'time':>8}")
    print("-" * 45)
    results = []
    for name, func, nv, d in TESTS:
        r = run_test(name, func, nv, d, seed_train=seed)
        results.append(r)
        m_str = f"{r['resid']:.4f}" if r['resid'] is not None else "  --  "
        print(f"{name:<12} {m_str:>10} {r['status']:>11} {r['time']:>7.2f}s")
    all_by_seed[seed] = results
    print()

# ============================================================
# PER-TEST STABILITY
# ============================================================
print("=" * 100)
print("  PER-TEST STABILITY ACROSS SEEDS")
print("=" * 100)
print()
print(f"{'test':<12} {'mean_resid':>11} {'std_resid':>10} {'statuses':<40}")
print("-" * 80)

for name, _, _, _ in TESTS:
    rows = [r for seed in SEEDS for r in all_by_seed[seed] if r["name"] == name]
    resids = [r["resid"] for r in rows if r["resid"] is not None]
    if resids:
        mr = np.mean(resids); sr = np.std(resids)
    else:
        mr = sr = float('nan')
    statuses = [r["status"] for r in rows]
    status_str = ",".join(statuses)
    print(f"{name:<12} {mr:>11.4f} {sr:>10.4f} {status_str:<40}")

# ============================================================
# OVERALL STATISTICS
# ============================================================
print()
print("=" * 100)
print("  OVERALL ACROSS ALL SEEDS")
print("=" * 100)
print()
statuses_all = [r["status"] for seed in SEEDS for r in all_by_seed[seed]]
from collections import Counter
c = Counter(statuses_all)
total = len(statuses_all)
print(f"Total runs: {total}")
print(f"  excellent: {c['excellent']} ({100*c['excellent']/total:.1f}%)")
print(f"  good:      {c['good']} ({100*c['good']/total:.1f}%)")
print(f"  ok:        {c['ok']} ({100*c['ok']/total:.1f}%)")
print(f"  poor:      {c['poor']} ({100*c['poor']/total:.1f}%)")
print(f"  eval_fail: {c['eval_fail']} ({100*c['eval_fail']/total:.1f}%)")
print(f"  no_eq:     {c['no_eq']} ({100*c['no_eq']/total:.1f}%)")
print(f"  error:     {c['error']} ({100*c['error']/total:.1f}%)")
print()

# Success rate = excellent + good
success = c['excellent'] + c['good']
print(f"SUCCESS (excellent+good): {success}/{total} = {100*success/total:.1f}%")

# ============================================================
# TIME STATISTICS
# ============================================================
print()
print("=" * 100)
print("  TIME STATISTICS")
print("=" * 100)
print()
times_all = [r["time"] for seed in SEEDS for r in all_by_seed[seed]]
print(f"  mean:   {np.mean(times_all):.2f}s")
print(f"  median: {np.median(times_all):.2f}s")
print(f"  max:    {np.max(times_all):.2f}s")
print(f"  min:    {np.min(times_all):.2f}s")
