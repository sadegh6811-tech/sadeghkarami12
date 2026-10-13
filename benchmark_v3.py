#!/usr/bin/env python3
"""FIXED benchmark v3 — correct n_equations = n - d."""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.expanduser("~/mathbot"))
from discover import discover_equations
from sympy import lambdify, Symbol

np.random.seed(42)

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

# ---- CORRECT: (name, sampler, n, d) ----
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

def evaluate(eq, sampler, mu, sd, n_vars, n_test=500):
    try:
        free = safe_sort_symbols(eq.free_symbols)
        if not free or len(free) != n_vars: return None, None
        syms = [Symbol(s) for s in free]
        np.random.seed(999)
        X_raw = sampler(n_test)
        X_std = (X_raw - mu) / sd
        f = lambdify(syms, eq, 'numpy')
        vals = np.asarray(f(*[X_std[:,i] for i in range(len(syms))]), dtype=float).ravel()
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0: return None, None
        return float(np.median(np.abs(vals))), float(np.percentile(np.abs(vals), 95))
    except Exception:
        return None, None

def run_test(name, sampler, n_vars, d, noise_frac=0.0):
    np.random.seed(42)
    n_pts = 800 if n_vars <= 3 else 1500
    X_raw = sampler(n_pts)
    mu = X_raw.mean(0); sd = X_raw.std(0); sd[sd < 1e-12] = 1.0
    Xs = (X_raw - mu) / sd
    if noise_frac > 0:
        Xs = Xs + np.random.randn(*Xs.shape) * noise_frac

    # *** THE FIX ***
    n_eqs = n_vars - d

    t0 = time.time()
    try:
        res = discover_equations(Xs, n_equations=n_eqs, degree=2,
                                 tol_ratio=0.05, use_trig=True)
        eqs = res[0]
    except Exception as e:
        return {"name": name, "noise": noise_frac, "status": "error",
                "n_found": 0, "resid": None, "p95": None, "time": time.time()-t0,
                "n_eqs": n_eqs}
    elapsed = time.time() - t0

    if not eqs:
        return {"name": name, "noise": noise_frac, "status": "no_eq",
                "n_found": 0, "resid": None, "p95": None, "time": elapsed,
                "n_eqs": n_eqs}

    evals = []
    for eq in eqs:
        m, p = evaluate(eq, sampler, mu, sd, n_vars)
        if m is not None: evals.append((m, p))

    if not evals:
        return {"name": name, "noise": noise_frac, "status": "eval_fail",
                "n_found": len(eqs), "resid": None, "p95": None, "time": elapsed,
                "n_eqs": n_eqs}

    bm = min(e[0] for e in evals); bp = min(e[1] for e in evals)
    if bm < 0.05:    st = "excellent"
    elif bm < 0.15:  st = "good"
    elif bm < 0.5:   st = "ok"
    else:            st = "poor"

    return {"name": name, "noise": noise_frac, "status": st,
            "n_found": len(eqs), "resid": bm, "p95": bp, "time": elapsed,
            "n_eqs": n_eqs}

# ---- MAIN ----
print("=" * 100)
print("  BENCHMARK v3 — FIXED n_equations = n - d")
print("=" * 100)
print()

NOISE_LEVELS = [0.00, 0.05, 0.10, 0.15]
all_results = []

for noise in NOISE_LEVELS:
    print(f"\n{'='*100}")
    print(f"  NOISE = {100*noise:.0f}%")
    print(f"{'='*100}")
    print(f"{'test':<12} {'n_eqs':>6} {'n_found':>8} {'resid':>10} "
          f"{'status':>11} {'time':>8}")
    print("-" * 70)
    for name, func, nv, d in TESTS:
        r = run_test(name, func, nv, d, noise_frac=noise)
        all_results.append(r)
        m_str = f"{r['resid']:.4f}" if r['resid'] is not None else "  --  "
        print(f"{name:<12} {r['n_eqs']:>6} {r['n_found']:>8} {m_str:>10} "
              f"{r['status']:>11} {r['time']:>7.2f}s")
    print()

print("=" * 100)
print("  SUMMARY")
print("=" * 100)
print()
print(f"{'noise':>6} {'excellent':>10} {'good':>6} {'ok':>5} {'poor':>6} {'fail':>6}")
print("-" * 50)
for noise in NOISE_LEVELS:
    sub = [r for r in all_results if r["noise"] == noise]
    exc = sum(1 for r in sub if r["status"] == "excellent")
    gd  = sum(1 for r in sub if r["status"] == "good")
    ok  = sum(1 for r in sub if r["status"] == "ok")
    pr  = sum(1 for r in sub if r["status"] == "poor")
    fl  = len(sub) - exc - gd - ok - pr
    print(f"{100*noise:>5.0f}% {exc:>10} {gd:>6} {ok:>5} {pr:>6} {fl:>6}")
