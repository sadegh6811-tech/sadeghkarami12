#!/usr/bin/env python3
"""FIXED benchmark for implicit manifold discovery."""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.expanduser("~/mathbot"))
from discover import discover_equations
from sympy import lambdify, Symbol

np.random.seed(42)

# ============================================================
# SAFE SYMBOL SORT
# ============================================================
def safe_sort_symbols(symbols):
    def keyfn(s):
        try:
            return int(s[1:])
        except (ValueError, IndexError):
            return 999
    return sorted([str(s) for s in symbols], key=keyfn)

# ============================================================
# MANIFOLD SAMPLERS (all random)
# ============================================================
def sample_circle(n):
    t = np.random.uniform(0, 2*np.pi, n)
    return np.column_stack([np.cos(t), np.sin(t)])

def sample_sphere3(n):
    u = np.random.randn(n, 3)
    return u / np.linalg.norm(u, axis=1, keepdims=True)

def sample_sphere4(n):
    u = np.random.randn(n, 4)
    return u / np.linalg.norm(u, axis=1, keepdims=True)

def sample_ellipsoid(n):
    u = np.random.randn(n, 3)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    return u * np.array([1.0, 2.0, 3.0])

def sample_cylinder(n):
    t = np.random.uniform(0, 2*np.pi, n)
    z = np.random.randn(n) * 0.5
    return np.column_stack([np.cos(t), np.sin(t), z])

def sample_paraboloid(n):
    u = np.random.randn(n) * 1.5
    v = np.random.randn(n) * 1.5
    return np.column_stack([u, v, u*u + v*v])

def sample_hyperbolic(n):
    u = np.random.randn(n) * 1.5
    v = np.random.randn(n) * 1.5
    return np.column_stack([np.sqrt(1 + u*u + v*v), u, v])

def sample_cone(n):
    t = np.random.uniform(0, 2*np.pi, n)
    r = np.random.uniform(0, 2, n)
    return np.column_stack([r*np.cos(t), r*np.sin(t), r])

def sample_helix(n):
    # RANDOM samples along the helix, not linspace
    t = np.random.uniform(0, 4*np.pi, n)
    return np.column_stack([np.cos(t), np.sin(t), t/(4*np.pi)])

def sample_torus4(n):
    u = np.random.uniform(0, 2*np.pi, n)
    v = np.random.uniform(0, 2*np.pi, n)
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

# ============================================================
# FIXED EVALUATION: apply SAME standardization to test points
# ============================================================
def evaluate_with_transform(eq, sampler, mu, sd, n_vars, n_test=500):
    """
    Evaluate g on fresh points transformed with SAME mu/sd.
    Returns median |g(x)|, p95 |g(x)|.
    """
    try:
        free = safe_sort_symbols(eq.free_symbols)
        if not free:
            return None, None
        syms = [Symbol(s) for s in free]
        n_syms = len(syms)
        if n_syms != n_vars:
            return None, None  # wrong dimensionality

        # Fresh test points — random, new seed
        np.random.seed(999)
        X_test_raw = sampler(n_test)
        X_test_std = (X_test_raw - mu) / sd

        f = lambdify(syms, eq, 'numpy')
        vals = f(*[X_test_std[:, i] for i in range(n_syms)])
        vals = np.asarray(vals, dtype=float).ravel()
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return None, None
        abs_vals = np.abs(vals)
        return float(np.median(abs_vals)), float(np.percentile(abs_vals, 95))
    except Exception:
        return None, None


# ============================================================
# RUN ONE TEST
# ============================================================
def run_test(name, sampler, n_vars, n_eqs_expected, noise_frac=0.0):
    np.random.seed(42)
    n_pts = 800 if n_vars <= 3 else 1500
    X_raw = sampler(n_pts)

    # Standardize
    mu = X_raw.mean(axis=0)
    sd = X_raw.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xs = (X_raw - mu) / sd

    # Add noise AFTER standardization (proportional to std=1)
    if noise_frac > 0:
        Xs = Xs + np.random.randn(*Xs.shape) * noise_frac

    t0 = time.time()
    try:
        res = discover_equations(Xs, n_equations=n_eqs_expected, degree=2,
                                 tol_ratio=0.05, use_trig=True)
        eqs = res[0]
    except Exception as e:
        return {"name": name, "noise": noise_frac, "status": "error",
                "error": str(e)[:60], "time": time.time()-t0,
                "n_found": 0, "resid": None, "p95": None}

    elapsed = time.time() - t0

    if not eqs:
        return {"name": name, "noise": noise_frac, "status": "no_eq",
                "time": elapsed, "n_found": 0, "resid": None, "p95": None}

    # Evaluate each equation with SAME mu/sd
    results = []
    for eq in eqs:
        m, p = evaluate_with_transform(eq, sampler, mu, sd, n_vars)
        if m is not None:
            results.append((m, p))

    if not results:
        return {"name": name, "noise": noise_frac, "status": "eval_fail",
                "time": elapsed, "n_found": len(eqs), "resid": None, "p95": None}

    best_med = min(r[0] for r in results)
    best_p95 = min(r[1] for r in results)

    if best_med < 0.05:
        status = "excellent"
    elif best_med < 0.15:
        status = "good"
    elif best_med < 0.5:
        status = "ok"
    else:
        status = "poor"

    return {"name": name, "noise": noise_frac, "status": status,
            "time": elapsed, "n_found": len(eqs),
            "resid": best_med, "p95": best_p95}


# ============================================================
# RUN ALL
# ============================================================
NOISE_LEVELS = [0.00, 0.05, 0.10, 0.15, 0.20]
all_results = []

print("=" * 100)
print("  IMPLICIT MANIFOLD BENCHMARK v2 (FIXED)")
print("=" * 100)
print()
print("Fixes applied:")
print("  - Evaluation uses SAME mu/sd as training")
print("  - Noise added AFTER standardization")
print("  - Helix uses random sampling (not linspace)")
print("  - safe_sort for free_symbols")
print("  - graceful lambdify error handling")
print()

for noise in NOISE_LEVELS:
    print(f"\n{'='*100}")
    print(f"  NOISE = {100*noise:.0f}%")
    print(f"{'='*100}")
    print(f"{'test':<12} {'n_found':>8} {'resid':>10} {'p95':>10} "
          f"{'status':>10} {'time':>8}")
    print("-" * 75)

    for name, func, nv, neq in TESTS:
        r = run_test(name, func, nv, neq, noise_frac=noise)
        all_results.append(r)
        m_str = f"{r['resid']:.4f}" if r['resid'] is not None else "  --  "
        p_str = f"{r['p95']:.4f}" if r['p95'] is not None else "  --  "
        print(f"{name:<12} {r['n_found']:>8} {m_str:>10} {p_str:>10} "
              f"{r['status']:>10} {r['time']:>7.2f}s")
    print()

# ============================================================
# NOISE ROBUSTNESS SUMMARY
# ============================================================
print("=" * 100)
print("  NOISE ROBUSTNESS SUMMARY")
print("=" * 100)
print()
print(f"{'noise':>6} {'excellent':>10} {'good':>6} {'ok':>5} {'poor':>6} "
      f"{'fail':>6} {'median_resid':>15}")
print("-" * 75)

for noise in NOISE_LEVELS:
    subset = [r for r in all_results if r["noise"] == noise]
    n = len(subset)
    exc = sum(1 for r in subset if r["status"] == "excellent")
    gd  = sum(1 for r in subset if r["status"] == "good")
    ok  = sum(1 for r in subset if r["status"] == "ok")
    pr  = sum(1 for r in subset if r["status"] == "poor")
    fail = n - exc - gd - ok - pr
    medians = [r["resid"] for r in subset if r["resid"] is not None]
    med_str = f"{np.median(medians):.4f}" if medians else "  --  "
    print(f"{100*noise:>5.0f}% {exc:>10} {gd:>6} {ok:>5} {pr:>6} {fail:>6} "
          f"{med_str:>15}")

# ============================================================
# PER-TEST TREND
# ============================================================
print()
print("=" * 100)
print("  PER-TEST: median residual across noise levels")
print("=" * 100)
print()
header = f"{'test':<12}"
for nz in NOISE_LEVELS:
    header += f" {100*nz:>5.0f}%"
header += "  worst_status"
print(header)
print("-" * (12 + 7*len(NOISE_LEVELS) + 15))

for name, _, _, _ in TESTS:
    row = sorted([r for r in all_results if r["name"] == name],
                 key=lambda r: r["noise"])
    line = f"{name:<12}"
    for r in row:
        m_str = f"{r['resid']:.3f}" if r['resid'] is not None else "--"
        line += f" {m_str:>5}"
    order = {"excellent": 0, "good": 1, "ok": 2, "poor": 3,
             "no_eq": 4, "error": 5, "eval_fail": 5}
    worst = max((r["status"] for r in row), key=lambda s: order.get(s, 5))
    line += f"  {worst}"
    print(line)

# Save
out = os.path.expanduser("~/mathbot/benchmark_v2_results.csv")
with open(out, 'w') as f:
    f.write("name,noise,n_found,resid,p95,status,time\n")
    for r in all_results:
        f.write(f"{r['name']},{r['noise']},{r['n_found']},"
                f"{r['resid']},{r['p95']},{r['status']},{r['time']:.3f}\n")

print()
print(f"Results: {out}")
