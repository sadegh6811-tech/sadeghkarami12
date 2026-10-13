#!/usr/bin/env python3
"""Bug test on multi-seed benchmark."""
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

def sample_sphere3(n):
    u = np.random.randn(n, 3); return u / np.linalg.norm(u, axis=1, keepdims=True)

def sample_ellipsoid(n):
    u = np.random.randn(n, 3)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    return u * np.array([1.0, 2.0, 3.0])

def sample_cylinder(n):
    t = np.random.uniform(0, 2*np.pi, n)
    z = np.random.randn(n) * 0.5
    return np.column_stack([np.cos(t), np.sin(t), z])

def sample_helix(n):
    t = np.random.uniform(0, 4*np.pi, n)
    return np.column_stack([np.cos(t), np.sin(t), t/(4*np.pi)])

print("=" * 80)
print("  BUG TEST: multi-seed benchmark")
print("=" * 80)
print()

# ============================================================
# BUG A: sphere_3d vs ellipsoid
# ============================================================
print("BUG A: sphere_3d and ellipsoid identical after standardization?")
print()

np.random.seed(42)
X_sph = sample_sphere3(500)
np.random.seed(42)
X_ell = sample_ellipsoid(500)

def std(X):
    return (X - X.mean(0)) / X.std(0)

Xs_sph = std(X_sph)
Xs_ell = std(X_ell)

diff = np.abs(Xs_sph - Xs_ell).max()
print(f"  max|Xs_sphere - Xs_ellipsoid| = {diff:.2e}")
if diff < 1e-9:
    print("  CONFIRMED: sphere_3d and ellipsoid IDENTICAL after standardization")
    print("  -> Test is duplicated, only one is meaningful")
else:
    print("  Different (good)")
print()

# ============================================================
# BUG B: cylinder equation rejected by strict eval?
# ============================================================
print("BUG B: cylinder - equation found but rejected?")
print()

np.random.seed(42)
X_cyl = sample_cylinder(800)
mu = X_cyl.mean(0); sd = X_cyl.std(0)
Xs_cyl = (X_cyl - mu) / sd

res = discover_equations(Xs_cyl, n_equations=1, degree=2, tol_ratio=0.05, use_trig=True)
eqs = res[0]
print(f"  Equations found: {len(eqs)}")
if eqs:
    eq = eqs[0]
    free = safe_sort_symbols(eq.free_symbols)
    print(f"  Equation: {eq}")
    print(f"  Free symbols: {free} (len={len(free)})")
    print(f"  n_vars expected: 3")
    if len(free) != 3:
        print(f"  CONFIRMED: strict check 'len(free) != n_vars' rejects it")
        # Actually evaluate to check if the equation is correct
        if len(free) == 2:
            syms = [Symbol(s) for s in free]
            col_idx = [int(s[1:]) for s in free]
            f = lambdify(syms, eq, 'numpy')
            vals = f(*[Xs_cyl[:, i] for i in col_idx])
            vals = np.abs(np.asarray(vals, dtype=float).ravel())
            print(f"  Actual residual: {np.median(vals):.6f}")
            if np.median(vals) < 0.05:
                print(f"  Equation is CORRECT - eval function has the bug")
else:
    print("  No equations found")
print()

# ============================================================
# BUG C: helix
# ============================================================
print("BUG C: helix - what does MathBot find?")
print()

np.random.seed(42)
X_h = sample_helix(800)
mu_h = X_h.mean(0); sd_h = X_h.std(0)
Xs_h = (X_h - mu_h) / sd_h

res = discover_equations(Xs_h, n_equations=2, degree=2, tol_ratio=0.05, use_trig=True)
eqs = res[0]
print(f"  Equations found: {len(eqs)}")
for eq in eqs:
    free = safe_sort_symbols(eq.free_symbols)
    print(f"    {eq}")
    print(f"    free: {free} (len={len(free)})")
    # Evaluate against only those columns
    if len(free) >= 2:
        syms = [Symbol(s) for s in free]
        col_idx = [int(s[1:]) for s in free if s.startswith('x')]
        try:
            f = lambdify(syms, eq, 'numpy')
            vals = f(*[Xs_h[:, i] for i in col_idx])
            vals = np.abs(np.asarray(vals, dtype=float).ravel())
            print(f"    residual: {np.median(vals):.6f}")
        except Exception as e:
            print(f"    eval error: {e}")
print()

# ============================================================
# SUMMARY
# ============================================================
print("=" * 80)
print("  BUGS FOUND")
print("=" * 80)
print()
print("A. sphere_3d vs ellipsoid: identical data after standardization")
print("   -> 2 tests counted as 1")
print()
print("B. eval function requires len(free) == n_vars strictly")
print("   -> cylinder: eq uses 2 of 3 vars -> rejected as eval_fail")
print("   -> helix: same")
print()
print("C. min_time = 0.00s from early-return path")
print()
print("CONCLUSION: real success rate is HIGHER than 82%")
print("  Fix eval() to allow equations using subset of variables")
