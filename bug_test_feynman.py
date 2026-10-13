#!/usr/bin/env python3
"""Bug test for Feynman benchmark — find architectural issues BEFORE running."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.expanduser("~/mathbot"))
from discover import discover_equations

np.random.seed(42)

print("=" * 90)
print("  BUG TEST: Feynman benchmark architectural issues")
print("=" * 90)
print()

# ============================================================
# BUG 1: `f"x{n_vars}" in eq_str` is fragile
# ============================================================
print("BUG 1: 'involves_y' check is fragile")
print()
print("  Original check: f'x{n_vars}' in str(eq)")
print()
print("  Problem 1: if n_vars=10, checks for 'x10' but 'x1' also matches inside 'x10'")
print("  Problem 2: if n_vars=1, checks for 'x1' but equation might use 'x0' only")
print("  Problem 3: sympy might print as 'x_0' not 'x0'")
print()
print("  Fix: use sympy's free_symbols to check if the y variable is present")
print()

# ============================================================
# BUG 2: MathBot was designed for IMPLICIT manifolds, not y = f(x)
# ============================================================
print("BUG 2: Architectural mismatch")
print()
print("  MathBot: designed for g(x0, x1, ..., xn) = 0  (sphere, torus, helix)")
print("  We need: y = f(x0, x1, ..., xn-1)  (explicit function)")
print()
print("  Both can be expressed as g(x0, ..., xn) = 0, but the discovery")
print("  process may find trivial relations in the x's, not involving y at all")
print()

# ============================================================
# BUG 3: Degree 2 is insufficient for most Feynman equations
# ============================================================
print("BUG 3: Degree limitation")
print()
# Count equations by required degree
equations = {
    "poly degree 1": "x0*x1",
    "poly degree 2": "x0**2 + x1*x2",
    "poly degree 3": "x0*x1*x2",
    "poly degree 4": "x0**4 + x1**2",
    "rational": "x0/(x1+x2)",
    "sqrt": "sqrt(x0**2 + x1**2)",
    "trig": "sin(x0) + cos(x1)",
    "exp": "exp(-x0**2/2)",
    "log": "log(x0/x1)",
}
print(f"  Current MathBot: degree=2 only")
print(f"  Equations needing degree 3+: {len([k for k in equations if 'degree 3' in k or 'degree 4' in k])}/9")
print(f"  Equations needing rational/sqrt/exp/log: {len([k for k in equations if k in ('rational','sqrt','exp','log')])}/9")
print()
print("  Impact: degree 2 alone can solve maybe 30% of Feynman")
print()

# ============================================================
# BUG 4: use_trig=True may trigger buggy frequency detector
# ============================================================
print("BUG 4: use_trig=True with buggy freq detector")
print()
print("  Earlier in this session, we saw 'spurious frequencies' in output")
print("  like 'x0^2 + x1^2 ≈ 2.000' when data wasn't circular")
print("  This could add noise columns to the library")
print()

# ============================================================
# ACTUAL TEST: run 3 sample equations and see what happens
# ============================================================
print("=" * 90)
print("  RUNTIME TEST: 3 sample equations")
print("=" * 90)
print()

def gen(func, n_vars, n_pts=500, noise=0.04):
    X = np.random.randn(n_pts, n_vars)
    X = np.clip(X, -3, 3)
    try:
        y = func(X)
        if np.isscalar(y): y = np.full(n_pts, y)
        y = np.asarray(y, dtype=float).reshape(-1)
    except:
        return None, None
    if np.isnan(y).any() or np.isinf(y).any(): return None, None
    ys = y.std()
    if ys < 1e-12: return None, None
    y = y / ys
    y = y + np.random.randn(n_pts) * noise
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    return X, y

samples = [
    ("poly_2d",  lambda x: x[:,0]**2 + x[:,1], 2),
    ("trig_2d",  lambda x: np.sin(x[:,0]) + np.cos(x[:,1]), 2),
    ("exp_1d",   lambda x: np.exp(-x[:,0]**2/2), 1),
]

for name, func, nv in samples:
    print(f"--- {name} (n_vars={nv}) ---")
    X, y = gen(func, nv)
    if X is None:
        print("  data generation failed")
        continue
    data = np.column_stack([X, y])
    print(f"  data shape: {data.shape}")
    print(f"  X range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"  y range: [{y.min():.3f}, {y.max():.3f}]")
    
    try:
        import time
        t0 = time.time()
        res = discover_equations(data, n_equations=1, degree=2, tol_ratio=0.05, use_trig=True)
        t = time.time() - t0
        eqs = res[0]
        print(f"  time: {t:.2f}s")
        print(f"  equations found: {len(eqs)}")
        if eqs:
            eq_str = str(eqs[0])
            print(f"  equation: {eq_str[:200]}")
            # Check: does it involve y (should be x{nv})?
            y_sym = f"x{nv}"
            involves_y = y_sym in eq_str
            print(f"  involves y ({y_sym})? {involves_y}")
            
            # Better check: use sympy
            from sympy import Symbol
            try:
                syms = eqs[0].free_symbols
                names = sorted([str(s) for s in syms])
                print(f"  free symbols: {names}")
                print(f"  correct: {y_sym in names}")
            except Exception as e:
                print(f"  sympy check failed: {e}")
        else:
            print(f"  (no equations)")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
    print()

# ============================================================
# SUMMARY
# ============================================================
print("=" * 90)
print("  ISSUES SUMMARY")
print("=" * 90)
print()
print("Critical:")
print("  1. 'involves_y' check uses string match — fragile")
print("  2. degree=2 limits coverage to ~30% of Feynman")
print("  3. use_trig=True may add spurious features")
print()
print("For accurate benchmark, need to:")
print("  - Fix involves_y check via sympy.free_symbols")
print("  - Try degree 2 AND 3 AND 4")
print("  - Report per-degree results")
print("  - Skip equations where y is essentially constant")
