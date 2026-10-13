#!/usr/bin/env python3
"""Bug test for implicit benchmark — architectural + logical issues."""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.expanduser("~/mathbot"))
from discover import discover_equations
from sympy import lambdify, Symbol

np.random.seed(42)

print("=" * 90)
print("  BUG TEST: Implicit benchmark")
print("=" * 90)
print()

# ============================================================
print("BUG 1 (CRITICAL): Evaluation mismatch — raw vs standardized space")
print()
print("  MathBot receives X_standardized = (X - mu) / sd")
print("  Discovered equation g is in the standardized space")
print("  But evaluate_equation_numeric does:")
print("    X_test = sample_func(300)   ← raw, unstandardized")
print("    g(X_test)                    ← wrong space!")
print()

# Demo
n = 500
t = np.random.uniform(0, 2*np.pi, n)
X_raw = np.column_stack([np.cos(t), np.sin(t)])
mu = X_raw.mean(axis=0)
sd = X_raw.std(axis=0)
X_std = (X_raw - mu) / sd

print(f"  circle_2d example:")
print(f"    raw point:          ({X_raw[0,0]:.4f}, {X_raw[0,1]:.4f})")
print(f"    standardized point: ({X_std[0,0]:.4f}, {X_std[0,1]:.4f})")
print(f"    x0²+x1² raw:         {X_raw[0,0]**2 + X_raw[0,1]**2:.4f}")
print(f"    x0²+x1² standardized:{X_std[0,0]**2 + X_std[0,1]**2:.4f}")
print()

# ============================================================
print("BUG 2: Noise added BEFORE standardization")
print()
X_test = np.column_stack([np.random.randn(100), np.random.randn(100)*0.01])
X_test_noisy = X_test + np.random.randn(100, 2) * X_test.std(axis=0) * 0.05
sd_before = X_test.std(axis=0)
sd_after = X_test_noisy.std(axis=0)
print(f"  sd before noise: {sd_before}")
print(f"  sd after noise:  {sd_after}")
print(f"  amplification:   {sd_after/sd_before}")
print()

# ============================================================
print("BUG 3: sympy free_symbols sorting crashes on 'y', 'z'")
print()
def safe_sort(symbols):
    def keyfn(s):
        try:
            return int(s[1:])
        except (ValueError, IndexError):
            return 999
    return sorted(symbols, key=keyfn)

test_syms = ['x2', 'x0', 'y', 'z']
print(f"  symbols: {test_syms}")
print(f"  safe sort: {safe_sort(test_syms)}")
print()

# ============================================================
print("BUG 4: Helix uses linspace (deterministic, not random)")
print()
t_helix = np.linspace(0, 4*np.pi, 800)
print(f"  Helix: linspace(0, 4π, 800) — ordered")
print(f"  SVD null-space assumes i.i.d.; ordered data may break it")
print()

# ============================================================
print("BUG 5: lambdify can fail on some sympy functions")
print()
print("  MathBot may produce equations with Abs, sign, erf")
print("  numpy lambdify supports most, but not all")
print("  → evaluate_equation_numeric silently returns None")
print()

# ============================================================
# RUNTIME DEMO: circle_2d
# ============================================================
print("=" * 90)
print("  RUNTIME DEMO: circle_2d")
print("=" * 90)
print()

n = 500
t = np.random.uniform(0, 2*np.pi, n)
X_raw = np.column_stack([np.cos(t), np.sin(t)])
mu = X_raw.mean(0)
sd = X_raw.std(0)
Xs = (X_raw - mu) / sd

print(f"  mu={mu}, sd={sd}")
print()

try:
    t0 = time.time()
    res = discover_equations(Xs, n_equations=1, degree=2, tol_ratio=0.05, use_trig=False)
    elapsed = time.time() - t0
    eqs = res[0]
    print(f"  time: {elapsed:.2f}s")
    print(f"  equations found: {len(eqs)}")

    if eqs:
        eq = eqs[0]
        print(f"  equation: {eq}")
        free = safe_sort([str(s) for s in eq.free_symbols])
        print(f"  free symbols: {free}")
        print()

        syms = [Symbol(s) for s in free]
        f = lambdify(syms, eq, 'numpy')

        # Buggy: raw space
        try:
            vals_raw = f(*[X_raw[:, i] for i in range(len(syms))])
            vals_raw = np.abs(np.asarray(vals_raw, dtype=float).ravel())
            print(f"  BUGGY eval (raw points):")
            print(f"    median |g(x)| = {np.median(vals_raw):.4f}")
        except Exception as e:
            print(f"  raw eval failed: {e}")

        # Correct: standardized space
        try:
            vals_std = f(*[Xs[:, i] for i in range(len(syms))])
            vals_std = np.abs(np.asarray(vals_std, dtype=float).ravel())
            print(f"  CORRECT eval (standardized points):")
            print(f"    median |g(x)| = {np.median(vals_std):.4f}")
        except Exception as e:
            print(f"  std eval failed: {e}")
except Exception as e:
    print(f"  ERROR: {type(e).__name__}: {e}")

print()
print("=" * 90)
print("  SUMMARY")
print("=" * 90)
print()
print("CRITICAL bugs found:")
print("  1. Evaluation in wrong space (raw vs standardized)")
print("  2. Different mu/sd between train and test sample")
print("  3. Noise amplification (added before standardization)")
print("  4. Helix sampling is deterministic (linspace)")
print("  5. lambdify failures silently give 'eval_fail'")
print("  6. free_symbols sorting crashes on 'y', 'z'")
print()
print("Benchmark as-written is INVALID.")
print("Fix requires: save mu/sd, apply same transform to test, noise after std.")
