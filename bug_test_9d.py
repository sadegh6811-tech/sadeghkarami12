#!/usr/bin/env python3
"""Bug test for 9D formula on parallel universes — BEFORE running anything."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.expanduser("~/mathbot"))

print("=" * 75)
print("  BUG TEST: 9D Formula on Parallel Universes")
print("=" * 75)
print()

# ============================================================
# BUG 1: What is actually inside parallel_universes.npy?
# ============================================================
print("BUG 1: Inspecting parallel_universes.npy")
print()

path = os.path.expanduser("~/parallel_universes.npy")
data = np.load(path, allow_pickle=True)
print(f"  Top-level: type={type(data)}, shape={data.shape if hasattr(data,'shape') else 'N/A'}")
print(f"  First element type: {type(data[0])}")

first = data[0]
if hasattr(first, 'shape'):
    print(f"  First universe: shape={first.shape}, dtype={first.dtype}")
    print(f"  Min: {first.min(axis=0)}")
    print(f"  Max: {first.max(axis=0)}")
    print(f"  First 3 rows:")
    print(first[:3])
else:
    print(f"  First: {first}")
print()

# ============================================================
# BUG 2: THE CRITICAL ONE — is the formula actually meaningful?
# ============================================================
print("BUG 2: Formula structure analysis")
print()

BEST_TREE = ('div',
    ('add', ('add', 'x7', 'x2'), ('const', 0.0)),
    ('add',
        ('add', ('abs', ('mul', ('sub', ('add', 'x6', 'x1'), ('sub', 'x1', 'x0')),
                         ('div', ('div', 'x1', 'x6'), ('abs', 'x0'))))),
            ('div', ('const', -1.0), 'x5')
        ),
        ('abs', ('sub', ('abs', 'x2'), ('div', ('const', 0.5), 'x7')))
    )
)

print("  Let's simplify the formula symbolically (what it ACTUALLY does):")
print()

# Manually trace
print("  Nominator: (x7 + x2) + 0  →  x7 + x2")
print("  Denominator:")
print("    term_A = |(x6 + x1 - (x1 - x0)) * ((x1/x6) / |x0|)|")
print("           = |(x6 + x0) * (x1 / (x6 * |x0|))|")
print("           = |(x6 + x0) * x1 / (x6 * |x0|)|")
print()
print("    term_B = -1/x5")
print("    term_C = | |x2| - 0.5/x7 |")
print()
print("    denominator = term_A + term_B + term_C")
print()
print("  CRITICAL OBSERVATION:")
print("    x5 = np.ones() → constant 1 → term_B = -1")
print("    x6 = np.zeros() → constant 0 → term_A = |x1/(0*|x0|)| = inf!")
print()
print("  → x6 is ZERO everywhere → division by zero in term_A")
print("  → Formula is MALFORMED for any input that uses x6 in denominator")
print()

# Test this numerically
def evaluate_tree(tree, X):
    if isinstance(tree, str):
        if tree.startswith('x'):
            idx = int(tree[1:])
            return X[:, idx] if idx < X.shape[1] else np.zeros(X.shape[0])
        return np.zeros(X.shape[0])
    if tree[0] == 'const':
        return np.full(X.shape[0], float(tree[1]))
    if tree[0] == 'add':
        return evaluate_tree(tree[1], X) + evaluate_tree(tree[2], X)
    if tree[0] == 'sub':
        return evaluate_tree(tree[1], X) - evaluate_tree(tree[2], X)
    if tree[0] == 'mul':
        return evaluate_tree(tree[1], X) * evaluate_tree(tree[2], X)
    if tree[0] == 'div':
        a = evaluate_tree(tree[1], X)
        b = evaluate_tree(tree[2], X)
        return a / (np.abs(b) + 1e-9)
    if tree[0] == 'abs':
        return np.abs(evaluate_tree(tree[1], X))
    return np.zeros(X.shape[0])

# Simulate the X constructed in test_universes_formula.py
n = 100
phi = np.random.randn(n)
phi_dot = np.random.randn(n)
phi_ddot = np.random.randn(n)

X = np.column_stack([
    (phi - np.mean(phi)) / (np.std(phi) + 1e-9),           # x0
    np.sign(np.gradient(phi)),                              # x1
    np.abs(phi_dot) / (np.std(phi_dot) + 1e-9),            # x2
    np.abs(phi_ddot) / (np.std(phi_ddot) + 1e-9),          # x3
    np.abs(phi * phi_dot) / (np.std(phi * phi_dot) + 1e-9),# x4
    np.ones_like(phi),                                       # x5
    np.zeros_like(phi),                                      # x6
    np.gradient(phi) / (np.std(np.gradient(phi)) + 1e-9),  # x7
    np.linspace(0, 2*np.pi, n)                               # x8
])

print("  Manual inspection of X columns:")
print(f"    x0: range [{X[:,0].min():.3f}, {X[:,0].max():.3f}]")
print(f"    x1: unique values = {np.unique(X[:,1])}")  # sign(gradient) → {-1, 0, 1}
print(f"    x2: range [{X[:,2].min():.3f}, {X[:,2].max():.3f}]")
print(f"    x3: range [{X[:,3].min():.3f}, {X[:,3].max():.3f}]")
print(f"    x4: range [{X[:,4].min():.3f}, {X[:,4].max():.3f}]")
print(f"    x5: unique = {np.unique(X[:,5])}")  # all 1
print(f"    x6: unique = {np.unique(X[:,6])}")  # all 0
print(f"    x7: range [{X[:,7].min():.3f}, {X[:,7].max():.3f}]")
print(f"    x8: range [{X[:,8].min():.3f}, {X[:,8].max():.3f}]")
print()

# Actually evaluate
try:
    out = evaluate_tree(BEST_TREE, X)
    print(f"  Output range: [{out.min():.4e}, {out.max():.4e}]")
    print(f"  Output mean: {out.mean():.4e}, std: {out.std():.4e}")
    print(f"  Any inf/nan? nan={np.isnan(out).sum()}, inf={np.isinf(out).sum()}")
    print()

    # Simple check: what's the distribution of nonzero outputs?
    nonzero = out[np.abs(out) > 0.1]
    print(f"  Outputs with |out| > 0.1: {len(nonzero)} / {len(out)}")
    if len(nonzero) > 0:
        print(f"    These are likely driven by the 1e-9 epsilon in denominator")
        print(f"    (division by ~0 when x6=0)")
except Exception as e:
    print(f"  ERROR: {e}")
print()

# ============================================================
# BUG 3: What does x5=1, x6=0 mean semantically?
# ============================================================
print("BUG 3: Semantic problem")
print()
print("  In test_universes_formula.py:")
print("    x5 = np.ones_like(phi)   # labeled 'buy'")
print("    x6 = np.zeros_like(phi)  # labeled 'sell'")
print()
print("  These are CONSTANT features (all 1s, all 0s).")
print("  They were designed for forex (buy=1 means 'we want to buy').")
print("  In cosmology (phi, phi_dot, phi_ddot), these have NO PHYSICAL MEANING.")
print()
print("  → Formula uses x5 and x6 which are Forex-specific constants")
print("  → Any result is meaningless on cosmology data")
print()

# ============================================================
# BUG 4: What is parallel_universes.npy actually?
# ============================================================
print("BUG 4: Source of parallel_universes.npy")
print()

# Analyze first few universes
print("  Looking for patterns in the data:")
for i in [0, 1, 2]:
    u = data[i]
    print(f"    Universe {i}: shape={u.shape}, dtype={u.dtype}")
    print(f"      column means: {u.mean(axis=0)}")
    print(f"      column stds:  {u.std(axis=0)}")
    print(f"      first row:    {u[0]}")
    print()

# Check if it's deterministic / repeated
if len(data) > 1:
    u0, u1 = data[0], data[1]
    if u0.shape == u1.shape:
        diff = np.abs(u0 - u1).mean()
        print(f"  Mean |universe[0] - universe[1]| = {diff:.6f}")
        if diff < 1e-6:
            print("  ⚠️  Universes are IDENTICAL — data is duplicated!")
        else:
            print(f"  ✓ Universes differ")
print()

# ============================================================
# BUG 5: Does the formula output anything that looks physical?
# ============================================================
print("BUG 5: Formula output sanity")
print()
print("  Expected behavior IF formula is a '9D boundary law':")
print("    - Should identify specific phi values (boundaries)")
print("    - Should relate to phi, phi_dot, phi_ddot in a meaningful way")
print()
print("  Expected behavior IF formula is forex-specific:")
print("    - Output depends on x5 (buy=1), x6 (sell=0)")
print("    - Output dominated by 1e-9 epsilon (division by ~0)")
print("    - Same output for all universes (since x5, x6 are constant)")
print()

# Test: apply formula to 3 different universes and compare
outs = []
for i in [0, 1, 2]:
    u = data[i]
    if u.shape[1] < 3: continue
    phi = u[:, 0]; phi_dot = u[:, 1]; phi_ddot = u[:, 2]
    X = np.column_stack([
        (phi - np.mean(phi)) / (np.std(phi) + 1e-9),
        np.sign(np.gradient(phi)),
        np.abs(phi_dot) / (np.std(phi_dot) + 1e-9),
        np.abs(phi_ddot) / (np.std(phi_ddot) + 1e-9),
        np.abs(phi * phi_dot) / (np.std(phi * phi_dot) + 1e-9),
        np.ones_like(phi),
        np.zeros_like(phi),
        np.gradient(phi) / (np.std(np.gradient(phi)) + 1e-9),
        np.linspace(0, 2*np.pi, len(phi))
    ])
    o = evaluate_tree(BEST_TREE, X)
    outs.append(o)

if len(outs) > 1:
    for i in [1, 2]:
        if len(outs[i]) == len(outs[0]):
            diff = np.abs(outs[i] - outs[0]).mean()
            print(f"  |output_universe_{i} - output_universe_0| = {diff:.6e}")
print()

# ============================================================
# BUG 6: The formula might have no x3, x4, x8 in it at all
# ============================================================
print("BUG 6: Unused features")
print()
print("  Formula variables used: x0, x1, x2, x5, x6, x7")
print("  Features NOT used:      x3, x4, x8")
print()
print("  So features 3 (phi_ddot), 4 (phi*phi_dot), 8 (time) are ignored.")
print("  → Formula doesn't actually model cosmology")
print()

# ============================================================
# FINAL SUMMARY
# ============================================================
print("=" * 75)
print("  SUMMARY OF BUGS")
print("=" * 75)
print()
print("CRITICAL:")
print("  1. Formula uses x5 (constant 1) and x6 (constant 0)")
print("     → x6=0 causes division by ~0 in denominator")
print("     → Output is noise from 1e-9 epsilon, not real signal")
print()
print("  2. Formula was designed for forex (x5='buy', x6='sell')")
print("     → Applying to cosmological data is category error")
print()
print("  3. Features x3, x4, x8 (phi_ddot, phi*phi_dot, time) are IGNORED")
print("     → Formula doesn't touch most of the data")
print()
print("SERIOUS:")
print("  4. Formula has no theoretical basis for cosmology")
print("  5. x1 = sign(gradient(phi)) → only 3 values {-1, 0, 1}")
print("     → Formula behaves like a step function, not a physical law")
print()
print("MINOR:")
print("  6. parallel_universes.npy origin unknown")
print("     → Should trace back to source before drawing conclusions")
print()
print("CONCLUSION:")
print("  The formula is a forex trading signal, NOT a cosmology law.")
print("  Applying it to parallel universes tests nothing meaningful.")
print("  If you want to test a 9D theory, you need:")
print("    - A formula that uses physics features (phi, phi_dot, phi_ddot)")
print("    - Data with real cosmological meaning")
print("    - A null hypothesis (random forest / random baseline)")
