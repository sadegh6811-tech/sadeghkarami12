#!/usr/bin/env python3
"""Test the 9D formula on parallel universes."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.expanduser("~/mathbot"))

# ---- Load data ----
path = os.path.expanduser("~/parallel_universes.npy")
universes = np.load(path, allow_pickle=True)
print(f"Loaded {len(universes)} universes")
print(f"Shape: {universes.shape}")
print(f"First universe shape: {universes[0].shape}")
print()

# ---- Formula tree ----
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

def evaluate_tree(tree, X):
    """Evaluate formula tree."""
    if isinstance(tree, str):
        if tree.startswith('x'):
            try:
                idx = int(tree[1:])
                return X[:, idx].astype(float)
            except (ValueError, IndexError):
                return np.zeros(X.shape[0])
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

# ---- Build 9D features and apply ----
print("=" * 70)
print("  Applying 9D formula to universes")
print("=" * 70)
print()

all_outputs = []
for i, u in enumerate(universes[:100]):
    phi = u[:, 0]
    phi_dot = u[:, 1]
    phi_ddot = u[:, 2]

    # Build 9D features (exactly as in original file)
    x0 = (phi - np.mean(phi)) / (np.std(phi) + 1e-9)
    x1 = np.sign(np.gradient(phi))
    x2 = np.abs(phi_dot) / (np.std(phi_dot) + 1e-9)
    x3 = np.abs(phi_ddot) / (np.std(phi_ddot) + 1e-9)
    x4 = np.abs(phi * phi_dot) / (np.std(phi * phi_dot) + 1e-9)
    x5 = np.ones_like(phi)
    x6 = np.zeros_like(phi)
    x7 = np.gradient(phi) / (np.std(np.gradient(phi)) + 1e-9)
    x8 = np.linspace(0, 2 * np.pi, len(phi))

    X = np.column_stack([x0, x1, x2, x3, x4, x5, x6, x7, x8])
    out = evaluate_tree(BEST_TREE, X)
    all_outputs.append(out)

all_outputs = np.concatenate(all_outputs)
print(f"Total points: {len(all_outputs)}")
print(f"Output min:  {all_outputs.min():.4e}")
print(f"Output max:  {all_outputs.max():.4e}")
print(f"Output mean: {all_outputs.mean():.4e}")
print(f"Output std:  {all_outputs.std():.4e}")
print(f"NaN count:   {np.isnan(all_outputs).sum()}")
print(f"Inf count:   {np.isinf(all_outputs).sum()}")
print()

# ---- Signal counts ----
up = (all_outputs > 0.1).sum()
down = (all_outputs < -0.1).sum()
neutral = len(all_outputs) - up - down

print("=" * 70)
print("  Signal Analysis")
print("=" * 70)
print(f"  UP signals:      {up} ({100*up/len(all_outputs):.1f}%)")
print(f"  DOWN signals:    {down} ({100*down/len(all_outputs):.1f}%)")
print(f"  NEUTRAL signals: {neutral} ({100*neutral/len(all_outputs):.1f}%)")
print()

# ---- Correlation with physics ----
print("=" * 70)
print("  Correlation with physics quantities")
print("=" * 70)
print()

for i in [0, 1, 2, 5, 10, 20, 50, 99]:
    if i >= len(universes): continue
    u = universes[i]
    phi = u[:, 0]; phi_dot = u[:, 1]; phi_ddot = u[:, 2]

    x0 = (phi - np.mean(phi)) / (np.std(phi) + 1e-9)
    x1 = np.sign(np.gradient(phi))
    x2 = np.abs(phi_dot) / (np.std(phi_dot) + 1e-9)
    x3 = np.abs(phi_ddot) / (np.std(phi_ddot) + 1e-9)
    x4 = np.abs(phi * phi_dot) / (np.std(phi * phi_dot) + 1e-9)
    x5 = np.ones_like(phi)
    x6 = np.zeros_like(phi)
    x7 = np.gradient(phi) / (np.std(np.gradient(phi)) + 1e-9)
    x8 = np.linspace(0, 2 * np.pi, len(phi))

    X = np.column_stack([x0, x1, x2, x3, x4, x5, x6, x7, x8])
    out = evaluate_tree(BEST_TREE, X)

    try:
        c_phi = np.corrcoef(out, phi)[0, 1] if np.std(out) > 1e-15 else 0
        c_dot = np.corrcoef(out, phi_dot)[0, 1] if np.std(out) > 1e-15 else 0
        c_ddot = np.corrcoef(out, phi_ddot)[0, 1] if np.std(out) > 1e-15 else 0
    except Exception:
        c_phi = c_dot = c_ddot = 0

    print(f"Universe {i:3d}: corr(out, phi)={c_phi:+.3f}  "
          f"corr(out, phi_dot)={c_dot:+.3f}  corr(out, phi_ddot)={c_ddot:+.3f}")

# ---- Check formula's x5, x6 ----
print()
print("=" * 70)
print("  Formula structure check")
print("=" * 70)
print()
print("Features used in formula:")
print("  x0 = normalized phi")
print("  x1 = sign(gradient(phi))")
print("  x2 = |phi_dot|")
print("  x5 = ones (buy flag, constant)")
print("  x6 = zeros (sell flag, CONSTANT = 0)")
print("  x7 = gradient(phi)")
print()
print("Features NOT used:")
print("  x3 (phi_ddot), x4 (phi*phi_dot), x8 (time)")
print()
print("CRITICAL: x6 = 0 is used in denominator")
print("  term_A = |(x6 + x0) * (x1/x6) / |x0||")
print("  → x1/x6 = x1/0 = inf")
print("  → denominator depends on 1e-9 epsilon")
print()

# Check if outputs differ across universes
print("=" * 70)
print("  Do outputs vary across universes?")
print("=" * 70)
print()

# Apply to all universes, compare
outputs_by_universe = []
for u in universes[:10]:
    phi = u[:, 0]; phi_dot = u[:, 1]; phi_ddot = u[:, 2]
    n = len(phi)

    x0 = (phi - np.mean(phi)) / (np.std(phi) + 1e-9)
    x1 = np.sign(np.gradient(phi))
    x2 = np.abs(phi_dot) / (np.std(phi_dot) + 1e-9)
    x3 = np.abs(phi_ddot) / (np.std(phi_ddot) + 1e-9)
    x4 = np.abs(phi * phi_dot) / (np.std(phi * phi_dot) + 1e-9)
    x5 = np.ones(n)
    x6 = np.zeros(n)
    x7 = np.gradient(phi) / (np.std(np.gradient(phi)) + 1e-9)
    x8 = np.linspace(0, 2 * np.pi, n)

    X = np.column_stack([x0, x1, x2, x3, x4, x5, x6, x7, x8])
    out = evaluate_tree(BEST_TREE, X)
    outputs_by_universe.append(out[:min(50, len(out))])

# Compare first 5
for i in range(1, min(5, len(outputs_by_universe))):
    diff = np.abs(outputs_by_universe[i] - outputs_by_universe[0]).mean()
    print(f"|output_{i} - output_0| = {diff:.6e}")

print()
print("If diff is same order as output magnitude → output IS universe-dependent")
print("If diff ~ 0 → output is NOT universe-dependent (formula is broken)")
