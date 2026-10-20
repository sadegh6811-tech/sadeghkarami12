import numpy as np
import sys
import os

# Add current path
sys.path.insert(0, os.path.expanduser("~/mathbot"))

# ---- Import BEST_TREE and evaluate_tree from original file ----
# We import from the original file to avoid syntax errors
try:
    from test_universes_formula import BEST_TREE, evaluate_tree
    print("✓ Imported BEST_TREE and evaluate_tree from test_universes_formula.py")
except ImportError as e:
    print(f"⚠ Could not import: {e}")
    print("Using inline definition instead")
    BEST_TREE = ('div',
        ('add', ('add', 'x7', 'x2'), ('const', 0.0)),
        ('add',
            ('add', ('abs', ('mul', ('sub', ('add', 'x6', 'x1'), ('sub', 'x1', 'x0')), ('div', ('div', 'x1', 'x6'), ('abs', 'x0'))))),
                ('div', ('const', -1.0), 'x5')
            ),
            ('abs', ('sub', ('abs', 'x2'), ('div', ('const', 0.5), 'x7')))
        )
    )
    def evaluate_tree(tree, X):
        if isinstance(tree, str):
            if tree.startswith('x'):
                try:
                    idx = int(tree[1:])
                    return X[:, idx].astype(float)
                except:
                    return np.zeros(X.shape[0])
            return np.zeros(X.shape[0])
        if tree[0] == 'const': return np.full(X.shape[0], float(tree[1]))
        if tree[0] == 'add': return evaluate_tree(tree[1], X) + evaluate_tree(tree[2], X)
        if tree[0] == 'sub': return evaluate_tree(tree[1], X) - evaluate_tree(tree[2], X)
        if tree[0] == 'mul': return evaluate_tree(tree[1], X) * evaluate_tree(tree[2], X)
        if tree[0] == 'div':
            a = evaluate_tree(tree[1], X)
            b = evaluate_tree(tree[2], X)
            return a / (np.abs(b) + 1e-9)
        if tree[0] == 'abs': return np.abs(evaluate_tree(tree[1], X))
        return np.zeros(X.shape[0])

# ---- Load data ----
universes = np.load(os.path.expanduser("~/parallel_universes.npy"), allow_pickle=True)
print(f"Loaded {len(universes)} universes\n")

# ---- Apply ----
all_outputs = []
for i, u in enumerate(universes):
    phi = u[:, 0]; phi_dot = u[:, 1]; phi_ddot = u[:, 2]
    n = len(phi)

    # Build 9D features
    X = np.column_stack([
        (phi - np.mean(phi)) / (np.std(phi) + 1e-9),
        np.sign(np.gradient(phi)),
        np.abs(phi_dot) / (np.std(phi_dot) + 1e-9),
        np.abs(phi_ddot) / (np.std(phi_ddot) + 1e-9),
        np.abs(phi * phi_dot) / (np.std(phi * phi_dot) + 1e-9),
        np.ones_like(phi),
        np.zeros_like(phi),
        np.gradient(phi) / (np.std(np.gradient(phi)) + 1e-9),
        np.linspace(0, 2*np.pi, n)
    ])

    out = evaluate_tree(BEST_TREE, X)
    all_outputs.append(out)

all_outputs = np.concatenate(all_outputs)

print("=" * 50)
print("RESULTS")
print("=" * 50)
print(f"Total points: {len(all_outputs)}")
print(f"Output min:   {all_outputs.min():.4e}")
print(f"Output max:   {all_outputs.max():.4e}")
print(f"Output mean:  {all_outputs.mean():.4e}")
print(f"Output std:   {all_outputs.std():.4e}")
print(f"NaN:          {np.isnan(all_outputs).sum()}")
print(f"Inf:          {np.isinf(all_outputs).sum()}")
print()

up = (all_outputs > 0.1).sum()
down = (all_outputs < -0.1).sum()
neutral = len(all_outputs) - up - down
print(f"UP signals:      {up} ({100*up/len(all_outputs):.1f}%)")
print(f"DOWN signals:    {down} ({100*down/len(all_outputs):.1f}%)")
print(f"NEUTRAL signals: {neutral} ({100*neutral/len(all_outputs):.1f}%)")
print()

# Check variability
print("Are universes different?")
for i in range(1, min(5, len(universes))):
    u0 = universes[0]; ui = universes[i]
    if u0.shape == ui.shape:
        diff = np.abs(u0 - ui).mean()
        print(f"  |universe[0] - universe[{i}]| = {diff:.6f}")
