#!/usr/bin/env python3
"""Test 9D formula on parallel universes — clean import version."""
import os, sys
import numpy as np

# ==== Extract BEST_TREE from source with proper paren matching ====

src = os.path.expanduser("~/mathbot/test_universes_formula.py")
with open(src) as f:
    code = f.read()

# Find "BEST_TREE = (" and match all parens correctly
start = code.find("BEST_TREE")
if start == -1:
    print("ERROR: BEST_TREE not found")
    sys.exit(1)

# Find first '(' after "BEST_TREE"
open_paren = code.find("(", start)
if open_paren == -1:
    print("ERROR: no '(' after BEST_TREE")
    sys.exit(1)

# Match parentheses manually
depth = 0
close_paren = -1
for i in range(open_paren, len(code)):
    if code[i] == '(':
        depth += 1
    elif code[i] == ')':
        depth -= 1
        if depth == 0:
            close_paren = i
            break

if close_paren == -1:
    print("ERROR: unbalanced parentheses")
    sys.exit(1)

tree_str = code[open_paren:close_paren+1]
print(f"Extracted {len(tree_str)} chars")
print(f"Starts with: {tree_str[:50]}")
print(f"Ends with:   {tree_str[-50:]}")
print()

# Parse it safely
BEST_TREE = eval(tree_str)
print(f"BEST_TREE root: {BEST_TREE[0]}")
print()

# ==== Evaluate function ====
def evaluate_tree(tree, X):
    if isinstance(tree, str):
        if tree.startswith('x'):
            try:
                idx = int(tree[1:])
                return X[:, idx].astype(float)
            except:
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

# ==== Load data ====
universes = np.load(os.path.expanduser("~/parallel_universes.npy"), allow_pickle=True)
print(f"Loaded {len(universes)} universes")
print(f"First universe shape: {universes[0].shape}")
print()

# ==== Apply formula ====
all_outputs = []
for u in universes:
    phi = u[:, 0]; phi_dot = u[:, 1]; phi_ddot = u[:, 2]
    n = len(phi)
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
    all_outputs.append(evaluate_tree(BEST_TREE, X))

all_out = np.concatenate(all_outputs)

print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Total points: {len(all_out)}")
print(f"Output min:   {all_out.min():.4e}")
print(f"Output max:   {all_out.max():.4e}")
print(f"Output mean:  {all_out.mean():.4e}")
print(f"Output std:   {all_out.std():.4e}")
print(f"NaN count:    {np.isnan(all_out).sum()}")
print(f"Inf count:    {np.isinf(all_out).sum()}")
print()

up = (all_out > 0.1).sum()
down = (all_out < -0.1).sum()
neutral = len(all_out) - up - down
print(f"UP signals:      {up} ({100*up/len(all_out):.1f}%)")
print(f"DOWN signals:    {down} ({100*down/len(all_out):.1f}%)")
print(f"NEUTRAL signals: {neutral} ({100*neutral/len(all_out):.1f}%)")
print()

# ==== Check formula structure ====
print("=" * 60)
print("Formula structure check")
print("=" * 60)
print(f"Root: {BEST_TREE[0]}")
print()
print("Constant features in formula:")
print("  x5 = ones  (constant 1)")
print("  x6 = zeros (constant 0)")
print()
print("If x6 (zeros) is used in denominator:")
print("  → division by ~0 (only 1e-9 epsilon)")
print("  → output dominated by numerical noise")

# ==== Variability check ====
print()
print("=" * 60)
print("Do outputs differ across universes?")
print("=" * 60)
outputs_per_univ = []
for u in universes[:5]:
    phi = u[:, 0]; phi_dot = u[:, 1]; phi_ddot = u[:, 2]
    n = len(phi)
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
    outputs_per_univ.append(evaluate_tree(BEST_TREE, X))

for i in range(1, len(outputs_per_univ)):
    if len(outputs_per_univ[i]) == len(outputs_per_univ[0]):
        diff = np.abs(outputs_per_univ[i] - outputs_per_univ[0]).mean()
        print(f"|output_{i} - output_0| = {diff:.6e}")
