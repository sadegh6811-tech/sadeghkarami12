#!/usr/bin/env python3
"""Test 9D formula on parallel universes — using ORIGINAL file."""
import os, sys, importlib.util

# Load the original file as a module
orig_path = os.path.expanduser("~/mathbot/test_universes_formula.py")
spec = importlib.util.spec_from_file_location("orig", orig_path)
orig = importlib.util.module_from_spec(spec)

# We need to bypass the print() calls in the original file
# by stubbing out its main body. Simpler: just read the tree from source.

import numpy as np
from genetic_engine_classifier import evaluate_tree

# Extract BEST_TREE by reading the file
with open(orig_path) as f:
    content = f.read()

# Find the BEST_TREE assignment
import re
match = re.search(r'BEST_TREE\s*=\s*(\([^)]*(?:\([^)]*(?:\([^)]*\)[^)]*)*\)[^)]*)*\))', content, re.DOTALL)
if not match:
    print("ERROR: could not find BEST_TREE in source")
    sys.exit(1)

tree_str = match.group(1)
# Clean up the captured string
tree_str = tree_str.strip()
# Remove trailing garbage after the last balanced paren
depth = 0
for i, ch in enumerate(tree_str):
    if ch == '(': depth += 1
    elif ch == ')': depth -= 1
    if depth == 0:
        tree_str = tree_str[:i+1]
        break

print(f"Extracted BEST_TREE ({len(tree_str)} chars)")
BEST_TREE = eval(tree_str)
print(f"BEST_TREE parsed successfully")
print()

# Load universes
universes = np.load(os.path.expanduser("~/parallel_universes.npy"), allow_pickle=True)
print(f"Loaded {len(universes)} universes")

# Apply formula
all_outputs = []
for i, u in enumerate(universes):
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
    out = evaluate_tree(BEST_TREE, X)
    all_outputs.append(out)

all_outputs = np.concatenate(all_outputs)

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Total points: {len(all_outputs)}")
print(f"Output min:   {all_outputs.min():.4e}")
print(f"Output max:   {all_outputs.max():.4e}")
print(f"Output mean:  {all_outputs.mean():.4e}")
print(f"Output std:   {all_outputs.std():.4e}")
print(f"NaN count:    {np.isnan(all_outputs).sum()}")
print(f"Inf count:    {np.isinf(all_outputs).sum()}")
print()
up = (all_outputs > 0.1).sum()
down = (all_outputs < -0.1).sum()
neutral = len(all_outputs) - up - down
print(f"UP signals:      {up} ({100*up/len(all_outputs):.1f}%)")
print(f"DOWN signals:    {down} ({100*down/len(all_outputs):.1f}%)")
print(f"NEUTRAL signals: {neutral} ({100*neutral/len(all_outputs):.1f}%)")

print()
print("=" * 60)
print("Formula structure check")
print("=" * 60)
print(f"BEST_TREE root operation: {BEST_TREE[0]}")
print()
print("Note: x5 = ones (constant), x6 = zeros (constant 0)")
print("  → x6 in denominator causes division by 1e-9")
print("  → Output may be dominated by numerical noise")
