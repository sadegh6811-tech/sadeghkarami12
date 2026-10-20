import numpy as np
from discover import discover_equations

np.random.seed(42)
N = 500
v = np.random.randn(N, 3)
v = v / np.linalg.norm(v, axis=1, keepdims=True)

print("=" * 60)
print("تست ۱: پارامترهای پیش‌فرض (tol_ratio=0.08)")
print("=" * 60)
eqs, S, mu, sigma = discover_equations(v, n_equations=1, degree=2)
print(f"معادله: {eqs[0]}")
print(f"S[:5]: {S[:5]}")

print("\n" + "=" * 60)
print("تست ۲: با tol_ratio=0.001")
print("=" * 60)
eqs2, S2, _, _ = discover_equations(v, n_equations=1, degree=2, tol_ratio=0.001)
print(f"معادله: {eqs2[0]}")
print(f"S[:5]: {S2[:5]}")

print("\n" + "=" * 60)
print("تست ۳: درجه ۳ با tol_ratio=0.01")
print("=" * 60)
eqs3, S3, _, _ = discover_equations(v, n_equations=1, degree=3, tol_ratio=0.01)
print(f"معادله: {eqs3[0]}")
print(f"S[:5]: {S3[:5]}")
