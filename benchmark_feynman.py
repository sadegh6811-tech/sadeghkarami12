"""
بنچمارک روی معادلات کلاسیک فیزیک.
هر معادله یک منیفولد تولید می‌کند.
"""
import numpy as np
import time
from mathbot_core import estimate_dimension_twonn, estimate_dimension_mle
from discover import discover_equations, convert_to_original


def make_manifold_1_sphere_3d(n=800):
    """کره واحد"""
    u = np.random.randn(n, 3)
    u -= u.mean(0)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    u -= u.mean(0)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u, 2, 2, "x0^2 + x1^2 + x2^2 = 1"


def make_manifold_2_ellipsoid(n=800):
    """بیضی‌گون با نیم‌محورهای 1، 2، 3"""
    u = np.random.randn(n, 3)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    X = u * np.array([1.0, 2.0, 3.0])
    return X, 2, 2, "x0^2/1 + x1^2/4 + x2^2/9 = 1"


def make_manifold_3_cylinder(n=800):
    """استوانه واحد"""
    t = np.random.rand(n) * 4 * np.pi
    z = np.random.randn(n) * 0.5
    return np.column_stack([np.cos(t), np.sin(t), z]), 2, 2, "x0^2 + x1^2 = 1"


def make_manifold_4_paraboloid(n=800):
    """سهمی‌گون"""
    u = np.random.randn(n) * 1.5
    v = np.random.randn(n) * 1.5
    z = u * u + v * v
    return np.column_stack([u, v, z]), 2, 2, "x2 = x0^2 + x1^2"


def make_manifold_5_sphere_4d(n=1500):
    """کره ۴بعدی"""
    u = np.random.randn(n, 4)
    u -= u.mean(0)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    u -= u.mean(0)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u, 4, 3, "x0^2 + x1^2 + x2^2 + x3^2 = 1"


def make_manifold_6_hyperbolic(n=800):
    """هذلول یک‌ورقی"""
    u = np.random.randn(n) * 1.5
    v = np.random.randn(n) * 1.5
    x = np.sqrt(1 + u*u + v*v)
    return np.column_stack([x, u, v]), 2, 2, "x0^2 - x1^2 - x2^2 = 1"


def make_manifold_7_cone(n=800):
    """مخروط"""
    t = np.random.rand(n) * 2 * np.pi
    r = np.random.rand(n) * 2
    return np.column_stack([r*np.cos(t), r*np.sin(t), r]), 2, 2, "x0^2 + x1^2 = x2^2"


def make_manifold_8_helix(n=800):
    """هلیکس"""
    t = np.linspace(0, 4*np.pi, n)
    return np.column_stack([np.cos(t), np.sin(t), t/(4*np.pi)]), 3, 1, "sin/cos"


def make_manifold_9_torus_4d(n=2000):
    """تور ۴بعدی"""
    u = np.random.rand(n) * 2 * np.pi
    v = np.random.rand(n) * 2 * np.pi
    return np.column_stack([np.cos(u), np.sin(u), np.cos(v), np.sin(v)]), 4, 2, "دو دایره"


def make_manifold_10_circle_2d(n=500):
    """دایره ۲بعدی"""
    t = np.random.rand(n) * 2 * np.pi
    return np.column_stack([np.cos(t), np.sin(t)]), 2, 1, "x0^2 + x1^2 = 1"


BENCHMARKS = [
    ("Sphere 3D",        make_manifold_1_sphere_3d,       2),
    ("Ellipsoid",        make_manifold_2_ellipsoid,       2),
    ("Cylinder",         make_manifold_3_cylinder,        2),
    ("Paraboloid",       make_manifold_4_paraboloid,      2),
    ("Sphere 4D",        make_manifold_5_sphere_4d,       2),
    ("Hyperbolic",       make_manifold_6_hyperbolic,      2),
    ("Cone",             make_manifold_7_cone,            2),
    ("Helix",            make_manifold_8_helix,           3),
    ("Torus 4D",         make_manifold_9_torus_4d,        2),
    ("Circle 2D",        make_manifold_10_circle_2d,      2),
]


def run_benchmark():
    print("=" * 70)
    print("بنچمارک MathBot — ۱۰ منیفولد مختلف")
    print("=" * 70)

    results = []
    for name, maker, max_deg in BENCHMARKS:
        X, n, d_true, description = maker()
        t0 = time.time()
        try:
            Xn = (X - X.mean(0)) / X.std(0)
            d_twonn = estimate_dimension_twonn(Xn)
            d_mle = estimate_dimension_mle(Xn, k=15)
            d_est = int(round((d_twonn + d_mle) / 2)) if abs(d_twonn - d_mle) < 1.5 else min(int(d_twonn), int(d_mle))
            d_est = max(1, min(d_est, n))

            n_eq = n - d_est
            if n_eq <= 0:
                n_eq = 1  # fallback: dim over-estimated

            eqs, S, mu, sigma = discover_equations(X, n_eq, degree=max_deg)
            elapsed = time.time() - t0

            ok_d = (d_est == d_true)
            ok_eq = len(eqs) == n_eq
            status = "✓" if (ok_d and ok_eq) else "✗"

            results.append({
                "name": name, "n": n, "d_true": d_true, "d_est": d_est,
                "n_eq_true": n - d_true, "n_eq_got": len(eqs),
                "time": elapsed, "status": status,
            })
            print(f"{status} {name:15s}  n={n}  d={d_est} (true {d_true})  eq={len(eqs)}/{n_eq}  {elapsed:.2f}s")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"✗ {name:15s}  خطا: {type(e).__name__}: {e}")
            results.append({"name": name, "status": "✗", "error": str(e)})

    print("=" * 70)
    ok = sum(1 for r in results if r.get("status") == "✓")
    total = len(results)
    print(f"نتیجه نهایی: {ok}/{total} موفق")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
