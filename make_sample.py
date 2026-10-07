"""ساخت داده‌های نمونه، شامل نسخه‌های نویزی."""
import os
import numpy as np


def add_noise(X, sigma_frac=0.04):
    """نویز گاوسی با انحراف معیار نسبی sigma_frac."""
    scale = X.std()
    noise = np.random.randn(*X.shape) * sigma_frac * scale
    return X + noise


def circle_2d(n=500):
    t = np.random.rand(n) * 2 * np.pi
    return np.column_stack([np.cos(t), np.sin(t)])


def sphere_3d(n=800):
    u = np.random.randn(n, 3)
    u -= u.mean(axis=0, keepdims=True)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    u -= u.mean(axis=0, keepdims=True)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u


def helix_3d(n=800):
    t = np.linspace(0, 4 * np.pi, n)
    return np.column_stack([np.cos(t), np.sin(t), t / (4 * np.pi)])


def sphere_4d(n=1500):
    u = np.random.randn(n, 4)
    u -= u.mean(axis=0, keepdims=True)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    u -= u.mean(axis=0, keepdims=True)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return u


def torus_4d(n=2000):
    u = np.random.rand(n) * 2 * np.pi
    v = np.random.rand(n) * 2 * np.pi
    return np.column_stack([np.cos(u), np.sin(u), np.cos(v), np.sin(v)])


if __name__ == "__main__":
    os.makedirs("samples", exist_ok=True)
    np.random.seed(42)

    clean = {
        "circle_2d":  circle_2d(),
        "sphere_3d":  sphere_3d(),
        "helix_3d":   helix_3d(),
        "sphere_4d":  sphere_4d(),
        "torus_4d":   torus_4d(),
    }
    for name, X in clean.items():
        np.savetxt(f"samples/{name}.csv", X, delimiter=",")
        np.savetxt(f"samples/{name}_noisy.csv", add_noise(X, 0.04), delimiter=",")

    print("[✓] نمونه‌ها ساخته شد: پاک + نویزی (سیگما نسبی 4%)")
