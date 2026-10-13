"""تست‌های واحد برای MathBot."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mathbot_core import estimate_dimension_twonn, estimate_dimension_mle
from mathbot_core import polynomial_library, trigonometric_library
from discover import discover_equations, convert_to_original
from freq_lib import detect_circle_pairs


def test_twonn_circle():
    """دایره ۲بعدی باید d=1 بدهد."""
    t = np.random.rand(500) * 2 * np.pi
    X = np.column_stack([np.cos(t), np.sin(t)])
    Xn = (X - X.mean(0)) / X.std(0)
    d = estimate_dimension_twonn(Xn)
    assert 0.7 < d < 1.4, f"d={d} برای دایره باید 1 باشد"


def test_twonn_sphere():
    """کره ۳بعدی باید d=2 بدهد."""
    u = np.random.randn(800, 3)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    Xn = (u - u.mean(0)) / u.std(0)
    d = estimate_dimension_twonn(Xn)
    assert 1.7 < d < 2.4, f"d={d} برای کره باید 2 باشد"


def test_polynomial_library_size():
    """کتابخانه درجه ۲ در ۳ متغیر باید ۱۰ ستون داشته باشد."""
    X = np.random.randn(50, 3)
    Theta, combos = polynomial_library(X, 2)
    assert Theta.shape[1] == 10


def test_circle_pair_detection():
    """روی داده‌ی دایره، جفت (0,1) باید پیدا شود."""
    t = np.random.rand(500) * 2 * np.pi
    X = np.column_stack([np.cos(t), np.sin(t)])
    Xn = (X - X.mean(0)) / X.std(0)
    pairs = detect_circle_pairs(Xn)
    assert (0, 1) in pairs or (1, 0) in pairs, "جفت دایره‌ای پیدا نشد"


def test_discover_sphere():
    """کشف معادله‌ی کره باید درست کار کند."""
    u = np.random.randn(500, 3)
    u -= u.mean(0)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    u -= u.mean(0)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    eqs, S, mu, sigma = discover_equations(u, 1, degree=2)
    assert len(eqs) == 1, "باید یک معادله پیدا شود"
    eq_str = str(eqs[0])
    assert "x0" in eq_str and "x1" in eq_str and "x2" in eq_str


if __name__ == "__main__":
    tests = [
        test_twonn_circle,
        test_twonn_sphere,
        test_polynomial_library_size,
        test_circle_pair_detection,
        test_discover_sphere,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"✓ {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\nنتیجه: {passed} موفق، {failed} ناموفق")
    sys.exit(0 if failed == 0 else 1)
