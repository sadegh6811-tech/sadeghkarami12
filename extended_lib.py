"""کتابخانه کامل توابع ریاضی برای MathBot."""
import numpy as np


def exp_library(X):
    n, d = X.shape
    cols, specs = [], []
    for i in range(d):
        xi = X[:, i]
        std = xi.std()
        if std < 1e-10:
            xi_norm = np.zeros(n)
        else:
            xi_norm = (xi - xi.mean()) / std
        xi_clip = np.clip(xi_norm, -3, 3)
        cols.append(np.exp(xi_clip)); specs.append(("exp", i, 0.0))
        cols.append(np.exp(-xi_clip)); specs.append(("expm", i, 0.0))
    return np.column_stack(cols), specs


def log_library(X, offset=0.1):
    n, d = X.shape
    cols, specs = [], []
    for i in range(d):
        xi = np.abs(X[:, i]) + offset
        cols.append(np.log(xi)); specs.append(("log", i, 0.0))
    return np.column_stack(cols), specs


def inverse_library(X, offset=0.1):
    n, d = X.shape
    cols, specs = [], []
    for i in range(d):
        xi = np.where(np.abs(X[:, i]) < offset, offset, X[:, i])
        cols.append(1.0 / xi); specs.append(("inv", i, 0.0))
    return np.column_stack(cols), specs


def inverse_square_library(X, offset=0.1):
    """1/x² — برای کشف معادلات گرانش و کولن."""
    n, d = X.shape
    cols, specs = [], []
    for i in range(d):
        xi = np.where(np.abs(X[:, i]) < offset, offset, X[:, i])
        cols.append(1.0 / (xi ** 2))
        specs.append(("inv2", i, 0.0))
    return np.column_stack(cols), specs


def inverse_cube_library(X, offset=0.1):
    """1/x³ — برای معادلات پیشرفته."""
    n, d = X.shape
    cols, specs = [], []
    for i in range(d):
        xi = np.where(np.abs(X[:, i]) < offset, offset, X[:, i])
        cols.append(1.0 / (xi ** 3))
        specs.append(("inv3", i, 0.0))
    return np.column_stack(cols), specs


def sqrt_library(X):
    n, d = X.shape
    cols, specs = [], []
    for i in range(d):
        xi = np.sqrt(np.abs(X[:, i]) + 1e-10)
        cols.append(xi); specs.append(("sqrt", i, 0.0))
    return np.column_stack(cols), specs


def hyperbolic_library(X):
    n, d = X.shape
    cols, specs = [], []
    for i in range(d):
        xi = X[:, i]
        std = xi.std()
        if std < 1e-10:
            xi_norm = np.zeros(n)
        else:
            xi_norm = (xi - xi.mean()) / std
        xi_clip = np.clip(xi_norm, -3, 3)
        cols.append(np.tanh(xi_clip)); specs.append(("tanh", i, 0.0))
        cols.append(np.sinh(xi_clip)); specs.append(("sinh", i, 0.0))
        cols.append(np.cosh(xi_clip)); specs.append(("cosh", i, 0.0))
    return np.column_stack(cols), specs


def abs_library(X):
    """|x| — قدر مطلق، برای کشف مرزهای هایپرکیوب."""
    n, d = X.shape
    cols, specs = [], []
    for i in range(d):
        cols.append(np.abs(X[:, i]))
        specs.append(("abs", i, 0.0))
    return np.column_stack(cols), specs


def max_abs_library(X, pairs):
    """max(|xi|, |xj|) — برای کشف هایپرکیوب‌های چندبعدی."""
    n, d = X.shape
    cols, specs = [], []
    for i, j in pairs:
        val = np.maximum(np.abs(X[:, i]), np.abs(X[:, j]))
        cols.append(val)
        specs.append(("maxabs", i, j))
    return np.column_stack(cols), specs
