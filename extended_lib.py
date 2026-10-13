"""کتابخانه‌ی توابع نمایی، لگاریتمی، کسری."""
import numpy as np


def exp_library(X):
    n, d = X.shape
    cols, specs = [], []
    for i in range(d):
        xi = np.clip(X[:, i], -5, 5)
        cols.append(np.exp(xi)); specs.append(("exp", i, 0.0))
        cols.append(np.exp(-xi)); specs.append(("expm", i, 0.0))
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


def hyperbolic_library(X):
    n, d = X.shape
    cols, specs = [], []
    for i in range(d):
        xi = np.clip(X[:, i], -10, 10)
        cols.append(np.tanh(xi)); specs.append(("tanh", i, 0.0))
        xi2 = np.clip(X[:, i], -3, 3)
        cols.append(np.sinh(xi2)); specs.append(("sinh", i, 0.0))
        cols.append(np.cosh(xi2)); specs.append(("cosh", i, 0.0))
    return np.column_stack(cols), specs
