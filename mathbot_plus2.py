import sys
import numpy as np
from itertools import combinations_with_replacement


def load_points(path):
    data = np.genfromtxt(path, delimiter=",", comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def build_features(X, degree, use_unary=True):
    n, d = X.shape
    base_names = [f"X{i}" for i in range(d)]
    base_cols = [X[:, i] for i in range(d)]

    if use_unary:
        for i in range(d):
            base_cols.append(np.cos(X[:, i]))
            base_names.append(f"cos(X{i})")
            base_cols.append(np.sin(X[:, i]))
            base_names.append(f"sin(X{i})")

    base = np.column_stack(base_cols)
    nb = base.shape[1]

    features = [np.ones(n)]
    names = ["1"]
    for deg in range(1, degree + 1):
        for combo in combinations_with_replacement(range(nb), deg):
            col = np.ones(n)
            parts = []
            for idx in combo:
                col = col * base[:, idx]
                parts.append(base_names[idx])
            if not np.all(np.isfinite(col)):
                continue
            features.append(col)
            names.append("*".join(parts))

    return np.column_stack(features), names


def normalize_columns(A):
    norms = np.linalg.norm(A, axis=0)
    norms[norms == 0] = 1.0
    return A / norms, norms


def omp_bic(A, y, max_terms=10, min_rel_improve=0.05):
    n, m = A.shape
    residual = y.copy()
    y_norm = np.linalg.norm(y)
    selected = []
    coefficients = np.zeros(m)
    history = []
    prev_bic = np.inf
    prev_mse = np.inf

    for step in range(max_terms):
        corr = np.abs(A.T @ residual)
        for s in selected:
            corr[s] = -np.inf
        best = int(np.argmax(corr))
        if corr[best] < 1e-12 * y_norm:
            break

        selected.append(best)
        A_sel = A[:, selected]
        coef_sel, *_ = np.linalg.lstsq(A_sel, y, rcond=None)
        coefficients = np.zeros(m)
        coefficients[selected] = coef_sel
        residual = y - A_sel @ coef_sel
        mse = np.mean(residual**2) + 1e-30
        k = len(selected)
        bic = n * np.log(mse) + k * np.log(n)
        history.append((k, bic, mse))

        rel_improve = (prev_mse - mse) / prev_mse if prev_mse < np.inf else 1.0

        if step >= 1 and (bic > prev_bic or rel_improve < min_rel_improve):
            selected.pop()
            A_sel = A[:, selected]
            coef_sel, *_ = np.linalg.lstsq(A_sel, y, rcond=None)
            coefficients = np.zeros(m)
            coefficients[selected] = coef_sel
            residual = y - A_sel @ coef_sel
            break

        prev_bic = bic
        prev_mse = mse
        if mse < 1e-25:
            break

    return selected, coefficients, residual, history


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python mathbot_plus2.py <points.csv> [max_degree] [max_terms]")
        sys.exit(1)
    path = args[0]
    max_degree = int(args[1]) if len(args) > 1 else 4
    max_terms = int(args[2]) if len(args) > 2 else 6

    data = load_points(path)
    n, d = data.shape
    X = data[:, :-1]
    y = data[:, -1]

    print(f"[*] Points: {n}, columns: {d} ({d-1} inputs + 1 output)")
    print(f"[*] max_degree={max_degree}, max_terms={max_terms}")

    Phi, names = build_features(X, max_degree, use_unary=True)
    print(f"[*] Feature count: {Phi.shape[1]}")

    Phi_norm, norms = normalize_columns(Phi)
    selected, coef_norm, residual, history = omp_bic(
        Phi_norm, y, max_terms=max_terms)

    coef = coef_norm / norms

    print(f"\n[+] Selected {len(selected)} terms:")
    ordered = sorted(selected, key=lambda i: -abs(coef[i]))
    for s in ordered:
        c = coef[s]
        if abs(c) < 1e-10:
            continue
        print(f"    {c:+.8f}  *  {names[s]}")

    y_pred = Phi[:, selected] @ coef[selected]
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    rmse = np.sqrt(ss_res / n)

    print(f"\n[+] R^2   = {r2:.10f}")
    print(f"[+] RMSE = {rmse:.4e}")

    print("\n[+] BIC history (k, BIC, MSE):")
    for k, bic, mse in history:
        print(f"    k={k}  BIC={bic:.4f}  MSE={mse:.4e}")


if __name__ == "__main__":
    main()
