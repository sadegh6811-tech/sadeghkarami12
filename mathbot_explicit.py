"""MathBot - کشف معادله صریح y = f(x) — نسخه نهایی با کتابخانه کامل."""
import sys
import numpy as np
import sympy as sp

from mathbot_core import polynomial_library, trigonometric_library


def discover_explicit(X, y, degree=3, tol_ratio=0.1, use_trig=True):
    """کشف معادله صریح y = f(x) با رگرسیون خطی."""
    sigma = X.std(axis=0) + 1e-12
    Xn = X / sigma
    
    Poly, poly_combos = polynomial_library(Xn, degree)
    n_poly = Poly.shape[1]
    
    freqs = {}
    if use_trig:
        try:
            from freq_lib import detect_all_frequencies
            freqs, _ = detect_all_frequencies(Xn, top_n=2)
        except Exception:
            pass
    
    Trig, trig_specs = trigonometric_library(Xn, freqs)
    
    parts = [Poly]
    if Trig.shape[1] > 0:
        parts.append(Trig)
    
    ext_specs = []
    try:
        from extended_lib import (sqrt_library, log_library, inverse_library,
                                   inverse_square_library, inverse_cube_library,
                                   exp_library, hyperbolic_library)
        Sqrt, sqrt_specs = sqrt_library(Xn)
        Log, log_specs = log_library(Xn)
        Inv, inv_specs = inverse_library(Xn)
        Inv2, inv2_specs = inverse_square_library(Xn)
        Inv3, inv3_specs = inverse_cube_library(Xn)
        Exp, exp_specs = exp_library(Xn)
        Hyp, hyp_specs = hyperbolic_library(Xn)
        
        for arr, specs in [(Sqrt, sqrt_specs), (Log, log_specs), 
                            (Inv, inv_specs), (Inv2, inv2_specs), 
                            (Inv3, inv3_specs), (Exp, exp_specs), 
                            (Hyp, hyp_specs)]:
            if arr.shape[1] > 0:
                parts.append(arr)
                ext_specs.extend(specs)
    except Exception as e:
        print(f"[i] extended_lib non caricata: {e}")
    
    Full = np.column_stack(parts)
    coef, _, _, _ = np.linalg.lstsq(Full, y, rcond=None)
    
    n_feats = X.shape[1]
    syms = sp.symbols(f"x0:{n_feats}")
    expr = sp.Integer(0)
    
    # poly
    for c, combo in zip(coef[:n_poly], poly_combos):
        if abs(c) < tol_ratio:
            continue
        term = sp.Float(c)
        for i in combo:
            term = term * (syms[i] / sp.Float(sigma[i]))
        expr += term
    
    # trig
    for c, spec in zip(coef[n_poly:n_poly+len(trig_specs)], trig_specs):
        if abs(c) < tol_ratio:
            continue
        kind, i, omega = spec
        arg = sp.Float(omega) * syms[i] / sp.Float(sigma[i])
        if kind == "sin":
            expr += sp.Float(c) * sp.sin(arg)
        else:
            expr += sp.Float(c) * sp.cos(arg)
    
    # extended
    start = n_poly + len(trig_specs)
    for c, spec in zip(coef[start:start+len(ext_specs)], ext_specs):
        if abs(c) < tol_ratio:
            continue
        kind, i = spec[0], spec[1]
        xi = syms[i] / sp.Float(sigma[i])
        if kind == "sqrt":
            expr += sp.Float(c) * sp.sqrt(xi)
        elif kind == "log":
            expr += sp.Float(c) * sp.log(xi)
        elif kind == "inv":
            expr += sp.Float(c) / xi
        elif kind == "inv2":
            expr += sp.Float(c) / (xi ** 2)
        elif kind == "inv3":
            expr += sp.Float(c) / (xi ** 3)
        elif kind == "exp":
            expr += sp.Float(c) * sp.exp(xi)
        elif kind == "expm":
            expr += sp.Float(c) * sp.exp(-xi)
        elif kind == "tanh":
            expr += sp.Float(c) * sp.tanh(xi)
        elif kind == "sinh":
            expr += sp.Float(c) * sp.sinh(xi)
        elif kind == "cosh":
            expr += sp.Float(c) * sp.cosh(xi)
    
    return expr, coef


def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python mathbot_explicit.py <file.csv> <target_col> [degree]")
        sys.exit(1)
    
    path = args[0]
    target_col = int(args[1]) if len(args) > 1 else -1
    degree = int(args[2]) if len(args) > 2 else 3
    
    X = np.loadtxt(path, delimiter=',', skiprows=1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    if target_col < 0:
        target_col = X.shape[1] + target_col
    
    y = X[:, target_col]
    X_feat = np.delete(X, target_col, axis=1)
    
    print(f"[*] داده: {X.shape[0]} نقطه، {X.shape[1]} ستون")
    print(f"[*] ستون {target_col} به عنوان y")
    
    eq, coef = discover_explicit(X_feat, y, degree=degree)
    
    y_sym = sp.Symbol('y')
    eq_final = sp.Eq(y_sym, eq.evalf(4))
    print(f"\n[✓] معادله صریح:")
    print(f"    {eq_final}")


if __name__ == "__main__":
    main()
