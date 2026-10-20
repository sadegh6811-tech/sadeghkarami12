"""MathBot Hybrid v3 - سنسور هندسی + رگرسیون + OMP."""
import sys
import numpy as np
import sympy as sp
from mathbot_core import polynomial_library, trigonometric_library, estimate_dimension_twonn
from extended_lib import (sqrt_library, log_library, inverse_library,
                           inverse_square_library, exp_library, abs_library)


# ═══════════════════════════════════════════════════════════
# ۱. سنسور تشخیصی و هندسی (Diagnostic & Geometric Sensor)
# ═══════════════════════════════════════════════════════════
def diagnostic_sensor(X, y):
    n, d = X.shape
    
    # ۱. نویز
    X_poly, _ = polynomial_library(X, degree=2)
    coef, _, _, _ = np.linalg.lstsq(X_poly, y, rcond=None)
    y_pred = X_poly @ coef
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_poly = 1 - ss_res / (ss_tot + 1e-12)
    noise_ratio = max(0.0, 1.0 - r2_poly)
    
    # ۲. همبستگی
    if d > 1:
        corr_matrix = np.corrcoef(X.T)
        max_corr = np.max(np.abs(corr_matrix - np.eye(d)))
    else:
        max_corr = 0.0
    
    # ۳. پیچیدگی
    n_poly = (d + 1) * (d + 2) // 2
    complexity = n_poly * 3
    
    # ۴. بعد ذاتی
    Xn = (X - X.mean(0)) / (X.std(0) + 1e-12)
    intrinsic_dim = estimate_dimension_twonn(Xn)
    
    # ۵. سنسور هندسی: تشخیص مکعب (Hypercube)
    #    ویژگی: نقاط در گوشه‌ها (مقادیر حدی) تجمع دارند
    X_abs = np.abs(X)
    max_vals = np.max(X_abs, axis=1)
    # نسبت نقاطی که در نزدیکی مرز (بیش از 0.9) هستند
    boundary_ratio = np.mean(max_vals > 0.9)
    
    # اگر نسبت نقاط مرزی بالا بود، احتمالا مکعب است
    is_cube = boundary_ratio > 0.3
    
    return {
        "noise_ratio": noise_ratio,
        "max_corr": max_corr,
        "complexity": complexity,
        "intrinsic_dim": intrinsic_dim,
        "n_samples": n,
        "n_features": d,
        "boundary_ratio": boundary_ratio,
        "is_cube": is_cube,
    }

# ═══════════════════════════════════════════════════════════
# ۲. هسته تصمیم‌گیری (Decision Core)
# ═══════════════════════════════════════════════════════════
def decision_core(sensor_data, degree=3):
    noise = sensor_data["noise_ratio"]
    is_cube = sensor_data["is_cube"]
    d = sensor_data["n_features"]
    
    # اگر داده مکعب است، از abs استفاده کن و sqrt را نادیده بگیر
    if is_cube:
        return "cube", None, 0.1
    
    # قواعد قبلی
    if noise < 0.1:
        return "regression", None, 0.1
    if noise > 0.1:
        return "omp", 5, 0.15
    if sensor_data["max_corr"] > 0.7:
        return "omp", 5, 0.1
    return "regression", None, 0.1

# ═══════════════════════════════════════════════════════════
# ۳. ساخت کتابخانه (Library Builder)
# ═══════════════════════════════════════════════════════════
def build_library(X, degree, use_trig=True, is_cube=False):
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
    specs = list(poly_combos)
    
    if Trig.shape[1] > 0:
        parts.append(Trig)
        specs.extend([("trig",) + tuple(s) for s in trig_specs])
        
    ext_specs = []
    try:
        Log, log_specs = log_library(Xn)
        Inv, inv_specs = inverse_library(Xn)
        Inv2, inv2_specs = inverse_square_library(Xn)
        Exp, exp_specs = exp_library(Xn)
        Abs, abs_specs = abs_library(Xn)
        
        # اگر مکعب است، sqrt را اضافه نکن
        if not is_cube:
            Sqrt, sqrt_specs = sqrt_library(Xn)
            parts.append(Sqrt)
            ext_specs.extend(sqrt_specs)
            
        for arr, s in [(Log, log_specs), (Inv, inv_specs), (Inv2, inv2_specs), (Exp, exp_specs), (Abs, abs_specs)]:
            if arr.shape[1] > 0:
                parts.append(arr)
                ext_specs.extend(s)
    except Exception as e:
        print(f"[i] extended_lib non caricata: {e}")
        
    Full = np.column_stack(parts)
    all_specs = specs + ext_specs
    return Full, all_specs, sigma

# ═══════════════════════════════════════════════════════════
# ۴. الگوریتم‌های رگرسیون و OMP
# ═══════════════════════════════════════════════════════════
def omp_regression(A, y, max_terms=5, tol=1e-3):
    norms = np.linalg.norm(A, axis=0)
    norms[norms < 1e-12] = 1.0
    A_norm = A / norms
    residual = y.copy()
    active = []
    coef = np.zeros(A.shape[1])
    
    for _ in range(min(max_terms, A.shape[1])):
        corr = A_norm.T @ residual
        corr[active] = -np.inf
        best_idx = np.argmax(np.abs(corr))
        if np.abs(corr[best_idx]) < tol:
            break
        active.append(best_idx)
        A_active = A_norm[:, active]
        coef_active, _, _, _ = np.linalg.lstsq(A_active, y, rcond=None)
        residual = y - A_active @ coef_active
        
    if active:
        coef[active] = coef_active / norms[active]
    return coef

# ═══════════════════════════════════════════════════════════
# ۵. تبدیل ضرایب به عبارت SymPy
# ═══════════════════════════════════════════════════════════
def expr_from_coef(coef, all_specs, sigma, tol_ratio):
    n_feats = len(sigma)
    syms = sp.symbols(f"x0:{n_feats}")
    expr = sp.Integer(0)
    
    for c, spec in zip(coef, all_specs):
        if abs(c) < tol_ratio: continue
        
        if isinstance(spec, tuple) and len(spec) >= 2 and isinstance(spec[0], str):
            kind = spec[0]
            if kind == "trig":
                kind_real, i, omega = spec[1], spec[2], spec[3] if len(spec) > 3 else 0.0
                arg = sp.Float(omega) * syms[i] / sp.Float(sigma[i])
                if kind_real == "sin": expr += sp.Float(c) * sp.sin(arg)
                else: expr += sp.Float(c) * sp.cos(arg)
                continue
            i = spec[1]
            xi = syms[i] / sp.Float(sigma[i])
            if kind == "exp": expr += sp.Float(c) * sp.exp(xi)
            elif kind == "expm": expr += sp.Float(c) * sp.exp(-xi)
            elif kind == "log": expr += sp.Float(c) * sp.log(xi)
            elif kind == "inv": expr += sp.Float(c) / xi
            elif kind == "inv2": expr += sp.Float(c) / (xi ** 2)
            elif kind == "sqrt": expr += sp.Float(c) * sp.sqrt(xi)
            elif kind == "abs": expr += sp.Float(c) * sp.Abs(xi)
        else:
            term = sp.Float(c)
            for i in spec: term = term * (syms[i] / sp.Float(sigma[i]))
            expr += term
            
    return expr

# ═══════════════════════════════════════════════════════════
# ۶. اجرای اصلی
# ═══════════════════════════════════════════════════════════
def main():
    args = sys.argv[1:]
    if not args:
        print("Usage: python mathbot_hybrid.py <file.csv> <target_col> [degree]")
        sys.exit(1)
        
    path = args[0]
    target_col = int(args[1]) if len(args) > 1 else -1
    degree = int(args[2]) if len(args) > 2 else 3
    
    X = np.loadtxt(path, delimiter=',', skiprows=1)
    if X.ndim == 1: X = X.reshape(-1, 1)
    if target_col < 0: target_col = X.shape[1] + target_col
    
    y = X[:, target_col]
    X_feat = np.delete(X, target_col, axis=1)
    
    print(f"[*] داده: {X.shape[0]} نقطه، {X.shape[1]} ستون")
    print(f"[*] ستون {target_col} به عنوان y")
    
    sensor = diagnostic_sensor(X_feat, y)
    print(f"\n[*] سنسور تشخیصی:")
    print(f"    Noise Ratio: {sensor['noise_ratio']:.3f}")
    print(f"    Boundary Ratio: {sensor['boundary_ratio']:.3f}")
    print(f"    Is Cube: {sensor['is_cube']}")
    
    method, max_terms, tol_ratio = decision_core(sensor, degree)
    print(f"\n[*] تصمیم: {method.upper()} (max_terms={max_terms}, tol={tol_ratio})")
    
    Full, all_specs, sigma = build_library(X_feat, degree, is_cube=(method=="cube"))
    print(f"[*] ویژگی‌ها: {Full.shape[1]}")
    
    if method == "omp":
        coef = omp_regression(Full, y, max_terms=max_terms, tol=tol_ratio)
    else:
        coef, _, _, _ = np.linalg.lstsq(Full, y, rcond=None)
        
    expr = expr_from_coef(coef, all_specs, sigma, tol_ratio)
    y_sym = sp.Symbol('y')
    eq_final = sp.Eq(y_sym, expr.evalf(4))
    print(f"\n[✓] معادله صریح:")
    print(f"    {eq_final}")

if __name__ == "__main__":
    main()
