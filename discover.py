"""کشف معادلات ضمنی با SVD — نسخه نهایی و تمیز."""
import numpy as np
import sympy as sp
from mathbot_core import polynomial_library, trigonometric_library


def _clean_coef(coef, tol_ratio):
    abs_coef = np.abs(coef)
    max_c = np.max(abs_coef)
    if max_c < 1e-12:
        return coef
    coef = coef / max_c
    abs_coef = np.abs(coef)
    
    effective_tol = max(tol_ratio, 0.01)
    mask = abs_coef > effective_tol
    n_kept = np.sum(mask)
    
    if n_kept < 2:
        sorted_idx = np.argsort(-abs_coef)
        n_keep = min(3, len(coef))
        mask = np.zeros_like(coef, dtype=bool)
        mask[sorted_idx[:n_keep]] = True
        for idx in sorted_idx[:n_keep]:
            if abs_coef[idx] < 0.001:
                mask[idx] = False
    
    coef[~mask] = 0.0
    
    for i in range(len(coef)):
        if abs(coef[i]) < 1e-10:
            continue
        rounded = round(coef[i])
        if abs(coef[i] - rounded) < 0.15 and abs(rounded) > 0.15:
            coef[i] = float(rounded)
        else:
            coef[i] = round(coef[i], 2)
    
    return coef


def _sparsify_vectors(vectors, n_iter=300, seed=42):
    vectors = np.array(vectors, dtype=float)
    n_eq = vectors.shape[0]
    if n_eq <= 1:
        return vectors.copy()
    rng = np.random.default_rng(seed)
    best = vectors.copy()
    best_score = np.sum(np.abs(best))
    for _ in range(n_iter):
        A = rng.normal(size=(n_eq, n_eq))
        Q, _ = np.linalg.qr(A)
        candidate = Q @ vectors
        score = np.sum(np.abs(candidate))
        if score < best_score:
            best = candidate
            best_score = score
    return best


def _extract_null_vectors(Theta, n_equations, tol_ratio):
    U, S, Vt = np.linalg.svd(Theta, full_matrices=False)
    vectors = np.array(Vt[-n_equations:])
    if n_equations > 1:
        vectors = _sparsify_vectors(vectors)
    results = [_clean_coef(v.copy(), tol_ratio) for v in vectors]
    return results, S


def _build_library(Xn, degree, freqs, use_extended=True, topology="smooth"):
    Poly, poly_combos = polynomial_library(Xn, degree)
    n_poly = Poly.shape[1]
    Trig, trig_specs = trigonometric_library(Xn, freqs)
    
    parts = [Poly]
    all_specs = list(poly_combos)
    
    # اگر توپولوژی مکعب است، فقط از abs استفاده کن
    if topology == "cube":
        try:
            from extended_lib import abs_library
            Abs, abs_specs = abs_library(Xn)
            if Abs.shape[1] > 0:
                parts = [Poly, Abs]
                all_specs.extend(abs_specs)
        except Exception as e:
            print(f"[i] abs_library non caricata: {e}")
        Full = np.column_stack(parts)
        return Full, all_specs, trig_specs, n_poly
    
    # حالت عادی (smooth)
    if Trig.shape[1] > 0:
        parts.append(Trig)
        all_specs.extend([("trig",) + tuple(s) if isinstance(s, tuple) else ("trig", s)
                          for s in trig_specs])
    
    if use_extended:
        try:
            from extended_lib import (exp_library, log_library,
                                       inverse_library, inverse_square_library,
                                       sqrt_library)
            has_trig = Trig.shape[1] > 0
            
            if not has_trig:
                Exp, exp_specs = exp_library(Xn)
                if Exp.shape[1] > 0:
                    parts.append(Exp)
                    all_specs.extend(exp_specs)
            
            Log, log_specs = log_library(Xn)
            Inv, inv_specs = inverse_library(Xn)
            Inv2, inv2_specs = inverse_square_library(Xn)
            Sqrt, sqrt_specs = sqrt_library(Xn)
            
            for arr, specs in [(Log, log_specs), (Inv, inv_specs), 
                                (Inv2, inv2_specs), (Sqrt, sqrt_specs)]:
                if arr.shape[1] > 0:
                    parts.append(arr)
                    all_specs.extend(specs)
        except Exception as e:
            print(f"[i] extended_lib non caricata: {e}")
    
    Full = np.column_stack(parts) if len(parts) > 1 else parts[0]
    return Full, all_specs, trig_specs, n_poly


def discover_equations(X, n_equations, degree=3, tol_ratio=0.12, use_trig=True, topology="smooth"):
    mu = np.zeros(X.shape[1])
    sigma = X.std(axis=0) + 1e-12
    Xn = X / sigma

    freqs = {}
    if use_trig:
        print("[*] تشخیص فرکانس...")
        try:
            from freq_lib import detect_all_frequencies
            freqs, circles = detect_all_frequencies(Xn, top_n=2)
            if circles:
                print("    جفت‌های دایره‌ای:")
                for (i, j), r2 in circles.items():
                    print(f"      x{i}² + x{j}² ≈ {r2:.3f}")
            found = False
            for j, fs in freqs.items():
                if fs:
                    found = True
                    print(f"    x{j}: [{', '.join(f'{f:.3f}' for f in fs)}]")
            if not found:
                print("    فرکانس مثلثاتی پیدا نشد.")
        except Exception as e:
            print(f"    خطا در تشخیص فرکانس: {e}")
            freqs = {}

    best = None
    for d_try in range(1, degree + 1):
        Theta, all_specs, trig_specs, n_poly = _build_library(Xn, d_try, freqs, topology=topology)
        if Theta.shape[1] <= n_equations:
            continue
        coefs_list, S = _extract_null_vectors(Theta, n_equations, tol_ratio)
        gap = np.log10(S[-n_equations - 1] / (S[-n_equations] + 1e-30) + 1e-30) if len(S) > n_equations else 1.0
        if best is None or gap > best[0] * 1.3:
            best = (gap, coefs_list, S, all_specs, trig_specs, n_poly, d_try)

    if best is None:
        print("[!] کتابخانه مناسب ساخته نشد.")
        return [], np.array([]), mu, sigma

    gap, coefs_list, S, all_specs, trig_specs, n_poly, best_degree = best
    print(f"[i] درجه انتخاب‌شده: {best_degree}")
    print(f"[i] شکاف طیفی: {gap:.2f}")
    print(f"[i] ویژگی‌ها: {n_poly} چندجمله‌ای + {len(trig_specs)} مثلثاتی + {len(all_specs)-n_poly} توسعه‌یافته")

    n_feats = X.shape[1]
    syms = sp.symbols(f"x0:{n_feats}")
    equations = []
    for coef in coefs_list:
        expr = sp.Integer(0)
        for c, spec in zip(coef, all_specs):
            if abs(c) < 1e-10:
                continue
            if isinstance(spec, tuple) and len(spec) >= 2 and isinstance(spec[0], str):
                kind = spec[0]
                if kind == "trig":
                    kind_real = spec[1]
                    i = spec[2]
                    omega = spec[3] if len(spec) > 3 else 0.0
                    xi = syms[i]
                    if kind_real == "sin":
                        expr += sp.Float(c) * sp.sin(sp.Float(omega) * xi)
                    elif kind_real == "cos":
                        expr += sp.Float(c) * sp.cos(sp.Float(omega) * xi)
                    continue
                i = spec[1]
                xi = syms[i]
                if kind == "exp":
                    expr += sp.Float(c) * sp.exp(xi)
                elif kind == "expm":
                    expr += sp.Float(c) * sp.exp(-xi)
                elif kind == "log":
                    expr += sp.Float(c) * sp.log(xi)
                elif kind == "inv":
                    expr += sp.Float(c) / xi
                elif kind == "inv2":
                    expr += sp.Float(c) / (xi ** 2)
                elif kind == "sqrt":
                    expr += sp.Float(c) * sp.sqrt(xi)
                elif kind == "abs":
                    expr += sp.Float(c) * sp.Abs(xi)
                elif kind == "abs":
                    expr += sp.Float(c) * sp.Abs(xi)
                elif kind == "sin":
                    omega = spec[2] if len(spec) > 2 else 0.0
                    expr += sp.Float(c) * sp.sin(sp.Float(omega) * xi)
                elif kind == "cos":
                    omega = spec[2] if len(spec) > 2 else 0.0
                    expr += sp.Float(c) * sp.cos(sp.Float(omega) * xi)
            else:
                combo = spec
                term = sp.Integer(1)
                for i in combo:
                    term = term * syms[i]
                expr += sp.Float(c) * term
        equations.append(expr)
    return equations, S, mu, sigma


def convert_to_original(eq, mu, sigma, n_feats):
    xs = sp.symbols(f"x0:{n_feats}")
    Xs = sp.symbols(f"X0:{n_feats}")
    if not hasattr(sigma, '__len__'): sigma = [float(sigma)] * n_feats
    if not hasattr(mu, '__len__'): mu = [float(mu)] * n_feats
    subs = {xs[i]: (Xs[i] - sp.Float(mu[i])) / sp.Float(sigma[i]) for i in range(n_feats)}
    out = eq.subs(subs)
    try:
        out = sp.expand(out)
    except Exception:
        pass
    try:
        poly = sp.Poly(out, *Xs)
        max_deg2 = 0.0
        for monom, coeff in poly.terms():
            if sum(monom) == 2:
                c = abs(float(coeff))
                if c > max_deg2: max_deg2 = c
        if max_deg2 > 1e-12:
            out = out / max_deg2
        else:
            coeffs = poly.coeffs()
            if coeffs:
                max_c = max(abs(float(c)) for c in coeffs)
                if max_c > 1e-12: out = out / max_c
    except Exception:
        pass
    return out.evalf(4)
