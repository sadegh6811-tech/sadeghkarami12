"""کشف معادلات ضمنی با SVD. نسخه‌ی پایدار و بدون باگ."""
import numpy as np
import sympy as sp
from mathbot_core import polynomial_library, trigonometric_library


def _clean_coef(coef, tol_ratio):
    """نرمالیزه، فیلتر، و گرد کردن ضرایب."""
    max_c = np.max(np.abs(coef))
    if max_c < 1e-12:
        return coef
    coef = coef / max_c
    coef[np.abs(coef) < tol_ratio] = 0.0
    int_r = np.round(coef)
    m = (np.abs(coef - int_r) < 0.1) & (np.abs(coef) > 0.2)
    coef[m] = int_r[m]
    m2 = (~m) & (np.abs(coef) > 0)
    coef[m2] = np.round(coef[m2] * 100) / 100
    return coef


def _sparsify_vectors(vectors, n_iter=300, seed=42):
    """چرخش تصادفی برای اسپارس کردن پایه‌ی SVD."""
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
    """استخراج بردارهای فضای پوچ با SVD."""
    cn = np.linalg.norm(Theta, axis=0)
    cn[cn < 1e-12] = 1.0
    Tn = Theta / cn
    U, S, Vt = np.linalg.svd(Tn, full_matrices=False)
    vectors = np.array([v / cn for v in Vt[-n_equations:]])
    if n_equations > 1:
        vectors = _sparsify_vectors(vectors)
    results = [_clean_coef(v.copy(), tol_ratio) for v in vectors]
    return results, S


def _build_library(Xn, degree, freqs):
    Poly, poly_combos = polynomial_library(Xn, degree)
    n_poly = Poly.shape[1]
    Trig, trig_specs = trigonometric_library(Xn, freqs)
    if Trig.shape[1] > 0:
        Full = np.column_stack([Poly, Trig])
    else:
        Full = Poly
    return Full, poly_combos, trig_specs, n_poly


def discover_equations(X, n_equations, degree=3, tol_ratio=0.08, use_trig=True):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-12] = 1.0
    Xn = (X - mu) / sigma

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
        Theta, combos, trig_specs, n_poly = _build_library(Xn, d_try, freqs)
        if Theta.shape[1] <= n_equations:
            continue
        coefs_list, S = _extract_null_vectors(Theta, n_equations, tol_ratio)
        gap = (S[-n_equations - 1] / (S[-n_equations] + 1e-30)
               if len(S) > n_equations else 1.0)
        if best is None or gap > best[0] * 1.3:
            best = (gap, coefs_list, S, combos, trig_specs, n_poly, d_try)

    if best is None:
        print("[!] کتابخانه مناسب ساخته نشد.")
        return [], np.array([]), mu, sigma

    gap, coefs_list, S, combos, trig_specs, n_poly, best_degree = best
    print(f"[i] درجه انتخاب‌شده: {best_degree}")
    print(f"[i] شکاف طیفی: {gap:.2f}")
    print(f"[i] ویژگی‌ها: {n_poly} چندجمله‌ای + {len(trig_specs)} مثلثاتی")

    n_feats = X.shape[1]
    syms = sp.symbols(f"x0:{n_feats}")
    equations = []
    for coef in coefs_list:
        expr = sp.Integer(0)
        for c, combo in zip(coef[:n_poly], combos):
            if abs(c) < 1e-10:
                continue
            term = sp.Integer(1)
            for i in combo:
                term = term * syms[i]
            expr += sp.Float(c) * term
        for c, spec in zip(coef[n_poly:], trig_specs):
            if abs(c) < 1e-10:
                continue
            kind, i, omega = spec
            arg = sp.Float(omega) * syms[i]
            expr += sp.Float(c) * (sp.sin(arg) if kind == "sin" else sp.cos(arg))
        equations.append(expr)
    return equations, S, mu, sigma


def convert_to_original(eq, mu, sigma, n_feats):
    """
    تبدیل به مختصات اصلی + نرمالیزه کردن صحیح.
    نرمالیزه بر اساس ضریب جمله‌ی درجه‌ی دوم (نه همه‌ی جملات).
    """
    xs = sp.symbols(f"x0:{n_feats}")
    Xs = sp.symbols(f"X0:{n_feats}")
    subs = {xs[i]: (Xs[i] - sp.Float(mu[i])) / sp.Float(sigma[i])
            for i in range(n_feats)}
    out = eq.subs(subs)
    try:
        out = sp.expand(out)
    except Exception:
        pass

    # نرمالیزه کردن بر اساس بزرگ‌ترین ضریب درجه‌ی دوم
    try:
        poly = sp.Poly(out, *Xs)
        max_deg2 = 0.0
        for monom, coeff in poly.terms():
            total_deg = sum(monom)
            if total_deg == 2:
                c = abs(float(coeff))
                if c > max_deg2:
                    max_deg2 = c
        if max_deg2 > 1e-12:
            out = out / max_deg2
        else:
            # اگر درجه‌ی دوم نبود، از بزرگ‌ترین ضریب کل استفاده کن
            coeffs = poly.coeffs()
            if coeffs:
                max_c = max(abs(float(c)) for c in coeffs)
                if max_c > 1e-12:
                    out = out / max_c
    except Exception:
        pass
    return out.evalf(4)
