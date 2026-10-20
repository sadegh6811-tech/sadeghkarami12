"""اصلاح discover_explicit برای کشف مستقیم y = f(x)."""
import re

# خواندن فایل
with open('discover.py', 'r') as f:
    content = f.read()

# پیدا کردن تابع قدیمی
pattern = r'def discover_explicit\(X, y, degree=3, tol_ratio=0\.08, use_trig=True\):.*?return expr, coef, mu, sigma, y\.mean\(\), y\.std\(\)'

# کد جدید
new_func = '''def discover_explicit(X, y, degree=3, tol_ratio=0.05, use_trig=True):
    """کشف معادله صریح y = f(x) — مستقیماً در فضای اصلی."""
    from mathbot_core import polynomial_library, trigonometric_library
    
    # نرمال‌سازی X
    sigma = X.std(axis=0) + 1e-12
    Xn = X / sigma
    
    # ساخت کتابخانه
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
    
    # extended (sqrt, log, inv)
    ext_specs = []
    try:
        from extended_lib import sqrt_library, log_library, inverse_library
        Sqrt, sqrt_specs = sqrt_library(Xn)
        Log, log_specs = log_library(Xn)
        Inv, inv_specs = inverse_library(Xn)
        if Sqrt.shape[1] > 0:
            parts.append(Sqrt)
            ext_specs.extend(sqrt_specs)
        if Log.shape[1] > 0:
            parts.append(Log)
            ext_specs.extend(log_specs)
        if Inv.shape[1] > 0:
            parts.append(Inv)
            ext_specs.extend(inv_specs)
    except Exception:
        pass
    
    Full = np.column_stack(parts)
    
    # رگرسیون
    coef, _, _, _ = np.linalg.lstsq(Full, y, rcond=None)
    
    # ساخت معادله sympy در فضای اصلی
    import sympy as sp
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
    
    return expr, coef, np.zeros(n_feats), sigma, y.mean(), y.std()'''

# جایگزینی
content_new = re.sub(pattern, new_func, content, flags=re.DOTALL)

if content_new != content:
    with open('discover.py', 'w') as f:
        f.write(content_new)
    print("[+] discover.py با موفقیت اصلاح شد")
else:
    print("[!] الگو پیدا نشد. لطفاً خطوط ۳۸۰ به بعد discover.py را چک کنید.")

# اصلاح mathbot.py
with open('mathbot.py', 'r') as f:
    mb_content = f.read()

old_mb = '''        from discover import discover_explicit
        eq, coefs, mu_x, sigma_x, y_mu, y_sigma = discover_explicit(X_features, y, degree=max_degree)
        
        eq_orig = convert_to_original(eq, mu_x, sigma_x, X_features.shape[1])
        
        import sympy as sp
        y_sym = sp.Symbol('y')
        eq_final = sp.Eq(y_sym, eq_orig.evalf(4))
        print(f"\\n[✓] معادله صریح:")
        print(f"    {eq_final}")
        return'''

new_mb = '''        from discover import discover_explicit
        eq, coefs, mu_x, sigma_x, y_mu, y_sigma = discover_explicit(X_features, y, degree=max_degree)
        
        import sympy as sp
        y_sym = sp.Symbol('y')
        eq_final = sp.Eq(y_sym, eq.evalf(4))
        print(f"\\n[✓] معادله صریح:")
        print(f"    {eq_final}")
        return'''

mb_new = mb_content.replace(old_mb, new_mb)
if mb_new != mb_content:
    with open('mathbot.py', 'w') as f:
        f.write(mb_new)
    print("[+] mathbot.py با موفقیت اصلاح شد")
else:
    print("[!] الگو در mathbot.py پیدا نشد.")
