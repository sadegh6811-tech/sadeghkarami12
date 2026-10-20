import re

with open('discover.py', 'r') as f:
    content = f.read()

# جایگزینی کل تابع _build_library با یک نسخه‌ی سخت‌گیرانه
pattern = r'def _build_library\(Xn, degree, freqs, use_extended=True, topology="smooth"\):.*?return Full, all_specs, trig_specs, n_poly'

new_build = '''def _build_library(Xn, degree, freqs, use_extended=True, topology="smooth"):
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
    return Full, all_specs, trig_specs, n_poly'''

content = re.sub(pattern, new_build, content, flags=re.DOTALL)

# همچنین در تولید معادله، باید مطمئن شویم abs درست ترجمه می‌شود
old_expr = '''                elif kind == "abs":
                    expr += sp.Float(c) * sp.Abs(xi)'''
new_expr = '''                elif kind == "abs":
                    expr += sp.Float(c) * sp.Abs(xi)'''
content = content.replace(old_expr, new_expr)

with open('discover.py', 'w') as f:
    f.write(content)
print("[+] discover.py برای استفاده اجباری از abs در حالت مکعب بازنویسی شد.")
