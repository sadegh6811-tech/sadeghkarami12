with open('discover.py', 'r') as f:
    content = f.read()

# تابعی برای ساخت کتابخانه با فیلتر
old_build = '''def _build_library(Xn, degree, freqs, use_extended=True):'''
new_build = '''def _build_library(Xn, degree, freqs, use_extended=True, topology="smooth"):'''
content = content.replace(old_build, new_build)

# فیلتر کردن توابع در _build_library
old_filter = '''            # اگر sin/cos داریم، exp رو اضافه نکن
            has_trig = Trig.shape[1] > 0'''
new_filter = '''            # اگر سنسور تشخیص داده که داده مکعب است، فقط از abs استفاده کن
            if topology == "cube":
                from extended_lib import abs_library
                Abs, abs_specs = abs_library(Xn)
                if Abs.shape[1] > 0:
                    parts.append(Abs)
                    all_specs.extend(abs_specs)
                raise Exception("Cube mode: only abs allowed")'''
content = content.replace(old_filter, new_filter)

# در discover_equations، topology را از ورودی بگیر
old_sig = '''def discover_equations(X, n_equations, degree=3, tol_ratio=0.12, use_trig=True):'''
new_sig = '''def discover_equations(X, n_equations, degree=3, tol_ratio=0.12, use_trig=True, topology="smooth"):'''
content = content.replace(old_sig, new_sig)

# ارسال topology به _build_library
old_call = '''        Theta, all_specs, trig_specs, n_poly = _build_library(Xn, d_try, freqs)'''
new_call = '''        Theta, all_specs, trig_specs, n_poly = _build_library(Xn, d_try, freqs, topology=topology)'''
content = content.replace(old_call, new_call)

with open('discover.py', 'w') as f:
    f.write(content)
print("[+] discover.py برای فیلتر کردن کتابخانه در حالت مکعب به‌روزرسانی شد.")
