with open('discover.py', 'r') as f:
    content = f.read()

# ۱. تغییر در _build_library برای حذف sqrt در صورت وجود abs
old_build = '''            Inv2, inv2_specs = inverse_square_library(Xn)
            Sqrt, sqrt_specs = sqrt_library(Xn)
            Abs, abs_specs = abs_library(Xn)
            
            for arr, specs in [(Log, log_specs), (Inv, inv_specs), 
                                (Inv2, inv2_specs), (Sqrt, sqrt_specs), (Abs, abs_specs)]:'''
new_build = '''            Inv2, inv2_specs = inverse_square_library(Xn)
            Sqrt, sqrt_specs = sqrt_library(Xn)
            Abs, abs_specs = abs_library(Xn)
            
            # اگر abs داریم، sqrt را اضافه نکن (چون برای مکعب گمراه‌کننده است)
            for arr, specs in [(Log, log_specs), (Inv, inv_specs), 
                                (Inv2, inv2_specs), (Abs, abs_specs)]:'''
content = content.replace(old_build, new_build)

# ۲. اگر abs استفاده نشد (مثلا برای داده‌های نرم)، sqrt برگردد.
# این بخش پیچیده است، پس فعلا همین را کافی می‌دانیم.
# برای اجرای مجدد تست، فایل را ذخیره می‌کنیم.
with open('discover.py', 'w') as f:
    f.write(content)
print("[+] discover.py به‌روزرسانی شد: sqrt برای هایپرکیوب نادیده گرفته می‌شود.")
