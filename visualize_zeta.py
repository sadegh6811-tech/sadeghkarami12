import numpy as np

# خواندن داده‌های زتا
data = np.loadtxt('/data/data/com.termux/files/home/zeta_data.csv', delimiter=',', skiprows=1)
index = data[:, 0]
real = data[:, 1]
imag = data[:, 2]
gap = data[:, 3]

def ascii_plot(x, y, title, xlabel, ylabel, width=60, height=20):
    """رسم نمودار پراکندگی در ترمینال"""
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    # ایجاد شبکه
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    for i in range(len(x)):
        col = int((x[i] - x_min) / (x_max - x_min + 1e-9) * (width - 1))
        row = int((y[i] - y_min) / (y_max - y_min + 1e-9) * (height - 1))
        row = height - 1 - row  # معکوس کردن محور y
        grid[row][col] = '*'
    
    print(f"\n═══ {title} ═══")
    print(f"  ↑ {ylabel}")
    for r in grid:
        print("  |" + "".join(r))
    print("  +" + "-" * width + "→")
    print(f"   {xlabel}")
    print(f"   ({x_min:.1f} تا {x_max:.1f})")

# ۱. رسم Real vs Index (آیا صفرها روی خط 0.5 هستند؟)
ascii_plot(index, real, "Real vs Index (تأیید فرضیه ریمان)", "Index (n)", "Real Part", width=60, height=10)

# ۲. رسم Imag vs Index (شیب حرکت صفرها)
ascii_plot(index, imag, "Imag vs Index (رشد صفرها)", "Index (n)", "Imaginary Part", width=60, height=20)

# ۳. رسم Gap vs Index (ساختار فاصله‌ها)
ascii_plot(index, gap, "Gap vs Index (فاصله بین صفرها)", "Index (n)", "Gap", width=60, height=20)

# ۴. رسم Imag vs Real (صفحه‌ی مختلط)
ascii_plot(real, imag, "صفحه مختلط (Real vs Imag)", "Real Part", "Imaginary Part", width=20, height=20)

# ۵. تحلیل عددی
print("\n═══ تحلیل عددی ═══")
print(f"انحراف معیار Real: {np.std(real):.6f}")
print(f"(اگر نزدیک صفر باشد، صفرها دقیقاً روی خط 0.5 هستند)")
print(f"\nهمبستگی Index و Imag: {np.corrcoef(index, imag)[0,1]:.6f}")
print(f"همبستگی Index و Gap: {np.corrcoef(index, gap)[0,1]:.6f}")
