import numpy as np
import mpmath as mp

# تولید داده
mp.mp.dps = 15
print("[*] تولید ۱۰۰۰ صفر...")
zeros = []
for i in range(1, 1001):
    z = mp.zetazero(i)
    zeros.append((i, float(mp.im(z))))
data = np.array(zeros)
n = data[:, 0]
imag = data[:, 1]

# فرمول ریمان (نسخه ساده)
riemann_simple = 2 * np.pi * n / np.log(n / (2 * np.pi) + 1e-9)

# فرمول ریمان (نسخه کامل با ترم‌های اضافی)
def riemann_full(T):
    T = np.abs(T) + 1e-9
    term1 = (T / (2 * np.pi)) * np.log(T / (2 * np.pi) + 1e-9)
    term2 = -T / (2 * np.pi)
    term3 = 7/8
    term4 = (1/np.pi) * np.arctan(T / (2 * np.pi))
    return term1 + term2 + term3 + term4

riemann_full_vals = riemann_full(n)

# محاسبه MSE
mse_simple = np.mean((imag - riemann_simple)**2)
mse_full = np.mean((imag - riemann_full_vals)**2)

print("\n═══ مقایسه ═══")
print(f"MSE ریمان ساده:  {mse_simple:.2f}")
print(f"MSE ریمان کامل:  {mse_full:.2f}")
print(f"بهبود: {100*(mse_simple - mse_full)/mse_simple:.1f}%")

# مقایسه با فرمول GP (که در کد قبلی کشف شد)
print(f"\nMSE GP:          11743.83")
print(f"MSE ریمان ساده:  {mse_simple:.2f}")
print(f"MSE ریمان کامل:  {mse_full:.2f}")
