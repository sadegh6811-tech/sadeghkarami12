import numpy as np

data = np.loadtxt('/data/data/com.termux/files/home/zeta_data_1000.csv', delimiter=',', skiprows=1)
T = data[:, 2]
actual = data[:, 3]

def riemann_formula(T):
    return 2 * np.pi / np.log(T / (2 * np.pi) + 1e-9)

# پیدا کردن ثابت بهینه برای ریمان (جستجوی شبکه‌ای دقیق)
best_C, best_mse = 1.0, float('inf')
base = riemann_formula(T)
mask = np.isfinite(base) & np.isfinite(actual)
base = base[mask]
actual = actual[mask]

for C in np.linspace(0.5, 1.5, 1000):
    mse = np.mean((actual - C * base)**2)
    if mse < best_mse:
        best_mse = mse
        best_C = C

print("═══ تصحیح فرمول ریمان-فون با ثابت C ═══")
print(f"ثابت بهینه C: {best_C:.4f}")
print(f"MSE بدون ثابت: {np.mean((actual - base)**2):.6f}")
print(f"MSE با ثابت:   {best_mse:.6f}")
print(f"بهبود: {100*(np.mean((actual - base)**2) - best_mse) / np.mean((actual - base)**2):.1f}%")

# مقایسه نهایی
print("\n═══ مقایسه نهایی ═══")
print(f"فرمول ریمان (خام):  MSE = {np.mean((actual - base)**2):.6f}")
print(f"فرمول ریمان (با C):  MSE = {best_mse:.6f}")
print(f"فرمول GP (با C):     MSE = 2.121711")
