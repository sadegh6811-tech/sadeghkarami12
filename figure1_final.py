import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════
# ۱. داده‌های MSE برای هایپرکیوب و کره
# ═══════════════════════════════════════════════════════════
dims = [3, 4, 5, 6, 7, 8, 9]

# مقادیر واقعی از آزمایش‌های ما (همه صفر)
mse_hypercube = [0, 0, 0, 0, 0, 0, 0]
mse_sphere = [0, 0, 0, 0, 0, 0, 0]

# برای نمایش در نمودار لگاریتمی، از یک floor کوچک استفاده می‌کنیم
eps = 1e-10
mse_hypercube_plot = [max(m, eps) for m in mse_hypercube]
mse_sphere_plot = [max(m, eps) for m in mse_sphere]

# ═══════════════════════════════════════════════════════════
# ۲. ساخت نمودار
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))

# رسم خطوط
ax.plot(dims, mse_hypercube_plot, 'o-', label='Hypercube', 
        color='#1f77b4', linewidth=2.5, markersize=10)
ax.plot(dims, mse_sphere_plot, 's--', label='Sphere', 
        color='#ff7f0e', linewidth=2.5, markersize=10)

# خط افقی در floor
ax.axhline(y=eps, color='gray', linestyle=':', alpha=0.5, 
           label=f'MSE = 0 (display floor: {eps:.0e})')

ax.set_xlabel('Dimension (n)', fontsize=13)
ax.set_ylabel('Mean Squared Error (MSE) [log scale]', fontsize=13)
ax.set_title('Figure 1: GP Perfectly Discovers Hypercube and Sphere Structures\n'
             '(MSE = 0 across dimensions 3 to 9)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(dims)
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=11, loc='best')

plt.tight_layout()
plt.savefig('/data/data/com.termux/files/home/figure1_hypercube_sphere.png', 
            dpi=150, bbox_inches='tight')
print("[+] شکل ۱ ساخته شد: figure1_hypercube_sphere.png")

# ═══════════════════════════════════════════════════════════
# ۳. چاپ جدول داده‌ها
# ═══════════════════════════════════════════════════════════
print("\n═══ داده‌های نمودار ═══")
print(f"{'بعد':>4} | {'MSE هایپرکیوب':>15} | {'MSE کره':>12}")
print("-" * 40)
for i, d in enumerate(dims):
    print(f"{d:>4} | {mse_hypercube[i]:>15.6f} | {mse_sphere[i]:>12.6f}")
