import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════
# داده‌های واقعی (از نتایج v8)
# ═══════════════════════════════════════════════════════════
equations = ['I.9.18\n(گرانش)', 'I.12.1\n(اصطکاک)', 'I.13.4\n(جنبشی)', 
             'I.8.14\n(فاصله)', 'II.2.42\n(کولن)', 'I.6.20\n(گاوسی)']
our_r2 = [0.991, 0.978, 0.960, 0.950, 0.940, 0.934]

# میانگین
mean_r2 = np.mean(our_r2)

# ═══════════════════════════════════════════════════════════
# نمودار میله‌ای
# ═══════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 1]})

# ── بخش ۱: R² برای هر معادله ──
colors = ['#2ecc71' if r2 > 0.95 else '#f39c12' for r2 in our_r2]
bars = ax1.bar(range(len(equations)), our_r2, color=colors, edgecolor='black', linewidth=1.5)

# خط ۰.۹
ax1.axhline(y=0.9, color='red', linestyle='--', linewidth=2, label='Target (R² = 0.9)')

# برچسب‌ها
for bar, r2 in zip(bars, our_r2):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{r2:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_xticks(range(len(equations)))
ax1.set_xticklabels(equations, fontsize=10)
ax1.set_ylabel('R² (Coefficient of Determination)', fontsize=13)
ax1.set_title('Figure 4a: Feynman Benchmark Results (Our Model)\n'
              f'All 6 equations achieve R² > 0.93 (Mean = {mean_r2:.4f})',
              fontsize=13, fontweight='bold')
ax1.set_ylim(0.85, 1.02)
ax1.legend(fontsize=11, loc='lower right')
ax1.grid(axis='y', alpha=0.3)

# ── بخش ۲: مقایسه با رقبا ──
methods = ['Our Model\n(v8)', 'PySR\n(2023)', 'AI-Feynman\n(2019)']
scores = [mean_r2, 0.95, 0.99]
colors_comp = ['#3498db', '#e74c3c', '#9b59b6']

bars2 = ax2.bar(range(len(methods)), scores, color=colors_comp, edgecolor='black', linewidth=1.5)

for bar, score in zip(bars2, scores):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{score:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax2.set_xticks(range(len(methods)))
ax2.set_xticklabels(methods, fontsize=9)
ax2.set_ylabel('Mean R²', fontsize=13)
ax2.set_title('Figure 4b: Comparison with Competitors\n'
              '(Note: AI-Feynman tested on 100 eqs, ours on 6)',
              fontsize=12, fontweight='bold')
ax2.set_ylim(0.85, 1.02)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/data/data/com.termux/files/home/figure4_feynman.png', 
            dpi=150, bbox_inches='tight')
print("[+] شکل ۴ ساخته شد: figure4_feynman.png")
print(f"[*] میانگین R² ربات: {mean_r2:.4f}")
print(f"[*] تعداد معادلات R² > 0.9: {sum(1 for r in our_r2 if r > 0.9)}/{len(our_r2)}")
