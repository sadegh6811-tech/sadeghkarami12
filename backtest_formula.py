import numpy as np
from genetic_engine_classifier import evaluate_tree

# ۱. بازسازی فرمول دقیق از خروجی
# (چون درخت طولانی است، آن را مستقیماً اینجا بازسازی می‌کنیم)
best_tree = ('div', 
    ('add', ('add', 'x7', 'x2'), ('sub', ('const', 0.0), ('const', 0.0))),
    ('add', ('add', ('abs', ('mul', ('sub', ('add', 'x6', 'x1'), ('sub', 'x1', 'x0')), 
    ('div', ('div', 'x1', 'x6'), ('abs', 'x0')))), ('div', ('const', -1.0), 'x5')), 
    ('abs', ('sub', ('abs', 'x2'), ('div', ('const', 0.5), 'x7'))))
)

# ۲. خواندن داده اصلی (برای PnL واقعی)
data = np.loadtxt('/storage/emulated/0/forex_bot/golden_points/EURUSD_golden.csv', delimiter=',', skiprows=1)
close_p = data[:, 5]
long_pnl = data[:, 8]
short_pnl = data[:, 9]

# ۳. محاسبه سیگنال‌ها روی همه داده‌ها
# ویژگی‌های توپولوژیک
o, h, l, c = data[:,2], data[:,3], data[:,4], data[:,5]
long_g, short_g = data[:,6], data[:,7]
range_p = h - l
dim1_pos = (c - np.mean(c)) / (np.std(c) + 1e-9)
dim2_dir = np.sign(c - o)
dim3_noise = ((h - np.maximum(o, c)) - (np.minimum(o, c) - l)) / (range_p + 1e-9)
dim4_liq = range_p / (np.std(range_p) + 1e-9)
dim5_vol = np.abs(long_pnl + short_pnl) / (np.std(long_pnl + short_pnl) + 1e-9)
dim6_buy = long_g
dim7_sell = short_g
dim8_mom = np.gradient(c) / (np.std(np.gradient(c)) + 1e-9)
dim9_time = np.linspace(0, 2*np.pi, len(c))

X = np.column_stack([dim1_pos, dim2_dir, dim3_noise, dim4_liq, dim5_vol, dim6_buy, dim7_sell, dim8_mom, dim9_time])

# پیش‌بینی
pred = evaluate_tree(best_tree, X)
pred_class = np.zeros_like(pred)
pred_class[pred > 0.1] = 1
pred_class[pred < -0.1] = -1

# ۴. محاسبه سود/زیان
total_pnl = 0.0
wins = 0
losses = 0

for i in range(len(pred_class) - 1):
    if pred_class[i] == 1:  # سیگنال خرید
        pnl = long_pnl[i+1] # سود کندل بعدی
        total_pnl += pnl
        if pnl > 0: wins += 1
        else: losses += 1
    elif pred_class[i] == -1: # سیگنال فروش
        pnl = short_pnl[i+1]
        total_pnl += pnl
        if pnl > 0: wins += 1
        else: losses += 1

print("═══ نتیجه Backtest استراتژی ═══")
print(f"[*] تعداد معاملات: {wins + losses}")
print(f"[*] معاملات برنده: {wins} ({100*wins/(wins+losses):.1f}%)")
print(f"[*] معاملات بازنده: {losses}")
print(f"[*] سود/زیان کل: {total_pnl:.4f}")
print(f"[*] میانگین سود هر معامله: {total_pnl/(wins+losses):.6f}")

if total_pnl > 0:
    print("\n✅ استراتژی سودآور است! کشف شما در عمل کار می‌کند.")
else:
    print("\n⚠️ استراتژی در این بازه ضررده بوده (باید روی بازه‌های دیگر تست شود).")
