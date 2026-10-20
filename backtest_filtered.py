import numpy as np
from genetic_engine_classifier import evaluate_tree

best_tree = ('div', 
    ('add', ('add', 'x7', 'x2'), ('sub', ('const', 0.0), ('const', 0.0))),
    ('add', ('add', ('abs', ('mul', ('sub', ('add', 'x6', 'x1'), ('sub', 'x1', 'x0')), 
    ('div', ('div', 'x1', 'x6'), ('abs', 'x0')))), ('div', ('const', -1.0), 'x5')), 
    ('abs', ('sub', ('abs', 'x2'), ('div', ('const', 0.5), 'x7'))))
)

data = np.loadtxt('/storage/emulated/0/forex_bot/golden_points/EURUSD_golden.csv', delimiter=',', skiprows=1)
c = data[:, 5]
long_pnl = data[:, 8]
short_pnl = data[:, 9]
o, h, l = data[:, 2], data[:, 3], data[:, 4]
long_g, short_g = data[:, 6], data[:, 7]
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
pred = evaluate_tree(best_tree, X)

# تغییر استراتژی: فقط سیگنال‌های قوی
for threshold in [0.1, 0.3, 0.5, 0.7, 1.0]:
    pred_class = np.zeros_like(pred)
    pred_class[pred > threshold] = 1
    pred_class[pred < -threshold] = -1
    
    total_pnl, wins, losses = 0.0, 0, 0
    for i in range(len(pred_class) - 1):
        if pred_class[i] == 1:
            pnl = long_pnl[i+1]
            total_pnl += pnl
            if pnl > 0: wins += 1
            else: losses += 1
        elif pred_class[i] == -1:
            pnl = short_pnl[i+1]
            total_pnl += pnl
            if pnl > 0: wins += 1
            else: losses += 1
            
    total = wins + losses
    if total > 0:
        print(f"Threshold={threshold:.1f} | معاملات={total:4d} | برد={100*wins/total:5.1f}% | سود کل={total_pnl:+.4f}")
    else:
        print(f"Threshold={threshold:.1f} | هیچ معامله‌ای انجام نشد")
