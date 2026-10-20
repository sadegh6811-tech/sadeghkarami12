import numpy as np
from genetic_engine_classifier import evaluate_tree

def load_data(path):
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    o, h, l, c = data[:,2], data[:,3], data[:,4], data[:,5]
    long_g, short_g = data[:,6], data[:,7]
    long_pnl, short_pnl = data[:,8], data[:,9]
    
    # محاسبه ATR (میانگین دامنه واقعی) برای تعیین حد ضرر و حد سود
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    atr = np.convolve(tr, np.ones(14)/14, mode='same')
    atr[0] = tr[0]
    
    range_p = h - l
    X = np.column_stack([
        (c - np.mean(c)) / (np.std(c) + 1e-9),
        np.sign(c - o),
        ((h - np.maximum(o, c)) - (np.minimum(o, c) - l)) / (range_p + 1e-9),
        range_p / (np.std(range_p) + 1e-9),
        np.abs(long_pnl + short_pnl) / (np.std(long_pnl + short_pnl) + 1e-9),
        long_g, short_g,
        np.gradient(c) / (np.std(np.gradient(c)) + 1e-9),
        np.linspace(0, 2*np.pi, len(c))
    ])
    return X, c, h, l, atr

BEST_TREE = ('div',
    ('add', ('add', 'x7', 'x2'), ('const', 0.0)),
    ('add', 
        ('add', ('abs', ('mul', ('sub', ('add', 'x6', 'x1'), ('sub', 'x1', 'x0')), ('div', ('div', 'x1', 'x6'), ('abs', 'x0')))),
            ('div', ('const', -1.0), 'x5')
        ),
        ('abs', ('sub', ('abs', 'x2'), ('div', ('const', 0.5), 'x7')))
    )
)

def simulate_trades(signals, c, h, l, atr, sl_mult=1.5, tp_mult=3.0, spread=0.0001):
    """شبیه‌ساز حرفه‌ای: با حد ضرر، حد سود و اسپرد"""
    balance = 0.0
    wins, losses = 0, 0
    trades = 0
    
    for i in range(len(signals) - 1):
        if signals[i] == 0: continue
        
        entry_price = c[i]
        atr_val = atr[i]
        if atr_val <= 0: continue
        
        sl_dist = atr_val * sl_mult
        tp_dist = atr_val * tp_mult
        
        trades += 1
        exited = False
        
        # بررسی کندل‌های بعدی (حداکثر ۱۰ کندل صبر می‌کنیم)
        for j in range(i+1, min(i+11, len(c))):
            if signals[i] == 1:  # خرید
                # برخورد با حد ضرر
                if l[j] <= entry_price - sl_dist:
                    balance -= sl_dist + spread
                    losses += 1
                    exited = True
                    break
                # برخورد با حد سود
                if h[j] >= entry_price + tp_dist:
                    balance += tp_dist - spread
                    wins += 1
                    exited = True
                    break
            else:  # فروش
                if h[j] >= entry_price + sl_dist:
                    balance -= sl_dist + spread
                    losses += 1
                    exited = True
                    break
                if l[j] <= entry_price - tp_dist:
                    balance += tp_dist - spread
                    wins += 1
                    exited = True
                    break
        
        # اگر بعد از ۱۰ کندل به TP یا SL نرسید، با قیمت فعلی خارج می‌شویم
        if not exited:
            exit_price = c[min(i+10, len(c)-1)]
            if signals[i] == 1:
                pnl = exit_price - entry_price - spread
            else:
                pnl = entry_price - exit_price - spread
            balance += pnl
            if pnl > 0: wins += 1
            else: losses += 1
    
    return balance, wins, losses, trades

if __name__ == "__main__":
    path = '/storage/emulated/0/forex_bot/golden_points/EURUSD_golden.csv'
    X, c, h, l, atr = load_data(path)
    
    pred = evaluate_tree(BEST_TREE, X)
    pred_sig = np.zeros_like(pred)
    pred_sig[pred > 0.1] = 1
    pred_sig[pred < -0.1] = -1
    
    print("[*] اجرای شبیه‌ساز حرفه‌ای با اسپرد و SL/TP...")
    print(f"[*] اسپرد: 0.0001 (1 پیپ)")
    print(f"[*] حد ضرر: 1.5 برابر ATR")
    print(f"[*] حد سود: 3.0 برابر ATR")
    
    balance, wins, losses, trades = simulate_trades(pred_sig, c, h, l, atr)
    
    print(f"\n═══ نتیجه شبیه‌سازی حرفه‌ای ═══")
    print(f"تعداد کل معاملات: {trades}")
    print(f"معاملات برنده: {wins} ({100*wins/max(trades,1):.1f}%)")
    print(f"معاملات بازنده: {losses}")
    print(f"موجودی نهایی: {balance:+.4f}")
    
    if balance > 0:
        print("\n✅ پیروزی! فرمول ۸۱٪ شما با مدیریت ریسک، سودآور است.")
    else:
        print("\n[!] نیاز به تنظیم SL/TP دارد.")
