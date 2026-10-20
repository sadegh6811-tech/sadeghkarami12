import numpy as np
from genetic_engine_classifier import evaluate_tree

def load_data(path):
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    o, h, l, c = data[:,2], data[:,3], data[:,4], data[:,5]
    long_g, short_g = data[:,6], data[:,7]
    long_pnl, short_pnl = data[:,8], data[:,9]
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
    return X, c, h, l

BEST_TREE = ('div',
    ('add', ('add', 'x7', 'x2'), ('const', 0.0)),
    ('add', 
        ('add', ('abs', ('mul', ('sub', ('add', 'x6', 'x1'), ('sub', 'x1', 'x0')), ('div', ('div', 'x1', 'x6'), ('abs', 'x0')))),
            ('div', ('const', -1.0), 'x5')
        ),
        ('abs', ('sub', ('abs', 'x2'), ('div', ('const', 0.5), 'x7')))
    )
)

def simulate_correct(signals, c, h, l, spread=0.00001):
    """شبیه‌ساز اصلاح‌شده با تنظیمات واقعی"""
    balance = 0.0
    wins, losses = 0, 0
    trades = 0
    
    sl_pct = 0.002  # حد ضرر ۰.۲٪
    tp_pct = 0.006  # حد سود ۰.۶٪ (نسبت ۱ به ۳)
    
    for i in range(len(signals) - 1):
        if signals[i] == 0: continue
        
        entry_price = c[i]
        trades += 1
        exited = False
        
        sl_price = entry_price * (1 - sl_pct) if signals[i] == 1 else entry_price * (1 + sl_pct)
        tp_price = entry_price * (1 + tp_pct) if signals[i] == 1 else entry_price * (1 - tp_pct)
        
        for j in range(i+1, min(i+20, len(c))):
            if signals[i] == 1:  # خرید
                if l[j] <= sl_price:
                    balance -= (entry_price - sl_price) + spread
                    losses += 1
                    exited = True
                    break
                if h[j] >= tp_price:
                    balance += (tp_price - entry_price) - spread
                    wins += 1
                    exited = True
                    break
            else:  # فروش
                if h[j] >= sl_price:
                    balance -= (sl_price - entry_price) + spread
                    losses += 1
                    exited = True
                    break
                if l[j] <= tp_price:
                    balance += (entry_price - tp_price) - spread
                    wins += 1
                    exited = True
                    break
        
        if not exited:
            exit_price = c[min(i+20, len(c)-1)]
            pnl = (exit_price - entry_price) if signals[i] == 1 else (entry_price - exit_price)
            balance += pnl - spread
            if pnl > 0: wins += 1
            else: losses += 1
    
    return balance, wins, losses, trades

if __name__ == "__main__":
    path = '/storage/emulated/0/forex_bot/golden_points/EURUSD_golden.csv'
    X, c, h, l = load_data(path)
    
    pred = evaluate_tree(BEST_TREE, X)
    pred_sig = np.zeros_like(pred)
    pred_sig[pred > 0.1] = 1
    pred_sig[pred < -0.1] = -1
    
    print("[*] اجرای شبیه‌ساز اصلاح‌شده...")
    balance, wins, losses, trades = simulate_correct(pred_sig, c, h, l)
    
    print(f"\n═══ نتیجه شبیه‌سازی اصلاح‌شده ═══")
    print(f"تعداد معاملات: {trades}")
    print(f"معاملات برنده: {wins} ({100*wins/max(trades,1):.1f}%)")
    print(f"معاملات بازنده: {losses}")
    print(f"موجودی نهایی: {balance:+.4f}")
    
    if balance > 0:
        print("\n✅ پیروزی! استراتژی با تنظیمات واقعی سودآور است.")
    else:
        print("\n[!] نیاز به تنظیم بیشتر SL/TP دارد.")
