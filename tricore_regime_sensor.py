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
    y = np.sign(np.roll(c, -1) - c)
    return X, y, long_pnl, short_pnl, c

BEST_TREE = ('div',
    ('add', ('add', 'x7', 'x2'), ('const', 0.0)),
    ('add', 
        ('add', 
            ('abs', ('mul', ('sub', ('add', 'x6', 'x1'), ('sub', 'x1', 'x0')), ('div', ('div', 'x1', 'x6'), ('abs', 'x0')))),
            ('div', ('const', -1.0), 'x5')
        ),
        ('abs', ('sub', ('abs', 'x2'), ('div', ('const', 0.5), 'x7')))
    )
)

def core_1_gp(X):
    pred = evaluate_tree(BEST_TREE, X)
    sig = np.zeros_like(pred)
    sig[pred > 0.1] = 1
    sig[pred < -0.1] = -1
    return sig

# --- هسته ۳: میانگین متحرک ساده ---
def core_3_ma(c, short_window=5, long_window=20):
    ma_short = np.convolve(c, np.ones(short_window)/short_window, mode='same')
    ma_long = np.convolve(c, np.ones(long_window)/long_window, mode='same')
    sig = np.zeros_like(c)
    sig[ma_short > ma_long] = 1
    sig[ma_short < ma_long] = -1
    return sig

# --- سنسور رژیم بازار ---
def regime_sensor(sig_gp, sig_ma, c, long_pnl, short_pnl):
    # تشخیص رژیم: اگر نوسان ۲۰ کندل اخیر خیلی کم باشد، بازار رنج است
    volatility = np.std(np.diff(c))
    rolling_vol = np.array([np.std(np.diff(c[max(0,i-20):i+1])) for i in range(len(c))])
    
    final_sig = np.zeros_like(sig_gp)
    for i in range(len(sig_gp)):
        # وتو کردن در بازار رنج
        if rolling_vol[i] < 0.3 * volatility: continue
        
        # تأیید دوگانه
        if sig_gp[i] != 0 and sig_ma[i] == sig_gp[i]:
            final_sig[i] = sig_gp[i]
    return final_sig

if __name__ == "__main__":
    path = '/storage/emulated/0/forex_bot/golden_points/EURUSD_golden.csv'
    X, y, long_pnl, short_pnl, c = load_data(path)
    print("[*] راه‌اندازی هسته ۱ (GP)...")
    sig_gp = core_1_gp(X)
    print("[*] راه‌اندازی هسته ۳ (MA)...")
    sig_ma = core_3_ma(c)
    print("[*] فعال‌سازی سنسور رژیم بازار...")
    final = regime_sensor(sig_gp, sig_ma, c, long_pnl, short_pnl)
    
    total_pnl, wins, losses = 0.0, 0, 0
    for i in range(len(final)-1):
        if final[i] == 1: pnl = long_pnl[i+1]
        elif final[i] == -1: pnl = short_pnl[i+1]
        else: continue
        total_pnl += pnl
        if pnl > 0: wins += 1
        else: losses += 1
    
    print(f"\n═══ نتیجه ربات با سنسور رژیم بازار ═══")
    print(f"معاملات: {wins+losses}, برد: {100*wins/max(wins+losses,1):.1f}%")
    print(f"سود کل: {total_pnl:+.4f}")
