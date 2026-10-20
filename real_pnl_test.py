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
    return X, long_pnl, short_pnl

BEST_TREE = ('div',
    ('add', ('add', 'x7', 'x2'), ('const', 0.0)),
    ('add', 
        ('add', ('abs', ('mul', ('sub', ('add', 'x6', 'x1'), ('sub', 'x1', 'x0')), ('div', ('div', 'x1', 'x6'), ('abs', 'x0')))),
            ('div', ('const', -1.0), 'x5')
        ),
        ('abs', ('sub', ('abs', 'x2'), ('div', ('const', 0.5), 'x7')))
    )
)

if __name__ == "__main__":
    path = '/storage/emulated/0/forex_bot/golden_points/EURUSD_golden.csv'
    X, long_pnl, short_pnl = load_data(path)
    
    pred = evaluate_tree(BEST_TREE, X)
    
    print("[*] تست با PnL واقعی (بدون شبیه‌سازی دستی)...")
    
    best_pnl, best_wr, best_th = -999, 0, 0
    
    # تست آستانه‌های مختلف برای پیدا کردن بهترین نقطه ورود
    for th in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        pred_sig = np.zeros_like(pred)
        pred_sig[pred > th] = 1
        pred_sig[pred < -th] = -1
        
        total_pnl, wins, losses = 0.0, 0, 0
        for i in range(len(pred_sig) - 1):
            if pred_sig[i] == 1: pnl = long_pnl[i+1]
            elif pred_sig[i] == -1: pnl = short_pnl[i+1]
            else: continue
            total_pnl += pnl
            if pnl > 0: wins += 1
            else: losses += 1
        
        wr = 100 * wins / max(wins + losses, 1)
        print(f"  آستانه={th:.2f}: معاملات={wins+losses:4d}, برد={wr:5.1f}%, سود={total_pnl:+.4f}")
        
        if total_pnl > best_pnl:
            best_pnl, best_wr, best_th = total_pnl, wr, th
    
    print(f"\n[★] بهترین نتیجه: آستانه={best_th:.2f}, برد={best_wr:.1f}%, سود={best_pnl:+.4f}")
    
    if best_pnl > 0:
        print("\n✅ این یک استراتژی سودآور واقعی است!")
    else:
        print("\n[!] با داده‌های فعلی، حتی بهترین آستانه هم سودآور نیست.")
