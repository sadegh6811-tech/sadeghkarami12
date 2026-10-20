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
    future_c = np.roll(c, -1)
    target = np.sign(future_c - c)
    return X, target, long_pnl, short_pnl

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
    X, target, long_pnl, short_pnl = load_data(path)
    
    pred = evaluate_tree(BEST_TREE, X)
    pred_sig = np.zeros_like(pred)
    pred_sig[pred > 0.1] = 1
    pred_sig[pred < -0.1] = -1
    
    # پیدا کردن ۱۹٪ خطاها
    errors = (pred_sig != target) & (pred_sig != 0) & (target != 0)
    corrects = (pred_sig == target) & (pred_sig != 0) & (target != 0)
    
    print(f"[*] موارد موفق (۸۱٪): {np.sum(corrects)}")
    print(f"[*] موارد خطا (۱۹٪): {np.sum(errors)}")
    
    # بررسی ویژگی‌های ۱۹٪ خطا
    print("\n═══ ویژگی‌های ۱۹٪ خطا ═══")
    features_names = ['position', 'direction', 'noise', 'liquidity', 'volume', 'buy', 'sell', 'momentum', 'time']
    for i, name in enumerate(features_names):
        mean_err = np.mean(np.abs(X[errors, i]))
        mean_cor = np.mean(np.abs(X[corrects, i]))
        diff = mean_err - mean_cor
        if abs(diff) > 0.1:
            print(f"  {name:10s}: خطا={mean_err:.3f}, موفق={mean_cor:.3f}, تفاوت={diff:+.3f} {'← مهم' if abs(diff) > 0.3 else ''}")
    
    # ساخت سنسور جبران‌کننده: اگر نویز بالا و مومنتوم ضعیف بود، سیگنال را رد کن
    print("\n═══ ساخت سنسور جبران‌کننده ═══")
    noise = X[:, 2]
    mom = X[:, 7]
    liq = X[:, 3]
    
    # فیلتر جدید: در شرایطی که نویز بالا و مومنتوم ضعیف است، معامله نکن
    compensation_sensor = ~((np.abs(noise) > 0.5) & (np.abs(mom) < 0.3))
    
    final_sig = np.where(compensation_sensor, pred_sig, 0)
    
    # ارزیابی با سنسور
    total_pnl, wins, losses = 0.0, 0, 0
    for i in range(len(final_sig) - 1):
        if final_sig[i] == 1: pnl = long_pnl[i+1]
        elif final_sig[i] == -1: pnl = short_pnl[i+1]
        else: continue
        total_pnl += pnl
        if pnl > 0: wins += 1
        else: losses += 1
    
    print(f"\n═══ نتیجه با سنسور جبران‌کننده ═══")
    print(f"معاملات: {wins+losses}, برد: {100*wins/max(wins+losses,1):.1f}%")
    print(f"سود کل: {total_pnl:+.4f}")
