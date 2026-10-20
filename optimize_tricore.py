import numpy as np
from genetic_engine_classifier import evaluate_tree

def load_and_prepare(path):
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    o, h, l, c = data[:,2], data[:,3], data[:,4], data[:,5]
    long_g, short_g = data[:,6], data[:,7]
    long_pnl, short_pnl = data[:,8], data[:,9]
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
    y = np.sign(np.roll(c, -1) - c)
    return X, y, long_pnl, short_pnl

BEST_TREE = ('div', 
    ('add', ('add', 'x7', 'x2'), ('sub', ('const', 0.0), ('const', 0.0))),
    ('add', ('add', ('abs', ('mul', ('sub', ('add', 'x6', 'x1'), ('sub', 'x1', 'x0')), 
    ('div', ('div', 'x1', 'x6'), ('abs', 'x0')))), ('div', ('const', -1.0), 'x5')), 
    ('abs', ('sub', ('abs', 'x2'), ('div', ('const', 0.5), 'x7'))))
)

def core_1_gp(X):
    pred = evaluate_tree(BEST_TREE, X)
    signal = np.zeros_like(pred)
    signal[pred > 0.1] = 1
    signal[pred < -0.1] = -1
    return signal

def core_3_ai_weights(X_train, y_train):
    weights = np.zeros(X_train.shape[1])
    for i in range(X_train.shape[1]):
        if np.std(X_train[:, i]) > 1e-6:
            weights[i] = np.corrcoef(X_train[:, i], y_train)[0, 1]
    return weights

if __name__ == "__main__":
    path = '/storage/emulated/0/forex_bot/golden_points/EURUSD_golden.csv'
    X, y, long_pnl, short_pnl = load_and_prepare(path)
    
    signal_gp = core_1_gp(X)
    
    split = int(len(X) * 0.8)
    w = core_3_ai_weights(X[:split], y[:split])
    raw_pred = np.dot(X, w)
    prob_up = np.clip((raw_pred + 1) / 2, 0, 1)
    
    best_pnl = -999
    best_params = None
    
    print("[*] در حال جستجوی بهترین آستانه‌های وتو...")
    for noise_th in [0.5, 1.0, 1.5, 2.0, 2.5]:
        for liq_th in [0.3, 0.5, 0.8, 1.0]:
            for ai_th in [0.5, 0.55, 0.6, 0.65]:
                final = np.zeros_like(signal_gp)
                for i in range(len(signal_gp)):
                    # فیلترهای هسته ۲
                    if abs(X[i, 2]) > noise_th and X[i, 3] < liq_th:
                        continue
                    
                    if signal_gp[i] == 1 and prob_up[i] > ai_th:
                        final[i] = 1
                    elif signal_gp[i] == -1 and (1 - prob_up[i]) > ai_th:
                        final[i] = -1
                
                total_pnl, wins, losses = 0.0, 0, 0
                for i in range(len(final) - 1):
                    if final[i] == 1:
                        pnl = long_pnl[i+1]
                    elif final[i] == -1:
                        pnl = short_pnl[i+1]
                    else:
                        continue
                    total_pnl += pnl
                    if pnl > 0: wins += 1
                    else: losses += 1
                
                if wins + losses > 50:  # حداقل تعداد معاملات
                    if total_pnl > best_pnl:
                        best_pnl = total_pnl
                        best_params = (noise_th, liq_th, ai_th, wins, losses)
    
    if best_params:
        nt, lt, at, w, l = best_params
        print(f"\n[★] بهترین تنظیمات پیدا شد:")
        print(f"    آستانه نویز: {nt}")
        print(f"    آستانه نقدینگی: {lt}")
        print(f"    آستانه AI: {at}")
        print(f"    معاملات: {w+l}, برد: {100*w/(w+l):.1f}%")
        print(f"    سود کل: {best_pnl:+.4f}")
        if best_pnl > 0:
            print("\n✅ استراتژی سودآور پیدا شد!")
    else:
        print("[!] هیچ تنظیماتی سوددهی پیدا نکرد.")
