import numpy as np
from genetic_engine_classifier import evaluate_tree

# ═══════════════════════════════════════════════════
# آماده‌سازی داده‌ها
# ═══════════════════════════════════════════════════
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
    future_c = np.roll(c, -1)
    y = np.sign(future_c - c)
    
    return X, y, c, long_pnl, short_pnl

# ═══════════════════════════════════════════════════
# هسته ۱: فرمول کشف‌شده GP
# ═══════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════
# هسته ۳: هوش مصنوعی (جایگزین سبک با آستانه‌گذاری آماری)
# ═══════════════════════════════════════════════════
def core_3_ai(X_train, y_train, X_test):
    # یک مدل سبک: میانگین‌گیری وزنی از ابعاد کلیدی در داده آموزش
    # این جایگزین سبک Random Forest است
    weights = np.zeros(X_train.shape[1])
    for i in range(X_train.shape[1]):
        if np.std(X_train[:, i]) > 1e-6:
            weights[i] = np.corrcoef(X_train[:, i], y_train)[0, 1]
    
    # پیش‌بینی با میانگین وزنی
    raw_pred = np.dot(X_test, weights)
    
    # تبدیل به احتمال (0 تا 1)
    prob_up = np.clip((raw_pred + 1) / 2, 0, 1)
    prob_down = 1 - prob_up
    
    # ساخت آرایه احتمال مشابه Random Forest
    prob_array = np.column_stack([prob_down, np.zeros(len(prob_up)), prob_up])
    
    # پیش‌بینی کلاس
    pred = np.zeros(len(prob_up))
    pred[prob_up > 0.55] = 1
    pred[prob_down > 0.55] = -1
    return pred, prob_array

# ═══════════════════════════════════════════════════
# هسته ۲: تصمیم‌گیری نهایی (Veto Logic)
# ═══════════════════════════════════════════════════
def core_2_decision(signal_gp, prob_ai, X):
    noise = X[:, 2]
    liq = X[:, 3]
    
    final_signal = np.zeros_like(signal_gp)
    
    for i in range(len(signal_gp)):
        # شرط: هر دو هسته هم‌جهت باشند
        if signal_gp[i] == 1 and prob_ai[i][2] > 0.5:
            final_signal[i] = 1
        elif signal_gp[i] == -1 and prob_ai[i][0] > 0.5:
            final_signal[i] = -1
        
        # وتوی نویز: اگر نویز خیلی بالا بود، معامله نکن
        if abs(noise[i]) > 1.5 and liq[i] < 0.5:
            final_signal[i] = 0
            
    return final_signal

# ═══════════════════════════════════════════════════
# اجرای ربات
# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    path = '/storage/emulated/0/forex_bot/golden_points/EURUSD_golden.csv'
    X, y, c, long_pnl, short_pnl = load_and_prepare(path)
    
    print("[*] ربات سه‌هسته‌ای در حال راه‌اندازی...")
    
    # اجرای هسته ۱
    signal_gp = core_1_gp(X)
    
    # آموزش و اجرای هسته ۳
    split = int(len(X) * 0.8)
    signal_ai, prob_ai = core_3_ai(X[:split], y[:split], X)
    
    # اجرای هسته ۲
    final_signal = core_2_decision(signal_gp, prob_ai, X)
    
    # محاسبه نتیجه
    total_pnl, wins, losses = 0.0, 0, 0
    for i in range(len(final_signal) - 1):
        if final_signal[i] == 1:
            pnl = long_pnl[i+1]
            total_pnl += pnl
            if pnl > 0: wins += 1
            else: losses += 1
        elif final_signal[i] == -1:
            pnl = short_pnl[i+1]
            total_pnl += pnl
            if pnl > 0: wins += 1
            else: losses += 1
            
    print("\n═══ نتیجه نهایی ربات سه‌هسته‌ای ═══")
    print(f"[*] تعداد کل سیگنال‌های صادرشده: {np.sum(final_signal != 0)}")
    print(f"[*] تعداد معاملات انجام‌شده: {wins + losses}")
    print(f"[*] معاملات برنده: {wins} ({100*wins/max(wins+losses,1):.1f}%)")
    print(f"[*] سود/زیان کل: {total_pnl:.4f}")
