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

# --- هسته‌ها ---
def core_1_gp(X):
    pred = evaluate_tree(BEST_TREE, X)
    sig = np.zeros_like(pred)
    sig[pred > 0.1] = 1
    sig[pred < -0.1] = -1
    return sig

def core_3_ai(X_train, y_train, X_test):
    weights = np.zeros(X_train.shape[1])
    for i in range(X_train.shape[1]):
        if np.std(X_train[:, i]) > 1e-6:
            weights[i] = np.corrcoef(X_train[:, i], y_train)[0, 1]
    raw_pred = np.dot(X_test, weights)
    prob_up = np.clip((raw_pred + 1) / 2, 0, 1)
    prob_down = 1 - prob_up
    return prob_up, prob_down

def core_2_veto_logic(sig_gp, prob_up, prob_down, X, noise_th=1.5, liq_th=0.5):
    noise = X[:, 2]
    liq = X[:, 3]
    final = np.zeros_like(sig_gp)
    for i in range(len(sig_gp)):
        if abs(noise[i]) > noise_th and liq[i] < liq_th:
            final[i] = 0
            continue
        if sig_gp[i] == 1 and prob_up[i] > 0.5: final[i] = 1
        elif sig_gp[i] == -1 and prob_down[i] > 0.5: final[i] = -1
    return final

# --- ارزیابی ---
def evaluate_signals(final_sig, long_pnl, short_pnl):
    total_pnl, wins, losses = 0.0, 0, 0
    for i in range(len(final_sig) - 1):
        if final_sig[i] == 1:
            pnl = long_pnl[i+1]
        elif final_sig[i] == -1:
            pnl = short_pnl[i+1]
        else:
            continue
        total_pnl += pnl
        if pnl > 0: wins += 1
        else: losses += 1
    return total_pnl, wins, losses

if __name__ == "__main__":
    path = '/storage/emulated/0/forex_bot/golden_points/EURUSD_golden.csv'
    X, y, long_pnl, short_pnl = load_and_prepare(path)
    split = int(len(X) * 0.8)
    
    print("[*] در حال اجرای ترکیبات...")
    
    # پیش‌بینی‌های خام
    sig_gp = core_1_gp(X)
    prob_up, prob_down = core_3_ai(X[:split], y[:split], X)
    
    # ─── ۱. حالت موازی (رأی‌گیری اکثریت) ───
    # فرض: اگر GP بگوید بالا و AI هم بگوید بالا → بالا
    # اگر GP بگوید پایین و AI هم بگوید پایین → پایین
    # در غیر این صورت → صفر
    parallel_sig = np.zeros_like(sig_gp)
    for i in range(len(sig_gp)):
        ai_sig = 1 if prob_up[i] > 0.5 else (-1 if prob_down[i] > 0.5 else 0)
        if sig_gp[i] == ai_sig: parallel_sig[i] = sig_gp[i]
    p_pnl, p_w, p_l = evaluate_signals(parallel_sig, long_pnl, short_pnl)

    # ─── ۲. حالت سری (سیگنال GP به عنوان فیلتر اول، سپس Veto) ───
    series_sig = core_2_veto_logic(sig_gp, prob_up, prob_down, X)
    s_pnl, s_w, s_l = evaluate_signals(series_sig, long_pnl, short_pnl)

    # ─── ۳. حالت ترکیبی (اول Veto روی نویز، سپس موازی با AI) ───
    hybrid_sig = np.zeros_like(sig_gp)
    for i in range(len(sig_gp)):
        if abs(X[i, 2]) > 1.5 and X[i, 3] < 0.5:
            continue
        ai_sig = 1 if prob_up[i] > 0.5 else (-1 if prob_down[i] > 0.5 else 0)
        if sig_gp[i] == ai_sig and sig_gp[i] != 0:
            hybrid_sig[i] = sig_gp[i]
    h_pnl, h_w, h_l = evaluate_signals(hybrid_sig, long_pnl, short_pnl)

    # چاپ نتایج
    print("\n═══ نتایج ترکیب هسته‌ها ═══\n")
    
    print("۱. حالت موازی (Parallel):")
    print(f"   معاملات: {p_w+p_l}, برد: {100*p_w/max(p_w+p_l,1):.1f}%, سود: {p_pnl:+.4f}\n")
    
    print("۲. حالت سری (Series):")
    print(f"   معاملات: {s_w+s_l}, برد: {100*s_w/max(s_w+s_l,1):.1f}%, سود: {s_pnl:+.4f}\n")
    
    print("۳. حالت ترکیبی (Hybrid):")
    print(f"   معاملات: {h_w+h_l}, برد: {100*h_w/max(h_w+h_l,1):.1f}%, سود: {h_pnl:+.4f}\n")
    
    best = max([(p_pnl, 'موازی'), (s_pnl, 'سری'), (h_pnl, 'ترکیبی')], key=lambda x: x[0])
    print(f"[★] بهترین ترکیب از نظر سود: {best[1]} با سود {best[0]:+.4f}")
