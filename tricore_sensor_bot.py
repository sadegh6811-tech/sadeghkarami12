import numpy as np
from genetic_engine_classifier import evaluate_tree

# ═══════════════════════════════════════════════
# ۱. آماده‌سازی داده‌ها
# ═══════════════════════════════════════════════
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
    return X, y, long_pnl, short_pnl

# ═══════════════════════════════════════════════
# ۲. هسته ۱ (فرمول GP)
# ═══════════════════════════════════════════════
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

# ═══════════════════════════════════════════════
# ۳. هسته ۳ (هوش مصنوعی - درخت تصمیم‌گیری)
# ═══════════════════════════════════════════════
class SimpleDecisionTree:
    def __init__(self, max_depth=3): self.max_depth = max_depth
    def fit(self, X, y): self.tree = self._build(X, y, 0)
    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.bincount(y.astype(int) + 1).argmax() - 1
        best_gain, best_f, best_t = -1, None, None
        for f in range(X.shape[1]):
            for t in np.unique(X[:, f]):
                left, right = y[X[:, f] <= t], y[X[:, f] > t]
                if len(left) == 0 or len(right) == 0: continue
                def ent(a):
                    if len(a) == 0: return 0
                    p = np.bincount(a.astype(int) + 1) / len(a)
                    return -np.sum(p * np.log2(p + 1e-9))
                gain = ent(y) - (len(left)/len(y))*ent(left) - (len(right)/len(y))*ent(right)
                if gain > best_gain: best_gain, best_f, best_t = gain, f, t
        if best_gain <= 0: return np.bincount(y.astype(int) + 1).argmax() - 1
        m = X[:, best_f] <= best_t
        return (best_f, best_t, self._build(X[m], y[m], depth+1), self._build(X[~m], y[~m], depth+1))
    def predict(self, X): return np.array([self._pred(x, self.tree) for x in X])
    def _pred(self, x, node):
        if not isinstance(node, tuple): return node
        f, t, l, r = node
        return self._pred(x, l) if x[f] <= t else self._pred(x, r)

# ═══════════════════════════════════════════════
# ۴. سنسور اطمینان‌سنجی (Confidence Sensor)
# ═══════════════════════════════════════════════
def confidence_sensor(sig_gp, ai_sig, X, noise_th=1.5, liq_th=0.5):
    """این سنسور بین هسته‌ها قرار می‌گیرد و سیگنال نهایی را فیلتر می‌کند"""
    noise = X[:, 2]  # لرزش ذره
    liq = X[:, 3]    # نقدینگی
    final_sig = np.zeros_like(sig_gp)
    
    for i in range(len(sig_gp)):
        # سنسور ۱: آیا نویز بیش از حد نیست؟ (لرزش ذره)
        if abs(noise[i]) > noise_th: continue
        
        # سنسور ۲: آیا نقدینگی کافی برای حرکت وجود دارد؟
        if liq[i] < liq_th: continue
        
        # سنسور ۳: هم‌گرایی (Consensus) - آیا هر دو هسته هم‌جهت هستند؟
        if sig_gp[i] != 0 and ai_sig[i] == sig_gp[i]:
            final_sig[i] = sig_gp[i]
            
    return final_sig

# ═══════════════════════════════════════════════
# ۵. اجرای نهایی و ارزیابی
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    path = '/storage/emulated/0/forex_bot/golden_points/EURUSD_golden.csv'
    X, y, long_pnl, short_pnl = load_data(path)
    split = int(len(X) * 0.8)
    
    print("[*] راه‌اندازی هسته ۱ (GP)...")
    sig_gp = core_1_gp(X)
    
    print("[*] راه‌اندازی هسته ۳ (AI)...")
    tree = SimpleDecisionTree(max_depth=3)
    tree.fit(X[:split], y[:split])
    ai_sig = tree.predict(X)
    
    print("[*] فعال‌سازی سنسور اطمینان‌سنجی...")
    
    # اجرای سنسور با تنظیمات پیش‌فرض
    final_sig = confidence_sensor(sig_gp, ai_sig, X)
    
    # محاسبه نتیجه
    total_pnl, wins, losses = 0.0, 0, 0
    for i in range(len(final_sig) - 1):
        if final_sig[i] == 1: pnl = long_pnl[i+1]
        elif final_sig[i] == -1: pnl = short_pnl[i+1]
        else: continue
        total_pnl += pnl
        if pnl > 0: wins += 1
        else: losses += 1
    
    print("\n═══ نتیجه ربات با سنسور اطمینان‌سنجی ═══")
    print(f"تعداد معاملات: {wins + losses}")
    print(f"معاملات برنده: {wins} ({100*wins/max(wins+losses,1):.1f}%)")
    print(f"سود/زیان کل: {total_pnl:+.4f}")
