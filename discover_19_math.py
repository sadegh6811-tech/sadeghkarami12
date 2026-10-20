import numpy as np
from genetic_engine_classifier import evaluate_tree

# ---------- آماده‌سازی داده ----------
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
    target = np.sign(np.roll(c, -1) - c)
    return X, target, long_pnl, short_pnl

# ---------- فرمول GP ----------
BEST_TREE = ('div',
    ('add', ('add', 'x7', 'x2'), ('const', 0.0)),
    ('add', 
        ('add', ('abs', ('mul', ('sub', ('add', 'x6', 'x1'), ('sub', 'x1', 'x0')), ('div', ('div', 'x1', 'x6'), ('abs', 'x0')))),
            ('div', ('const', -1.0), 'x5')
        ),
        ('abs', ('sub', ('abs', 'x2'), ('div', ('const', 0.5), 'x7')))
    )
)

# ---------- درخت تصمیم‌گیری ----------
class SimpleDecisionTree:
    def __init__(self, max_depth=4): self.max_depth = max_depth
    def fit(self, X, y): self.tree = self._build(X, y, 0)
    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.bincount(y.astype(int)).argmax()
        best_gain, best_f, best_t = -1, None, None
        for f in range(X.shape[1]):
            for t in np.unique(X[:, f]):
                left, right = y[X[:, f] <= t], y[X[:, f] > t]
                if len(left) == 0 or len(right) == 0: continue
                def ent(a):
                    if len(a) == 0: return 0
                    p = np.bincount(a.astype(int)) / len(a)
                    return -np.sum(p * np.log2(p + 1e-9))
                gain = ent(y) - (len(left)/len(y))*ent(left) - (len(right)/len(y))*ent(right)
                if gain > best_gain: best_gain, best_f, best_t = gain, f, t
        if best_gain <= 0: return np.bincount(y.astype(int)).argmax()
        m = X[:, best_f] <= best_t
        return (best_f, best_t, self._build(X[m], y[m], depth+1), self._build(X[~m], y[~m], depth+1))
    def predict(self, X): return np.array([self._pred(x, self.tree) for x in X])
    def _pred(self, x, node):
        if not isinstance(node, tuple): return node
        f, t, l, r = node
        return self._pred(x, l) if x[f] <= t else self._pred(x, r)

# ---------- کشف الگوی خطا ----------
if __name__ == "__main__":
    path = '/storage/emulated/0/forex_bot/golden_points/EURUSD_golden.csv'
    X, target, long_pnl, short_pnl = load_data(path)
    
    # ۱. اجرای GP
    pred = evaluate_tree(BEST_TREE, X)
    pred_sig = np.zeros_like(pred)
    pred_sig[pred > 0.1] = 1
    pred_sig[pred < -0.1] = -1
    
    # ۲. برچسب‌گذاری: ۱ = GP درست، ۰ = GP خطا
    mask = (pred_sig != 0) & (target != 0)
    X_valid = X[mask]
    y_valid = (pred_sig[mask] == target[mask]).astype(int)
    
    print(f"[*] کل موارد: {len(y_valid)}")
    print(f"[*] موفق (۸۱٪): {np.sum(y_valid==1)}")
    print(f"[*] خطا (۱۹٪): {np.sum(y_valid==0)}")
    
    # ۳. آموزش درخت تصمیم روی الگوی خطاها (۸۰٪ آموزش، ۲۰٪ تست)
    split = int(len(X_valid) * 0.8)
    tree = SimpleDecisionTree(max_depth=4)
    tree.fit(X_valid[:split], y_valid[:split])
    
    # ۴. ارزیابی درخت تصمیم
    tree_pred_train = tree.predict(X_valid[:split])
    tree_pred_test = tree.predict(X_valid[split:])
    
    train_acc = np.mean(tree_pred_train == y_valid[:split])
    test_acc = np.mean(tree_pred_test == y_valid[split:])
    print(f"\n[+] دقت درخت در کشف موفقیت GP (Train): {train_acc*100:.2f}%")
    print(f"[+] دقت درخت در کشف موفقیت GP (Test): {test_acc*100:.2f}%")
    
    # ۵. فیلتر هوشمند: فقط جایی معامله کن که درخت می‌گوید GP درست می‌گوید
    final_sig = np.zeros_like(pred_sig)
    tree_pred_all = tree.predict(X_valid)
    final_sig[mask] = np.where(tree_pred_all == 1, pred_sig[mask], 0)
    
    # ۶. ارزیابی مالی
    total_pnl, wins, losses = 0.0, 0, 0
    for i in range(len(final_sig) - 1):
        if final_sig[i] == 1: pnl = long_pnl[i+1]
        elif final_sig[i] == -1: pnl = short_pnl[i+1]
        else: continue
        total_pnl += pnl
        if pnl > 0: wins += 1
        else: losses += 1
    
    print(f"\n═══ نتیجه پس از فیلتر هوشمند (کشف ۱۹٪) ═══")
    print(f"معاملات باقی‌مانده: {wins+losses}")
    print(f"درصد برد: {100*wins/max(wins+losses,1):.1f}%")
    print(f"سود کل: {total_pnl:+.4f}")
    
    # مقایسه با بدون فیلتر
    total_pnl_raw, wins_raw, losses_raw = 0.0, 0, 0
    for i in range(len(pred_sig) - 1):
        if pred_sig[i] == 1: pnl = long_pnl[i+1]
        elif pred_sig[i] == -1: pnl = short_pnl[i+1]
        else: continue
        total_pnl_raw += pnl
        if pnl > 0: wins_raw += 1
        else: losses_raw += 1
    
    print(f"\n═══ مقایسه ═══")
    print(f"بدون فیلتر: برد={100*wins_raw/max(wins_raw+losses_raw,1):.1f}%, سود={total_pnl_raw:+.4f}")
    print(f"با فیلتر:   برد={100*wins/max(wins+losses,1):.1f}%, سود={total_pnl:+.4f}")
    print(f"بهبود سود:  {total_pnl - total_pnl_raw:+.4f}")
