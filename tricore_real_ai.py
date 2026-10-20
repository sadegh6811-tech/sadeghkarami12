import numpy as np
from genetic_engine_classifier import evaluate_tree

# ---------- داده ----------
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

# ---------- درخت تصمیم‌گیری سبک (هسته ۳) ----------
class SimpleDecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self._build(X, y, 0)
    
    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.bincount(y.astype(int) + 1).argmax() - 1
        
        best_gain, best_feat, best_thresh = -1, None, None
        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            for th in thresholds:
                left = y[X[:, feat] <= th]
                right = y[X[:, feat] > th]
                if len(left) == 0 or len(right) == 0: continue
                
                def entropy(arr):
                    if len(arr) == 0: return 0
                    p = np.bincount(arr.astype(int) + 1) / len(arr)
                    return -np.sum(p * np.log2(p + 1e-9))
                
                gain = entropy(y) - (len(left)/len(y))*entropy(left) - (len(right)/len(y))*entropy(right)
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, th
        
        if best_gain <= 0:
            return np.bincount(y.astype(int) + 1).argmax() - 1
        
        left_mask = X[:, best_feat] <= best_thresh
        return (best_feat, best_thresh,
                self._build(X[left_mask], y[left_mask], depth+1),
                self._build(X[~left_mask], y[~left_mask], depth+1))
    
    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])
    
    def _predict_one(self, x, node):
        if not isinstance(node, tuple): return node
        feat, th, left, right = node
        if x[feat] <= th: return self._predict_one(x, left)
        return self._predict_one(x, right)

# ---------- هسته ۱ (فرمول GP) ----------
BEST_TREE = ('div',
    ('add', ('add', 'x7', 'x2'), ('const', 0.0)),
    ('add', 
        ('add', 
            ('abs', 
                ('mul', 
                    ('sub', ('add', 'x6', 'x1'), ('sub', 'x1', 'x0')),
                    ('div', ('div', 'x1', 'x6'), ('abs', 'x0'))
                )
            ),
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

# ---------- اجرا ----------
if __name__ == "__main__":
    path = '/storage/emulated/0/forex_bot/golden_points/EURUSD_golden.csv'
    X, y, long_pnl, short_pnl = load_data(path)
    split = int(len(X) * 0.8)
    
    print("[*] آموزش درخت تصمیم‌گیری (هسته ۳)...")
    tree = SimpleDecisionTree(max_depth=4)
    tree.fit(X[:split], y[:split])
    ai_sig = tree.predict(X)
    print("[+] هسته ۳ آماده شد.")
    
    sig_gp = core_1_gp(X)
    
    # --- ترکیب سری: هر دو باید هم‌جهت باشند ---
    final = np.zeros_like(sig_gp)
    for i in range(len(sig_gp)):
        if sig_gp[i] != 0 and ai_sig[i] == sig_gp[i]:
            final[i] = sig_gp[i]
    
    total_pnl, wins, losses = 0.0, 0, 0
    for i in range(len(final) - 1):
        if final[i] == 1: pnl = long_pnl[i+1]
        elif final[i] == -1: pnl = short_pnl[i+1]
        else: continue
        total_pnl += pnl
        if pnl > 0: wins += 1
        else: losses += 1
    
    print(f"\n═══ نتیجه ربات با AI واقعی (درخت تصمیم‌گیری) ═══")
    print(f"معاملات: {wins+losses}, برد: {100*wins/max(wins+losses,1):.1f}%")
    print(f"سود کل: {total_pnl:+.4f}")
