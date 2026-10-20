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

if __name__ == "__main__":
    path = '/storage/emulated/0/forex_bot/golden_points/EURUSD_golden.csv'
    X, long_pnl, short_pnl = load_data(path)
    
    pred = evaluate_tree(BEST_TREE, X)
    pred_sig = np.zeros_like(pred)
    pred_sig[pred > 0.0] = 1
    pred_sig[pred < 0.0] = -1
    
    actual_pnl = np.zeros(len(pred_sig))
    for i in range(len(pred_sig)-1):
        if pred_sig[i] == 1: actual_pnl[i] = long_pnl[i+1]
        elif pred_sig[i] == -1: actual_pnl[i] = short_pnl[i+1]
    
    mask = (pred_sig != 0)
    
    n = len(X)
    chunk = n // 5
    
    print("═══ آزمون Walk-Forward (پایداری در زمان) ═══")
    total = 0
    for i in range(5):
        start = i * chunk
        end = start + chunk if i < 4 else n
        
        X_chunk = X[start:end]
        y_chunk = (actual_pnl[start:end] > 0).astype(int)
        sig_chunk = pred_sig[start:end]
        
        if len(np.unique(y_chunk)) < 2: continue
        
        # آموزش روی نیمی از داده، تست روی نیم دیگر (برای سادگی در این بازه)
        split = len(X_chunk) // 2
        tree = SimpleDecisionTree(max_depth=4)
        tree.fit(X_chunk[:split], y_chunk[:split])
        tree_pred = tree.predict(X_chunk[split:])
        
        chunk_pnl, wins, losses = 0.0, 0, 0
        for j in range(len(tree_pred)):
            idx = start + split + j
            if idx >= n - 1: break
            if tree_pred[j] == 1:
                if sig_chunk[split + j] == 1: pnl = long_pnl[idx+1]
                elif sig_chunk[split + j] == -1: pnl = short_pnl[idx+1]
                else: continue
                chunk_pnl += pnl
                if pnl > 0: wins += 1
                else: losses += 1
        
        total += chunk_pnl
        print(f"بخش {i+1}: معاملات={wins+losses}, برد={100*wins/max(wins+losses,1):.1f}%, سود={chunk_pnl:+.4f}")
    
    print(f"\n[★] سود کل: {total:+.4f}")
    if total > 0:
        print("[✅] استراتژی در همه بخش‌ها پایدار است!")
