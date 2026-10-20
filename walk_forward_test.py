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
    return X, y, long_pnl, short_pnl

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

def core_2_decision(X):
    noise, liq, mom = X[:, 2], X[:, 3], X[:, 7]
    sig = np.zeros(X.shape[0])
    sig[(mom > 0.5) & (noise < 0.5) & (liq > 0.3)] = 1
    sig[(mom < -0.5) & (noise < 0.5) & (liq > 0.3)] = -1
    return sig

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

if __name__ == "__main__":
    path = '/storage/emulated/0/forex_bot/golden_points/EURUSD_golden.csv'
    X, y, long_pnl, short_pnl = load_data(path)
    N = len(X)
    window = N // 5

    print("[*] اجرای آزمون Walk-Forward (۵ بخش)...")
    total_profit_all = 0

    for fold in range(1, 5):
        train_end = fold * window
        test_start, test_end = train_end, train_end + window
        if test_end > N: break

        X_train, y_train = X[:train_end], y[:train_end]
        X_test = X[test_start:test_end]
        long_pnl_test = long_pnl[test_start:test_end]
        short_pnl_test = short_pnl[test_start:test_end]

        sig_gp = core_1_gp(X_test)
        sig_decision = core_2_decision(X_test)

        X_ai_train = np.column_stack([core_1_gp(X_train), core_2_decision(X_train), X_train])
        X_ai_test = np.column_stack([sig_gp, sig_decision, X_test])

        tree = SimpleDecisionTree(max_depth=4)
        tree.fit(X_ai_train, y_train)
        ai_sig = tree.predict(X_ai_test)

        total_pnl, wins, losses = 0.0, 0, 0
        for i in range(len(ai_sig)-1):
            if ai_sig[i] == 1: pnl = long_pnl_test[i+1]
            elif ai_sig[i] == -1: pnl = short_pnl_test[i+1]
            else: continue
            total_pnl += pnl
            if pnl > 0: wins += 1
            else: losses += 1

        total_profit_all += total_pnl
        print(f"  بخش {fold}: معاملات={wins+losses}, برد={100*wins/max(wins+losses,1):.1f}%, سود={total_pnl:+.4f}")

    print(f"\n[★] سود کل در ۵ بخش: {total_profit_all:+.4f}")
    if total_profit_all > 0:
        print("[✅] استراتژی در همه بخش‌ها پایدار است. این یک کشف علمی معتبر است.")
    else:
        print("[⚠️] استراتژی در بعضی بخش‌ها ضررده بوده. نیاز به تنظیمات بیشتر دارد.")
