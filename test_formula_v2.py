import numpy as np, csv, sys, os
sys.path.insert(0, '/data/data/com.termux/files/home/mathbot')

print("=" * 75)
print("BACKTEST: Discovered Formula v2 (no look-ahead)")
print("=" * 75)

# ---- Load ----
H, L, O, C = [], [], [], []
with open('/storage/emulated/0/forex_bot/golden_points/AUDCAD_golden.csv') as f:
    for r in csv.DictReader(f):
        try:
            h,l,o,c = float(r['high']),float(r['low']),float(r['open']),float(r['close'])
            if h > l and h > 0 and l > 0:
                H.append(h); L.append(l); O.append(o); C.append(c)
        except (ValueError, KeyError): pass

H,L,O,C = map(np.array, [H,L,O,C])
n = len(H)
print(f"Total candles: {n}")

# ---- Features (raw, no z-score yet) ----
den = H - L
wick_raw = (H + L - O - C) / den
cpos_raw = (C - L) / den
range_pct_raw = den / C

# ---- Rolling Z-score (NO LOOK-AHEAD) ----
def rolling_zscore(x, window=100):
    n = len(x)
    z = np.zeros(n)
    for i in range(window, n):
        w = x[i-window:i]
        m = w.mean()
        s = w.std()
        if s > 1e-10:
            z[i] = (x[i] - m) / s
    return z

WINDOW = 100
wick_z = rolling_zscore(wick_raw, WINDOW)
cpos_z = rolling_zscore(cpos_raw, WINDOW)
range_pct_z = rolling_zscore(range_pct_raw, WINDOW)

print(f"Rolling z-score with window={WINDOW}")
print(f"  wick_z non-zero: {np.sum(wick_z != 0)}")

# ---- FORMULA (eq[3]) ----
signal = (0.23*wick_z - 0.24*cpos_z + 0.39*range_pct_z 
          - 0.16*wick_z*cpos_z)

# ---- Future return (FIX 3: no shift) ----
HORIZON = 5
future_ret = np.zeros(n)
for i in range(n - HORIZON):
    if C[i] > 0:
        future_ret[i] = (C[i+HORIZON] - C[i]) / C[i]
# future_ret در آخر NaN دارد (چون C[n-HORIZON+i] وجود ندارد)
# فقط بخش معتبر

valid_len = n - HORIZON
sig_v = signal[:valid_len]
ret_v = future_ret[:valid_len]

# حذف nan
mask = np.isfinite(sig_v) & np.isfinite(ret_v) & (wick_z[:valid_len] != 0)
sig_v = sig_v[mask]
ret_v = ret_v[mask]

print(f"\nValid samples: {len(sig_v)}")

# ---- Correlation ----
corr = np.corrcoef(sig_v, ret_v)[0,1]
print(f"\nCorrelation(signal, future_ret) = {corr:+.4f}")

# ---- Compare with hand-crafted (rolling-z) ----
hand_signal = np.tanh(0.5 * wick_raw[:valid_len])
hand_signal = hand_signal[mask]
hand_corr = np.corrcoef(hand_signal, ret_v)[0,1]
print(f"Correlation(tanh(0.5*wick), future_ret) = {hand_corr:+.4f}")

# ---- Also test raw wick ----
raw_wick_v = wick_raw[:valid_len][mask]
raw_corr = np.corrcoef(raw_wick_v, ret_v)[0,1]
print(f"Correlation(raw wick, future_ret) = {raw_corr:+.4f}")

# ---- Backtest with quantile threshold ----
FEE = 0.0004
COOLDOWN = 5
TP = 0.01
SL = 0.01
HOLD = 5

# آستانه بر اساس percentile
th_hi = np.percentile(sig_v, 85)
th_lo = np.percentile(sig_v, 15)
print(f"\nThresholds: top 15% = {th_hi:+.4f}, bottom 15% = {th_lo:+.4f}")

# ---- backtest فقط از محدوده ی rolling-z ----
trades = []
daily_pnl = {}
last_idx = -COOLDOWN - 1

for i in range(WINDOW + 5, n - HORIZON - 1):
    s = signal[i]
    if s == 0: continue
    
    if s > th_hi: pos = 1
    elif s < th_lo: pos = -1
    else: continue
    
    if i - last_idx <= COOLDOWN: continue
    
    entry = C[i]
    exit_p = C[min(i+HOLD, n-1)]
    
    for j in range(i+1, min(i+HOLD+1, n)):
        if pos == 1:
            if L[j] <= entry*(1-SL): exit_p = entry*(1-SL); break
            if H[j] >= entry*(1+TP): exit_p = entry*(1+TP); break
        else:
            if H[j] >= entry*(1+SL): exit_p = entry*(1+SL); break
            if L[j] <= entry*(1-TP): exit_p = entry*(1-TP); break
    
    pnl = (exit_p - entry)/entry * pos - FEE
    trades.append(pnl)
    last_idx = i

if trades:
    pnls = np.array(trades)
    print(f"\n" + "=" * 75)
    print("BACKTEST RESULTS")
    print("=" * 75)
    print(f"  Trades:       {len(pnls)}")
    print(f"  Win Rate:     {(pnls > 0).mean()*100:.2f}%")
    print(f"  Mean PnL:     {pnls.mean()*100:+.4f}%")
    print(f"  Total Return: {pnls.sum()*100:+.2f}%")
    
    # Sharpe تقریبی
    if len(pnls) > 5:
        m = pnls.mean()
        s = pnls.std()
        if s > 1e-10:
            sharpe_t = m/s
            print(f"  Sharpe/trade: {sharpe_t:+.4f}")
else:
    print("\nNo trades.")

print("\n" + "=" * 75)
print("NOTE: rolling_zscore prevents look-ahead bias")
print("=" * 75)
