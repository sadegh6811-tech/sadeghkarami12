import numpy as np

def resample_to_lower_timeframe(open_p, high_p, low_p, close_p, factor):
    """
    تبدیل داده‌های ۱ ساعته به ۵ یا ۱۵ دقیقه (با شبیه‌سازی فراکتالی)
    factor: تعداد کندل‌های جدید در هر کندل قدیم (مثلاً ۱۲ برای ۵ دقیقه، ۴ برای ۱۵ دقیقه)
    """
    n = len(close_p)
    new_n = n * factor
    
    new_open = np.zeros(new_n)
    new_high = np.zeros(new_n)
    new_low = np.zeros(new_n)
    new_close = np.zeros(new_n)
    
    for i in range(n):
        # ساخت sub-candle ها
        sub_prices = np.linspace(open_p[i], close_p[i], factor + 1)
        sub_noise = np.random.normal(0, (high_p[i] - low_p[i]) / factor, factor + 1)
        sub_prices += sub_noise
        
        for j in range(factor):
            idx = i * factor + j
            new_open[idx] = sub_prices[j]
            new_close[idx] = sub_prices[j+1]
            new_high[idx] = max(sub_prices[j], sub_prices[j+1]) + abs(np.random.normal(0, 0.0001))
            new_low[idx] = min(sub_prices[j], sub_prices[j+1]) - abs(np.random.normal(0, 0.0001))
    
    return new_open, new_high, new_low, new_close

def build_features_and_target(o, h, l, c):
    """ساخت ویژگی‌های مدل دوگانه"""
    n = len(c)
    range_p = h - l
    body = c - o
    
    # ویژگی‌های پایه
    x0 = (c - np.mean(c)) / (np.std(c) + 1e-9)
    x1 = np.sign(body)
    x2 = ((h - np.maximum(o, c)) - (np.minimum(o, c) - l)) / (range_p + 1e-9)
    x3 = range_p / (np.std(range_p) + 1e-9)
    x4 = np.abs(body) / (np.std(body) + 1e-9)
    x5 = (body > 0).astype(float)
    x6 = (body < 0).astype(float)
    x7 = np.gradient(c) / (np.std(np.gradient(c)) + 1e-9)
    x8 = np.linspace(0, 2*np.pi, n)
    
    X = np.column_stack([x0, x1, x2, x3, x4, x5, x6, x7, x8])
    
    # هدف: سیگنال PnL
    next_c = np.roll(c, -1)
    next_c[-1] = c[-1]
    next_pnl_long = (next_c - c) / c
    next_pnl_short = (c - next_c) / c
    
    return X, next_pnl_long, next_pnl_short

# ═══════════════════════════════════════════════════════════
# اجرا
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ۱. بارگذاری داده‌های اصلی
    data = np.loadtxt('/storage/emulated/0/forex_bot/golden_points/EURUSD_golden.csv', delimiter=',', skiprows=1)
    o, h, l, c = data[:,2], data[:,3], data[:,4], data[:,5]
    print(f"[*] داده اصلی: {len(c)} کندل")
    
    # ۲. ساخت داده‌های ۵ دقیقه
    print("[*] شبیه‌سازی تایم‌فریم ۵ دقیقه (فاکتور ۱۲)...")
    o5, h5, l5, c5 = resample_to_lower_timeframe(o, h, l, c, factor=12)
    X5, yl5, ys5 = build_features_and_target(o5, h5, l5, c5)
    print(f"    {len(c5)} کندل ۵ دقیقه‌ای ساخته شد.")
    
    # ۳. ساخت داده‌های ۱۵ دقیقه
    print("[*] شبیه‌سازی تایم‌فریم ۱۵ دقیقه (فاکتور ۴)...")
    o15, h15, l15, c15 = resample_to_lower_timeframe(o, h, l, c, factor=4)
    X15, yl15, ys15 = build_features_and_target(o15, h15, l15, c15)
    print(f"    {len(c15)} کندل ۱۵ دقیقه‌ای ساخته شد.")
    
    # ۴. ذخیره
    np.save('/data/data/com.termux/files/home/timeframe_5m.npy', (X5, yl5, ys5))
    np.save('/data/data/com.termux/files/home/timeframe_15m.npy', (X15, yl15, ys15))
    print("[+] داده‌ها ذخیره شدند.")
