import numpy as np
import sys

def prepare(input_path, output_path):
    data = np.loadtxt(input_path, delimiter=',', skiprows=1)
    print(f"[*] داده اصلی: {data.shape}")
    
    o, h, l, c = data[:,2], data[:,3], data[:,4], data[:,5]
    long_gold = data[:,6]
    short_gold = data[:,7]
    
    # ویژگی‌های پایه
    body = c - o
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    range_p = h - l
    log_return = np.log(c / np.roll(c, 1) + 1e-12)
    log_return[0] = 0
    
    # ویژگی‌های جدید (نگاه از درون سیستم)
    order_flow = long_gold - short_gold
    liquidity = range_p / (np.abs(body) + 1e-6)
    buyer_power = long_gold / (long_gold + short_gold + 1e-6)
    
    # هدف: بازده پیوسته‌ی کندل بعدی
    next_return = np.roll(log_return, -1)
    next_return[-1] = 0
    
    def norm(x): return (x - np.mean(x)) / (np.std(x) + 1e-12)
    
    features = np.column_stack([
        norm(body), norm(upper_wick), norm(lower_wick), 
        norm(range_p), norm(log_return),
        norm(order_flow), norm(liquidity), norm(buyer_power)
    ])
    target = norm(next_return)
    
    final = np.column_stack([features, target])
    header = "body,upper_wick,lower_wick,range,log_return,order_flow,liquidity,buyer_power,next_return"
    np.savetxt(output_path, final, delimiter=',', header=header, comments='')
    print(f"[+] داده‌ی OrderFlow ساخته شد: {final.shape}")
    print(f"    ویژگی‌های جدید: order_flow, liquidity, buyer_power")

if __name__ == "__main__":
    prepare('/storage/emulated/0/forex_bot/golden_points/EURUSD_golden.csv', '/data/data/com.termux/files/home/forex_orderflow.csv')
