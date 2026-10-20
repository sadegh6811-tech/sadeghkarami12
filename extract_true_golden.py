"""استخراج نقاط طلایی واقعی + تقسیم به آموزش/تست"""
import numpy as np
import sys

def extract_and_prepare(input_path, output_path_train, output_path_test):
    print(f"[*] خواندن داده از {input_path}...")
    data = np.loadtxt(input_path, delimiter=',', skiprows=1)
    print(f"[*] داده اصلی: {data.shape}")
    
    open_p = data[:, 2]
    high_p = data[:, 3]
    low_p = data[:, 4]
    close_p = data[:, 5]
    long_pnl = data[:, 8]
    
    # ویژگی‌های تکنیکال
    body = close_p - open_p
    upper_wick = high_p - np.maximum(open_p, close_p)
    lower_wick = np.minimum(open_p, close_p) - low_p
    range_p = high_p - low_p
    log_return = np.log(close_p / np.roll(close_p, 1) + 1e-12)
    log_return[0] = 0
    
    # فیلتر نقاط طلایی: سود > 0.3٪
    mask = long_pnl > 0.003
    n_golden = np.sum(mask)
    print(f"[*] نقاط طلایی (سود > 0.3٪): {n_golden} از {len(data)}")
    
    if n_golden < 100:
        print("[!] نقاط طلایی کافی نیست.")
        return
    
    # ویژگی‌ها و هدف (همه ۱ هستند چون نقاط طلایی هستند)
    X_golden = np.column_stack([
        body[mask], upper_wick[mask], lower_wick[mask], range_p[mask], log_return[mask]
    ])
    y_golden = np.ones(n_golden)  # همه نقاط طلایی، هدف ۱
    
    # نرمال‌سازی
    def normalize(x):
        return (x - np.mean(x)) / (np.std(x) + 1e-12)
    
    for i in range(X_golden.shape[1]):
        X_golden[:, i] = normalize(X_golden[:, i])
    
    # تقسیم به آموزش/تست (۸۰/۲۰)
    n_train = int(n_golden * 0.8)
    X_train, X_test = X_golden[:n_train], X_golden[n_train:]
    y_train, y_test = y_golden[:n_train], y_golden[n_train:]
    
    # ذخیره
    header = "body,upper_wick,lower_wick,range,log_return,target"
    np.savetxt(output_path_train, np.column_stack([X_train, y_train]), 
               delimiter=',', header=header, comments='')
    np.savetxt(output_path_test, np.column_stack([X_test, y_test]), 
               delimiter=',', header=header, comments='')
    
    print(f"[+] آموزش: {X_train.shape}, تست: {X_test.shape}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python extract_true_golden.py <input.csv> <train.csv> <test.csv>")
        sys.exit(1)
    extract_and_prepare(sys.argv[1], sys.argv[2], sys.argv[3])
