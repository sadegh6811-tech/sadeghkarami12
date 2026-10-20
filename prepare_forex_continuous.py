"""آماده‌سازی داده‌های فارکس با هدف پیوسته (بازده)"""
import numpy as np
import sys

def prepare_data(input_path, output_path):
    data = np.loadtxt(input_path, delimiter=',', skiprows=1)
    print(f"[*] داده اصلی: {data.shape}")
    
    open_p = data[:, 2]
    high_p = data[:, 3]
    low_p = data[:, 4]
    close_p = data[:, 5]
    
    # ویژگی‌های تکنیکال
    body = close_p - open_p
    upper_wick = high_p - np.maximum(open_p, close_p)
    lower_wick = np.minimum(open_p, close_p) - low_p
    range_p = high_p - low_p
    log_return = np.log(close_p / np.roll(close_p, 1) + 1e-12)
    log_return[0] = 0
    
    # هدف پیوسته: بازده کندل بعدی
    next_log_return = np.roll(log_return, -1)
    next_log_return[-1] = 0
    
    # نرمال‌سازی
    def normalize(x):
        return (x - np.mean(x)) / (np.std(x) + 1e-12)
    
    features = np.column_stack([
        normalize(body),
        normalize(upper_wick),
        normalize(lower_wick),
        normalize(range_p),
        normalize(log_return)
    ])
    
    target = normalize(next_log_return)
    
    final_data = np.column_stack([features, target])
    header = "body,upper_wick,lower_wick,range,log_return,next_return"
    np.savetxt(output_path, final_data, delimiter=',', header=header, comments='')
    
    print(f"[+] داده‌ی پیوسته آماده شد: {final_data.shape}")
    print(f"    هدف (بازده پیوسته) میانگین: {np.mean(target):.4f}, انحراف معیار: {np.std(target):.4f}")

if __name__ == "__main__":
    prepare_data(sys.argv[1], sys.argv[2])
