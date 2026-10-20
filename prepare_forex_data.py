import numpy as np
import sys

def prepare_data(input_path, output_path):
    print(f"[*] خواندن داده از {input_path}...")
    data = np.loadtxt(input_path, delimiter=',', skiprows=1)
    print(f"[*] شکل داده: {data.shape}")
    
    open_p = data[:, 2]
    high_p = data[:, 3]
    low_p = data[:, 4]
    close_p = data[:, 5]
    
    body = close_p - open_p
    upper_wick = high_p - np.maximum(open_p, close_p)
    lower_wick = np.minimum(open_p, close_p) - low_p
    range_p = high_p - low_p
    log_return = np.log(close_p / np.roll(close_p, 1) + 1e-12)
    log_return[0] = 0
    
    target = data[:, 6]
    
    features = np.column_stack([body, upper_wick, lower_wick, range_p, log_return])
    final_data = np.column_stack([features, target])
    
    header = "body,upper_wick,lower_wick,range,log_return,target"
    np.savetxt(output_path, final_data, delimiter=',', header=header, comments='')
    
    print(f"[+] داده‌ی آماده‌شده: {final_data.shape}")
    print(f"    توزیع هدف: 0={int(np.sum(target==0))}, 1={int(np.sum(target==1))}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(1)
    prepare_data(sys.argv[1], sys.argv[2])
