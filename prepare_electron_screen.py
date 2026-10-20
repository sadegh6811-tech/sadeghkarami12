import numpy as np

def prepare(input_path, output_path):
    data = np.loadtxt(input_path, delimiter=',', skiprows=1)
    print(f"[*] داده اصلی: {data.shape}")
    
    close_p = data[:, 5]
    high_p = data[:, 3]
    low_p = data[:, 4]
    open_p = data[:, 2]
    
    # ۱. مختصات پیکسلی (قیمت نرمال‌شده به بازه 0-1000)
    pixel_y = (close_p - np.min(close_p)) / (np.max(close_p) - np.min(close_p)) * 1000
    
    # ۲. تعداد الکترون‌ها (حجم تقریبی از دامنه کندل)
    electron_count = (high_p - low_p) * 1e4 + 1e-6
    
    # ۳. نویز کوانتومی (قانون پواسون: sqrt(N))
    shot_noise = np.sqrt(electron_count)
    
    # ۴. چگالی الکترون (نقدینگی واقعی)
    electron_density = electron_count / (shot_noise + 1e-6)
    
    # ۵. مومنتوم پیکسلی
    pixel_momentum = np.gradient(pixel_y)
    
    # ۶. هدف: فروپاشی بعدی (جهت حرکت پیکسل)
    future_pixel = np.roll(pixel_y, -1)
    target = np.sign(future_pixel - pixel_y)
    target[-1] = 0
    
    def norm(x): 
        return (x - np.mean(x)) / (np.std(x) + 1e-12)
    
    features = np.column_stack([
        norm(pixel_y), 
        norm(shot_noise), 
        norm(electron_density), 
        norm(pixel_momentum),
        norm(electron_count)
    ])
    
    final_data = np.column_stack([features, target])
    header = "pixel_y,shot_noise,electron_density,pixel_momentum,electron_count,target"
    np.savetxt(output_path, final_data, delimiter=',', header=header, comments='')
    
    print(f"[+] داده الکترون-پیکسل ساخته شد: {final_data.shape}")
    print(f"    توزیع هدف: صعودی={np.sum(target==1)}, نزولی={np.sum(target==-1)}, رنج={np.sum(target==0)}")

prepare('/storage/emulated/0/forex_bot/golden_points/EURUSD_golden.csv', '/data/data/com.termux/files/home/forex_electron.csv')
