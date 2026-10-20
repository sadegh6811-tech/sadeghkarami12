import numpy as np

def prepare_hypercube(input_path, output_path):
    data = np.loadtxt(input_path, delimiter=',', skiprows=1)
    print(f"[*] داده خام فارکس: {data.shape}")
    
    o, h, l, c = data[:,2], data[:,3], data[:,4], data[:,5]
    long_g, short_g = data[:,6], data[:,7]
    long_pnl, short_pnl = data[:,8], data[:,9]
    
    # ۱. موقعیت روی یال (فاصله قیمت از میانگین)
    close_mean = np.mean(c)
    dim1_pos = (c - close_mean) / (np.std(c) + 1e-9)
    
    # ۲. جهت حرکت (یال صعودی +1، نزولی -1)
    dim2_dir = np.sign(c - o)
    
    # ۳. نویز (لرزش ذره)
    dim3_noise = ((h - np.maximum(o, c)) - (np.minimum(o, c) - l)) / (h - l + 1e-9)
    
    # ۴. نقدینگی (نیروی محرکه)
    dim4_liq = (h - l) / (np.std(h - l) + 1e-9)
    
    # ۵. حجم معاملات (الکترون‌ها)
    dim5_vol = np.abs(long_pnl + short_pnl) / (np.std(long_pnl + short_pnl) + 1e-9)
    
    # ۶ و ۷. تعادل خریدار/فروشنده
    dim6_buy = long_g
    dim7_sell = short_g
    
    # ۸. مومنتوم (سرعت حرکت)
    dim8_mom = np.gradient(c) / (np.std(np.gradient(c)) + 1e-9)
    
    # ۹. زمان (فاز چرخه)
    dim9_time = np.linspace(0, 2*np.pi, len(c))
    
    features = np.column_stack([
        dim1_pos, dim2_dir, dim3_noise, dim4_liq, 
        dim5_vol, dim6_buy, dim7_sell, dim8_mom, dim9_time
    ])
    
    # هدف: حرکت ذره به راس بعدی
    future_c = np.roll(c, -1)
    target = np.sign(future_c - c)
    target[-1] = 0
    
    final_data = np.column_stack([features, target])
    header = "pos,dir,noise,liq,vol,buy,sell,mom,time,target_edge"
    np.savetxt(output_path, final_data, delimiter=',', header=header, comments='')
    
    print(f"[+] داده‌ی توپولوژیک ساخته شد: {final_data.shape}")
    print(f"    توزیع هدف: صعودی={np.sum(target==1)}, نزولی={np.sum(target==-1)}, رنج={np.sum(target==0)}")

prepare_hypercube('/storage/emulated/0/forex_bot/golden_points/EURUSD_golden.csv', '/data/data/com.termux/files/home/forex_topology.csv')
