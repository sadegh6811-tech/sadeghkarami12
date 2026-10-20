import numpy as np

def simulate_parallel_universes(n_universes=1000, n_steps=100):
    """
    شبیه‌سازی جهان‌های موازی بر اساس مدل تورم کیهانی.
    هر جهان یک مسیر زمانی از نوسانات میدان اسکالر است.
    """
    np.random.seed(42)
    
    # پارامترهای فیزیکی (بر اساس مدل تورم کیهانی)
    H = 1.0  # نرخ هابل
    k_min, k_max = 1e-3, 1e3  # محدوده اعداد موج
    
    universes = []
    for i in range(n_universes):
        # ۱. تولید نوسانات کوانتومی میدان اسکالر
        k = np.logspace(np.log10(k_min), np.log10(k_max), n_steps)
        # دامنه نوسانات: H / (2*pi) * sqrt(k^3 / (2*pi^2))
        amplitude = (H / (2 * np.pi)) * np.sqrt(k**3 / (2 * np.pi**2))
        # فاز تصادفی (نوسانات کوانتومی)
        phase = np.random.uniform(0, 2*np.pi, n_steps)
        # میدان اسکالر
        phi = amplitude * np.sin(phase)
        
        # ۲. تحول زمانی (معادله حرکت میدان اسکالر)
        # d^2(phi)/dt^2 + 3H d(phi)/dt + dV/d(phi) = 0
        # برای سادگی، از یک پتانسیل ساده (V = 0.5 * m^2 * phi^2) استفاده می‌کنیم
        m = 1e-2  # جرم میدان
        dt = 0.01
        phi_dot = np.gradient(phi)
        phi_ddot = -3 * H * phi_dot - m**2 * phi
        
        # ۳. استخراج ویژگی‌های جهان
        universe_data = np.column_stack([phi, phi_dot, phi_ddot])
        universes.append(universe_data)
    
    return universes

# ذخیره‌ی داده‌های جهان‌های موازی
universes = simulate_parallel_universes()
np.save('/data/data/com.termux/files/home/parallel_universes.npy', universes)
print(f"[+] {len(universes)} جهان موازی شبیه‌سازی شد.")
print(f"    شکل هر جهان: {universes[0].shape}")
print(f"    نمونه‌ای از میدان اسکالر: {universes[0][:3, 0]}")
