import numpy as np
import sys
import time
import warnings
warnings.filterwarnings("ignore")
import scipy.special as sp_spec

sys.path.insert(0, "/data/data/com.termux/files/home/mathbot")
from final_engine_v8_clean import final_engine

ULTIMATE_MATH_V2 = [
    ("۱. آرک‌سینوس (arcsin)", lambda X: np.arcsin(X[:,0]), 1),
    ("۲. آرک‌تانژانت (arctan)", lambda X: np.arctan(X[:,0]), 1),
    ("۳. آرک‌کسینوس هذلولی (arccosh)", lambda X: np.arccosh(X[:,0] + 1e-9), 1),
    ("۴. تابع خطای مکمل (erfc)", lambda X: sp_spec.erfc(X[:,0]), 1),
    ("۵. انتگرال نمایی (Ei)", lambda X: sp_spec.expi(X[:,0] + 1e-9), 1),
    ("۶. لگاریتم انتگرالی (Li)", lambda X: sp_spec.expi(np.log(X[:,0] + 1e-9)), 1),
    ("۷. بسل نوع دوم (Y0)", lambda X: sp_spec.y0(X[:,0] + 1e-9), 1),
    ("۸. تابع هوایی (Airy Ai)", lambda X: sp_spec.airy(X[:,0])[0], 1),
]

print("=" * 85)
print("  چالش نهایی ۲: توابع معکوس و خاص پیشرفته ریاضی")
print("=" * 85)

n_samples = 300
success_count = 0

for i, (name, func, n_vars) in enumerate(ULTIMATE_MATH_V2):
    np.random.seed(300 + i)
    
    # بازه داده‌ها باید در دامنه‌ی مجاز هر تابع باشد
    if "arcsin" in name:
        X = np.random.uniform(-0.99, 0.99, (n_samples, n_vars))
    elif "arccosh" in name:
        X = np.random.uniform(1.01, 3.0, (n_samples, n_vars))
    elif "Li" in name:
        X = np.random.uniform(1.1, 3.0, (n_samples, n_vars))
    else:
        X = np.random.uniform(0.1, 2.0, (n_samples, n_vars))
        
    try:
        y = func(X)
        mask = np.isfinite(y)
        X, y = X[mask], y[mask]
        if len(y) < 50: continue
    except Exception:
        print(f"[-] {name}: خطا در داده‌سازی.")
        continue
        
    print(f"[*] کشف فرمول برای: {name} ...", end="", flush=True)
    start = time.time()
    
    try:
        result = final_engine(X, y)
        elapsed = time.time() - start
        best_r2 = -np.inf
        if isinstance(result, dict):
            best_r2 = result.get('r2', -np.inf)
        elif isinstance(result, (tuple, list)):
            for item in result:
                if isinstance(item, (int, float)) and -1.5 < item <= 1.0:
                    best_r2 = item
                    break
        elif isinstance(result, (int, float)):
            best_r2 = result
            
        if best_r2 > 0.9:
            success_count += 1
            status = "✅ عالی"
        elif best_r2 > 0.7:
            status = "🟡 خوب"
        else:
            status = "❌ ضعیف"
        print(f" {status} (R² = {best_r2:.4f}) در {elapsed:.1f} ثانیه")
    except Exception as e:
        print(f" ❌ خطا: {str(e)[:40]}")

print()
print("=" * 85)
print(f"  نتیجه نهایی: {success_count} از {len(ULTIMATE_MATH_V2)} فرمول کشف شدند.")
print("=" * 85)
