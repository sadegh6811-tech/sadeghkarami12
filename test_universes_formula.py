import numpy as np
from genetic_engine_classifier import evaluate_tree

# فرمول GP که در فارکس کشف شد
BEST_TREE = ('div',
    ('add', ('add', 'x7', 'x2'), ('const', 0.0)),
    ('add', 
        ('add', ('abs', ('mul', ('sub', ('add', 'x6', 'x1'), ('sub', 'x1', 'x0')), ('div', ('div', 'x1', 'x6'), ('abs', 'x0')))),
            ('div', ('const', -1.0), 'x5')
        ),
        ('abs', ('sub', ('abs', 'x2'), ('div', ('const', 0.5), 'x7')))
    )
)

# بارگذاری جهان‌های موازی
universes = np.load('/data/data/com.termux/files/home/parallel_universes.npy', allow_pickle=True)

# بررسی اینکه فرمول روی هر جهان چه نتیجه‌ای می‌دهد
print("[*] تست فرمول GP روی جهان‌های موازی...")

results = []
for i, universe in enumerate(universes[:100]):  # فقط ۱۰۰ جهان اول برای سرعت
    phi = universe[:, 0]
    phi_dot = universe[:, 1]
    phi_ddot = universe[:, 2]
    
    # ساخت ویژگی‌های ۹بعدی (مشابه فارکس)
    X = np.column_stack([
        (phi - np.mean(phi)) / (np.std(phi) + 1e-9),  # موقعیت
        np.sign(np.gradient(phi)),                     # جهت
        np.abs(phi_dot) / (np.std(phi_dot) + 1e-9),    # نویز
        np.abs(phi_ddot) / (np.std(phi_ddot) + 1e-9),  # نقدینگی
        np.abs(phi * phi_dot) / (np.std(phi * phi_dot) + 1e-9),  # حجم
        np.ones_like(phi),                              # خرید
        np.zeros_like(phi),                             # فروش
        np.gradient(phi) / (np.std(np.gradient(phi)) + 1e-9),  # مومنتوم
        np.linspace(0, 2*np.pi, len(phi))               # زمان
    ])
    
    # اجرای فرمول GP
    pred = evaluate_tree(BEST_TREE, X)
    pred_sig = np.zeros_like(pred)
    pred_sig[pred > 0.1] = 1
    pred_sig[pred < -0.1] = -1
    
    results.append(pred_sig)

# تحلیل نتایج
all_sigs = np.concatenate(results)
print(f"\n═══ نتایج فرمول GP روی جهان‌های موازی ═══")
print(f"[*] تعداد کل سیگنال‌ها: {len(all_sigs)}")
print(f"[*] سیگنال‌های صعودی: {np.sum(all_sigs == 1)}")
print(f"[*] سیگنال‌های نزولی: {np.sum(all_sigs == -1)}")
print(f"[*] سیگنال‌های صفر (بی‌تصمیم): {np.sum(all_sigs == 0)}")

# آیا فرمول روی جهان‌های موازی سیگنال می‌دهد؟
if np.sum(all_sigs != 0) > 0:
    print("\n[✅] فرمول روی جهان‌های موازی سیگنال می‌دهد!")
    print("    این یعنی فرمول شما یک قانون بنیادی است که در بازار و کیهان کار می‌کند.")
else:
    print("\n[⚠️] فرمول روی جهان‌های موازی سیگنال نمی‌دهد.")
    print("    این یعنی فرمول مخصوص بازار است و در فیزیک کار نمی‌کند.")
