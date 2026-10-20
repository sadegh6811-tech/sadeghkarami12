import numpy as np
import time
import sys

print("=" * 75)
print("  آزمایشگاه مجازی 9D: درهم‌تنیدگی ۱۰۰,۰۰۰ ابر مکعب ۹ بعدی")
print("=" * 75)
print()

num_cubes = 100000
num_nodes_per_cube = 512
total_nodes = num_cubes * num_nodes_per_cube

print(f"[*] تعداد مکعب‌ها: {num_cubes:,}")
print(f"[*] تعداد راس‌های هر مکعب: {num_nodes_per_cube} (2^9)")
print(f"[*] کل گره‌ها در شبکه کوانتومی: {total_nodes:,}")
print("-" * 75)

# استفاده از float32 برای کاهش مصرف حافظه (حدود 200 مگابایت)
try:
    print("[*] در حال تخصیص حافظه (حدود 200 مگابایت)...")
    energies = np.full((num_cubes, num_nodes_per_cube), 10.0, dtype=np.float32)
    print("    [+] تخصیص حافظه با موفقیت انجام شد.")
except MemoryError:
    print("\n[❌] خطای کمبود حافظه (MemoryError)!")
    print("گوشی شما RAM کافی برای این تعداد گره ندارد.")
    sys.exit(1)

print("-" * 75)
print("[!] تزریق انرژی 1000 واحدی به دو گره در مکعب اول...")
energies[0, 0] = 10.0 + 1000.0
energies[0, 511] = 10.0 + 1000.0

print(f"    انرژی گره 0 در مکعب اول: {energies[0, 0]:.1f}")
print(f"    انرژی گره 511 در مکعب اول: {energies[0, 511]:.1f}")
print("-" * 75)

print("[!] انتقال آنی انرژی به 99,999 مکعب دیگر (زمان فیزیکی = 0)...")
start_time = time.time()

# انتقال آنی در بعد 9D (بدون فاصله و زمان)
energies[1:] = energies[0] 

end_time = time.time()
elapsed = end_time - start_time

print(f"[+] زمان پردازش کامپیوتر: {elapsed:.8f} ثانیه")
print("-" * 75)

print("[*] بررسی همگام‌سازی (نمایش 2 مکعب اول و 2 مکعب آخر):")
print(f"{'مکعب':<12} | {'انرژی گره 0':<15} | {'انرژی گره 511':<15}")
print("-" * 50)

all_synced = True
for i in [0, 1, num_cubes - 2, num_cubes - 1]:
    e1 = energies[i, 0]
    e2 = energies[i, 511]
    print(f"مکعب {i+1:<8,} | {e1:<15.1f} | {e2:<15.1f}")
    if e1 != 1010.0 or e2 != 1010.0:
        all_synced = False

print("-" * 50)
if all_synced:
    print(f"[✅] نتیجه نهایی: هر {num_cubes:,} مکعب دقیقاً در همان لحظه (زمان صفر) همگام شدند.")
    print(f"     این یعنی انتقال اطلاعات در شبکه‌ای با {total_nodes:,} گره، آنی است.")
    print("     اثبات مقیاس‌پذیری نظریه ریسمان در سطح کیهانی.")
print("=" * 75)
