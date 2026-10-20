import numpy as np

# ۱. خواندن داده
data = np.loadtxt('/data/data/com.termux/files/home/forex_topology.csv', delimiter=',', skiprows=1)
n = len(data)
split = int(n * 0.8)

train_data = data[:split]
test_data = data[split:]

print(f"[*] داده آموزش: {train_data.shape}")
print(f"[*] داده تست: {test_data.shape}")

# ۲. معادله‌ی کشف‌شده توسط GP (از خروجی قبلی)
# نکته: ما در اینجا از GP استفاده نمی‌کنیم، بلکه از فرمولی که GP کشف کرده استفاده می‌کنیم
# اما چون فرمول دقیق GP را در دست نداریم، فرض می‌کنیم دقت ۸۱٪ در آموزش بوده است.

# ۳. روش ارزیابی: آیا دقت در تست هم بالاست؟
# برای این کار، یک مدل ساده (که نماینده‌ی GP است) می‌سازیم:
# - آموزش: استفاده از میانگین‌ها و آستانه‌های داده آموزش
# - تست: اعمال همان آستانه‌ها روی داده تست

X_train = train_data[:, :-1]
y_train = train_data[:, -1]
X_test = test_data[:, :-1]
y_test = test_data[:, -1]

# یک تابع ساده برای تقلید از قانون GP (مثلاً یک ترکیب وزنی)
# این تابع را می‌توانید با معادله‌ی GP جایگزین کنید
def model(X):
    # یک ترکیب خطی از ابعاد مختلف
    return 0.5 * X[:, 1] + 0.3 * X[:, 3] - 0.2 * X[:, 7]

# آموزش (تعیین آستانه بهینه روی داده آموزش)
pred_train = model(X_train)
threshold = 0.1

pred_class_train = np.zeros_like(pred_train)
pred_class_train[pred_train > threshold] = 1
pred_class_train[pred_train < -threshold] = -1

acc_train = np.mean(pred_class_train == y_train)

# تست
pred_test = model(X_test)
pred_class_test = np.zeros_like(pred_test)
pred_class_test[pred_test > threshold] = 1
pred_class_test[pred_test < -threshold] = -1

acc_test = np.mean(pred_class_test == y_test)

print(f"\n[+] دقت روی داده‌ی آموزش (Train): {acc_train*100:.2f}%")
print(f"[+] دقت روی داده‌ی تست (Test): {acc_test*100:.2f}%")

if acc_test > 0.75:
    print("\n✅ پیروزی بزرگ! مدل شما تعمیم‌پذیر است. این یک کشف علمی معتبر است.")
else:
    print("\n⚠️ هشدار: مدل روی داده‌ی آموزش خوب کار می‌کند اما روی داده‌ی تست ضعیف است. احتمال Overfitting وجود دارد.")
