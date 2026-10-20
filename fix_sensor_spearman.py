with open('genetic_engine_sensor.py', 'r') as f:
    content = f.read()

# جایگزینی تابع local_sensor با نسخه اسپیرمن
old_sensor = '''def local_sensor(X, y, threshold=0.1):
    """
    تشخیص اهمیت متغیرها با استفاده از همبستگی.
    متغیرهایی که اهمیتشان از threshold بیشتر است، اجباری می‌شوند.
    """
    n, d = X.shape
    importances = np.zeros(d)
    
    for i in range(d):
        if np.std(X[:, i]) > 1e-6:
            corr = np.corrcoef(X[:, i], y)[0, 1]
            importances[i] = abs(corr) if not np.isnan(corr) else 0.0'''

new_sensor = '''def local_sensor(X, y, threshold=0.05):
    """
    تشخیص اهمیت متغیرها با استفاده از همبستگی رتبه‌ای (Spearman).
    این روش برای توابع غیرخطی مثل max و abs بسیار مقاوم‌تر است.
    """
    from scipy.stats import spearmanr
    n, d = X.shape
    importances = np.zeros(d)
    
    for i in range(d):
        if np.std(X[:, i]) > 1e-6:
            # استفاده از همبستگی اسپیرمن (غیرخطی)
            try:
                corr, _ = spearmanr(X[:, i], y)
                importances[i] = abs(corr) if not np.isnan(corr) else 0.0
            except:
                importances[i] = 0.0'''

if old_sensor in content:
    content = content.replace(old_sensor, new_sensor)
    with open('genetic_engine_sensor.py', 'w') as f:
        f.write(content)
    print("[+] تابع local_sensor به Spearman تغییر کرد.")
else:
    print("[!] الگوی local_sensor پیدا نشد.")
