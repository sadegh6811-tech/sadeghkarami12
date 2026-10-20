import numpy as np

# بارگذاری داده‌ها
data = np.loadtxt('/data/data/com.termux/files/home/zeta_data_1000.csv', delimiter=',', skiprows=1)
T = data[:, 2]
actual = data[:, 3]

def riemann(T):
    return 2 * np.pi / np.log(T / (2 * np.pi) + 1e-9)

pred_riemann = riemann(T)

# حذف نقاط پرت برای خوانایی
mask = np.isfinite(pred_riemann) & (pred_riemann < 10) & (actual < 10)
T = T[mask]
actual = actual[mask]
pred_riemann = pred_riemann[mask]

# ساخت HTML با Chart.js
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Riemann vs Actual</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background: #1a1a1a; color: white; font-family: Arial; padding: 20px; }
        canvas { background: #2a2a2a; border-radius: 10px; }
        h1 { text-align: center; }
    </style>
</head>
<body>
    <h1>Gap vs T: Actual vs Riemann-Formula</h1>
    <canvas id="myChart" width="800" height="400"></canvas>
    <script>
        const T = %s;
        const actual = %s;
        const riemann = %s;
        
        const ctx = document.getElementById('myChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: T.map(x => x.toFixed(1)),
                datasets: [
                    {
                        label: 'Actual Gap',
                        data: actual,
                        borderColor: 'rgba(255, 99, 132, 1)',
                        backgroundColor: 'rgba(255, 99, 132, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.1
                    },
                    {
                        label: 'Riemann Formula',
                        data: riemann,
                        borderColor: 'rgba(54, 162, 235, 1)',
                        backgroundColor: 'rgba(54, 162, 235, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: { display: true, text: 'Comparison of Actual Gap vs Riemann Formula', color: 'white' },
                    legend: { labels: { color: 'white' } }
                },
                scales: {
                    x: { title: { display: true, text: 'T (imaginary part)', color: 'white' }, ticks: { color: 'white', maxTicksLimit: 20 } },
                    y: { title: { display: true, text: 'Gap', color: 'white' }, ticks: { color: 'white' } }
                }
            }
        });
    </script>
</body>
</html>
""" % (T.tolist(), actual.tolist(), pred_riemann.tolist())

with open('/data/data/com.termux/files/home/riemann_plot.html', 'w') as f:
    f.write(html)

print("[+] نمودار ساخته شد: ~/riemann_plot.html")
print("[*] برای مشاهده: در مرورگر گوشی، فایل را باز کنید.")
print("[*] یا با دستور: termux-open ~/riemann_plot.html")
