import numpy as np
from genetic_engine_classifier import evaluate_tree

BEST_TREE = ('div',
    ('add', ('add', 'x7', 'x2'), ('const', 0.0)),
    ('add', 
        ('add', ('abs', ('mul', ('sub', ('add', 'x6', 'x1'), ('sub', 'x1', 'x0')), ('div', ('div', 'x1', 'x6'), ('abs', 'x0')))),
            ('div', ('const', -1.0), 'x5')
        ),
        ('abs', ('sub', ('abs', 'x2'), ('div', ('const', 0.5), 'x7')))
    )
)

def normalize(x):
    """نرمال‌سازی MinMax به بازه [-1, 1]"""
    if np.max(x) - np.min(x) < 1e-12:
        return np.zeros_like(x)
    return 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1

universes = np.load('/data/data/com.termux/files/home/parallel_universes.npy', allow_pickle=True)

print("[*] تست فرمول GP با نرمال‌سازی MinMax...")

results = []
for i, universe in enumerate(universes[:100]):
    phi = universe[:, 0]
    phi_dot = universe[:, 1]
    phi_ddot = universe[:, 2]
    
    X = np.column_stack([
        normalize(phi),
        normalize(np.sign(np.gradient(phi))),
        normalize(np.abs(phi_dot)),
        normalize(np.abs(phi_ddot)),
        normalize(np.abs(phi * phi_dot)),
        np.ones_like(phi),
        np.zeros_like(phi),
        normalize(np.gradient(phi)),
        normalize(np.linspace(0, 2*np.pi, len(phi)))
    ])
    
    pred = evaluate_tree(BEST_TREE, X)
    pred_sig = np.zeros_like(pred)
    pred_sig[pred > 0.1] = 1
    pred_sig[pred < -0.1] = -1
    
    results.append(pred_sig)

all_sigs = np.concatenate(results)
print(f"\n═══ نتایج با نرمال‌سازی MinMax ═══")
print(f"[*] سیگنال‌های صعودی: {np.sum(all_sigs == 1)}")
print(f"[*] سیگنال‌های نزولی: {np.sum(all_sigs == -1)}")
print(f"[*] سیگنال‌های صفر: {np.sum(all_sigs == 0)}")
