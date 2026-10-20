"""Sparse regression با OMP — نسخه ۲ (با نرمال‌سازی صحیح)."""
import numpy as np


def omp(A, y, n_terms=10, tol=1e-10):
    """
    Orthogonal Matching Pursuit با استانداردسازی داخلی.
    
    A: (n_samples, n_features)
    y: (n_samples,)  یا  (n_samples, n_targets)
    n_terms: تعداد ترم‌های غیرصفر
    
    Returns:
        coef      — ضرایب در فضای اصلی A (n_features,) یا (n_features, n_targets)
        active    — لیست اندیس ترم‌های انتخاب‌شده
        intercept — عرض از مبدأ در فضای اصلی
    """
    y_was_1d = (y.ndim == 1)
    if y_was_1d:
        y = y.reshape(-1, 1)
    
    n_samples, n_features = A.shape
    n_targets = y.shape[1]
    
    # ─── استانداردسازی ───
    A_mean = A.mean(axis=0)
    A_std = A.std(axis=0)
    A_std = np.where(A_std < 1e-12, 1.0, A_std)
    A_proc = (A - A_mean) / A_std
    
    y_mean = y.mean(axis=0)
    y_std = y.std(axis=0)
    y_std = np.where(y_std < 1e-12, 1.0, y_std)
    y_proc = (y - y_mean) / y_std
    
    coef_proc = np.zeros((n_features, n_targets))
    residual = y_proc.copy()
    active = []
    
    for _ in range(min(n_terms, n_features)):
        corr = A_proc.T @ residual           # (n_features, n_targets)
        if n_targets > 1:
            score = np.linalg.norm(corr, axis=1)
        else:
            score = np.abs(corr[:, 0])
        
        # ترم‌های انتخاب‌شده رو نادیده بگیر
        for idx in active:
            score[idx] = -1.0
        
        new_idx = int(np.argmax(score))
        if score[new_idx] < tol:
            break
        
        active.append(new_idx)
        A_active = A_proc[:, active]
        coef_active, *_ = np.linalg.lstsq(A_active, y_proc, rcond=None)
        coef_proc[active] = coef_active
        residual = y_proc - A_active @ coef_active
    
    # ─── تبدیل به فضای اصلی ───
    coef_orig = np.zeros((n_features, n_targets))
    if active:
        # y = intercept + A[:, active] @ coef_orig
        # از y_proc = A_proc[:, active] @ coef_proc نتیجه می‌گیریم:
        coef_orig[active] = coef_proc[active] * y_std[None, :] / A_std[active, None]
        intercept = y_mean - (coef_proc[active] * y_std[None, :] / A_std[active, None]).T @ A_mean[active]
    else:
        intercept = y_mean.copy()
    
    if y_was_1d:
        coef_orig = coef_orig[:, 0]
        intercept = float(intercept[0]) if hasattr(intercept, '__len__') else float(intercept)
    
    return coef_orig, active, intercept
