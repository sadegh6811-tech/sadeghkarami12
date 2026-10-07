---
title: 'MathBot: Symbolic Discovery of Implicit Manifold Equations from Point Clouds'
tags:
  - Python
  - symbolic regression
  - manifold learning
  - implicit equations
  - dimensionality reduction
  - scientific computing
authors:
  - name: Sadegh Karami Chefteh
    affiliation: 1
affiliations:
  - name: Independent Researcher, Iran
    index: 1
date: 7 October 2026
bibliography: paper.bib
---

# Summary

`MathBot` is a Python package that discovers the mathematical equations describing a point cloud lying on an unknown manifold in $\mathbb{R}^n$. Given a set of points in 2, 3, 4 or 5 dimensions, `MathBot` returns one or more implicit equations $F(x_1, \dots, x_n) = 0$ or explicit transcendental relations such as $y = \sin(\omega x)$, written in the original coordinate system of the input data.

The package is designed for researchers who need to reverse-engineer the governing equation of an empirically observed manifold, without prior knowledge of the polynomial degree, the intrinsic dimension, or the functional family to which the equation belongs.

Unlike general symbolic-regression tools that produce free-form expression trees, `MathBot` is specialized for the implicit manifold discovery problem: it explicitly estimates the intrinsic dimension $d$ of the data and therefore knows exactly how many equations $n - d$ it must find, providing a principled stopping criterion that general-purpose regressors lack.

# Statement of need

Symbolic regression has become a standard technique for discovering closed-form laws from data [@schmidt2009; @cranmer2023]. Existing packages such as PySR [@cranmer2023] and AI Feynman [@udrescu2020] excel at recovering a single scalar function $y = f(\mathbf{x})$ from input-output pairs. Sparse identification methods such as SINDy [@brunton2016] recover ordinary and partial differential equations from time-series data. None of these tools, however, is directly aimed at the manifold setting, in which the given data is a point cloud that lies on an unknown lower-dimensional surface $M \subset \mathbb{R}^n$ and the task is to produce the implicit equations of $M$.

`MathBot` fills this gap. It combines well-established ingredients in a new pipeline: (i) robust intrinsic dimension estimation with the TwoNN and Maximum Likelihood estimators [@facco2017; @levina2004], (ii) local PCA denoising for noisy point clouds [@zhang2004], (iii) a circle-pair pre-filter that detects pairs of variables satisfying $x_i^2 + x_j^2 \approx \mathrm{const}$ before any frequency analysis, (iv) a grid-search frequency estimator that is more robust than FFT on sparse or non-uniform data, (v) a mixed polynomial-trigonometric dictionary, and (vi) an SVD-based null-space solver whose polynomial degree is chosen automatically from the spectral gap of the design matrix.

The circle-pair pre-filter and the grid-search frequency detector are the two most distinctive components. Together they allow `MathBot` to discover equations involving trigonometric terms, such as the helix $x = \cos(\omega z), y = \sin(\omega z)$, that polynomial-only pipelines miss, while avoiding the spurious frequencies that a naive FFT produces when two variables are constrained to lie on a circle.

# Implementation

`MathBot` is implemented in pure Python and depends only on `numpy`, `scipy` (optional, with a pure-NumPy fallback for the KD-tree), and `sympy`. No machine-learning framework is required, and the package runs on CPU-only environments, including Android through Termux.

The package exposes a single command-line entry point:

    python mathbot.py samples/sphere_3d.csv 2

which, for a 3-D point cloud sampled from the unit sphere, returns $x_0^2 + x_1^2 + x_2^2 - 1 = 0$ to four decimal places. Every discovery can be saved in `.txt`, `.tex`, and `.json` formats with the `--save` flag.

# Benchmarks

`MathBot` was tested on six synthetic manifolds of known ground truth covering intrinsic dimensions 1, 2, and 3 and ambient dimensions 2, 3, and 4.

| Input                  | $n$ | $d$ | Discovered equation                     |
|------------------------|-----|-----|-----------------------------------------|
| circle (2-D)           | 2   | 1   | $x_0^2 + x_1^2 = 1$                     |
| sphere (3-D)           | 3   | 2   | $x_0^2 + x_1^2 + x_2^2 = 1$             |
| sphere (4-D)           | 4   | 3   | $x_0^2 + x_1^2 + x_2^2 + x_3^2 = 1$     |
| torus (4-D)            | 4   | 2   | $x_0^2 + x_1^2 = 1,\ x_2^2 + x_3^2 = 1$ |
| helix (3-D)            | 3   | 1   | $x_1 = \sin(4\pi x_2),\ x_0 = \cos(4\pi x_2)$ |
| sphere (3-D, 4% noise) | 3   | 2   | $x_0^2 + x_1^2 + x_2^2 = 1$             |

All six cases were recovered with intrinsic dimensions matching the ground truth and polynomial degrees chosen automatically: degree 2 for quadrics and degree 1 for the trigonometric helix.

# Acknowledgements

The author thanks the developers of `numpy`, `scipy`, and `sympy` for their foundational open-source tools.

# References
