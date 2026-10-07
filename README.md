# MathBot — Symbolic Discovery of Implicit Manifold Equations

**Author:** Sadegh Karami Chefteh (صادق کرمی چفته) — Iran

**MathBot** discovers the mathematical equations describing a point cloud lying on an unknown manifold in R^n (n = 2..5). It automatically estimates intrinsic dimension, denoises, builds a mixed polynomial-trigonometric library, detects circle pairs, scans frequencies, and finds implicit equations via SVD.

## Features

- Dimension-agnostic: works in R^2 to R^5
- Automatic intrinsic-dimension estimation (TwoNN + MLE)
- Local PCA denoising (up to 4% noise)
- Circle-aware pre-filtering for FFT
- Grid-search frequency detection
- Automatic polynomial degree selection
- Symbolic output ready for LaTeX
- Auto-save to TXT, TEX, JSON

## Installation

### Standard Python

    git clone https://github.com/sadeghkarami/mathbot.git
    cd mathbot
    pip install numpy scipy sympy

### Termux (Android)

    pkg update && pkg upgrade -y
    pkg install -y python python-numpy python-scipy python-pip clang
    pip install --break-system-packages sympy

Note: On Termux, never run "pip install --upgrade pip". Use pkg for system packages.

## Quick Start

    python make_sample.py
    python mathbot.py samples/sphere_3d.csv 2
    python mathbot.py samples/sphere_3d.csv 2 --save

## Benchmarks

| Input | n | d | Discovered equation |
|---|---|---|---|
| circle_2d | 2 | 1 | X0^2 + X1^2 = 1 |
| sphere_3d | 3 | 2 | X0^2 + X1^2 + X2^2 = 1 |
| sphere_4d | 4 | 3 | X0^2 + X1^2 + X2^2 + X3^2 = 1 |
| torus_4d | 4 | 2 | X0^2+X1^2=1 and X2^2+X3^2=1 |
| helix_3d | 3 | 1 | X1 = sin(4pi*X2), X0 = cos(4pi*X2) |
| sphere_3d_noisy | 3 | 2 | X0^2 + X1^2 + X2^2 = 1 (4% noise) |

## Project Structure

    mathbot/
      mathbot.py           Main entry point
      mathbot_core.py      Intrinsic dim + libraries
      discover.py          Equation discovery with SVD
      freq_lib.py          Frequency detection
      denoise.py           Local PCA denoising
      saver.py             Auto-save discoveries
      make_sample.py       Data generator
      samples/             Test data (CSV)
      results/             Auto-generated discoveries

## Dependencies

- numpy
- scipy (optional, has fallback)
- sympy

No ML frameworks required. Runs on CPU, works on Android via Termux.

## Author

**Sadegh Karami Chefteh** — Independent Researcher, Iran

GitHub: https://github.com/sadeghkarami

## Citing

    @software{karami2026mathbot,
      author = {Karami Chefteh, Sadegh},
      title  = {MathBot: Symbolic Discovery of Implicit Manifold Equations},
      year   = 2026,
      url    = {https://github.com/sadeghkarami/mathbot}
    }

## License

MIT License — see LICENSE file.
