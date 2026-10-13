# Cross-Market Validation of the Wick Formula

**Date:** 2026-10-07
**Author:** Sadegh Karami Chafteh

## Method

Tested Wick formula `(H+L-O-C)/(H-L)` on:
- **Forex:** 28 pairs, 20 years daily, label = long_golden
- **Crypto:** 20 pairs, 8 years daily, target = future_return_1

## Results

### Forex
- Mean corr(Wick, golden) = **+0.3628**
- Mean accuracy (threshold=0.30) = **0.6952**
- 25/28 pairs have positive corr (+0.31 to +0.46)
- 3 exceptions: AUDCHF, EURGBP, NZDUSD (corr ≈ 0)

### Crypto
- Mean corr(Wick, future_return) = **-0.0083**
- Range: -0.045 to +0.019
- No pair shows meaningful correlation

## Conclusion

Wick is **market-specific**. It does NOT generalize to crypto D1.
This validates the design choice of Cube Strategy for crypto.

## Caveat

Forex uses TP/SL-based label (long_golden).
Crypto uses 1-bar return.
Comparison is not perfectly fair — crypto needs a matching label.
