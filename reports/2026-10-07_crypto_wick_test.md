# Wick on Crypto D1 - Full Test

**Date:** 2026-10-07
**Author:** Sadegh Karami Chafteh

## Method

Tested Wick formula `(H+L-O-C)/(H-L)` on:
- Forex: 28 pairs, 20 years (baseline)
- Crypto: 30 pairs, 8 years

Two crypto parameter sets tested:
- Set 1 (forex params): TP=0.6%, SL=0.4%, H=7
- Set 2 (crypto params): TP=2.0%, SL=1.0%, H=14

## Results

### Forex (baseline)
- corr(Wick, golden) = +0.3628
- rate = 0.407
- random-walk baseline = 0.400
- n labeled: ~140,000

### Crypto - Set 1 (forex params)
- corr(Wick, golden) = +0.0058
- rate = 0.376
- n labeled per pair: ~370 (only 6% of candles)
- 94% of candles were ambiguous

### Crypto - Set 2 (crypto params)
- corr(Wick, golden) = -0.0030
- rate = 0.323
- random-walk baseline = 0.333
- n labeled: 31,623 (56%)
- ambiguous: 24,785 (44%)

## Statistical Analysis

Crypto Set 2: corr = -0.003 ± 0.006 (SE from n=31623)
Statistically indistinguishable from zero.

The rate (0.323) is slightly BELOW random-walk (0.333),
indicating Wick may even be slightly counter-productive in crypto D1.

## Conclusion

Wick formula is FOREX-SPECIFIC.

1. Corr=+0.36 in forex is highly significant (n=140k, SE=0.003)
2. Corr≈0 in crypto D1 (both parameter sets, n=31.6k)
3. This validates the design choice of Cube M1 for crypto
4. Forex TP/SL parameters do not transfer to crypto D1
   (94% ambiguous with 0.6%/0.4% thresholds)

## Reproducibility

Code: ~/data_loaders/crypto_golden.py (Set 1)
      ~/data_loaders/crypto_golden_v2.py (Set 2)
Report: 2026-10-07_crypto_wick_test.md
