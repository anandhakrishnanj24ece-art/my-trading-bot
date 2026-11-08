# Edge search — 2026-11-08

## Question

After retracting the look-ahead-inflated TrendVol 15-asset result, is there a
**robust, post-cost, statistically defensible edge** in this crypto data?

Short answer: **no.** The least-bad candidate is BTC+ETH trend, but it is not
statistically significant and its parameters are fragile.

## Protocol (fixed before looking at results)

- **Universe:** 44 assets including the collapsed coins LUNC, USTC, FTT, CEL.
  Daily close and volume, 2020-11-07 → 2026-11-07. This reduces (but cannot
  eliminate) survivorship bias.
- **Split:** train 2021-05-26 → 2024-01-01; **test 2024-01-01 → 2026-11-07**.
  The test window was not used to choose parameters.
- **Causality:** every indicator is a trailing rolling window. Weights at bar
  `t` use only data up to `t` and earn the return from `t` to `t+1`.
- **Execution:** weekly rebalance; holdings drift with prices; 10 bps per unit
  turnover (30 bps stress); 3 bp/day short financing for long-short.
- **Strategies:** 9 pre-specified, economically motivated rules — not a grid:
  time-series momentum, cross-sectional momentum (top 5), cross-sectional
  momentum + BTC 200d regime, long-short cross-sectional momentum, 7-day
  reversal with a 200d trend filter, dual momentum, vol-managed equal weight,
  BTC+ETH trend, BTC-only trend.
- **Significance:** circular block bootstrap (block 20), Bonferroni across the
  9 trials.

Reproduce with:

```bash
python research_edge.py
python sensitivity_btc_eth.py
```

## Results (net of costs)

| Strategy | Train CAGR | Train Sharpe | Train MDD | Test CAGR | Test Sharpe | Test MDD | Test alpha vs BTC | Test p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BTC-only trend | +11.2% | 0.92 | 17.7% | +6.5% | 0.54 | 17.7% | +1.8%/yr | 0.206 |
| **BTC+ETH trend** | +14.0% | 0.86 | 21.9% | **+14.4%** | **0.84** | 19.1% | +7.7%/yr | 0.115 |
| Dual mom top5 (90/200) | +6.3% | 0.26 | 40.7% | −6.3% | −0.22 | 37.0% | −14.1%/yr | 0.572 |
| TSMOM broad (50d) | +7.0% | 0.12 | 74.1% | −2.5% | −0.05 | 65.5% | −8.9%/yr | 0.397 |
| BTC buy & hold | +3.8% | 0.06 | 76.6% | +22.7% | 0.48 | 53.1% | 0.0 | 0.137 |
| XSMOM top5 + BTC200 | +0.9% | 0.05 | 36.6% | −4.4% | −0.20 | 41.2% | −8.8%/yr | 0.565 |
| XSMOM top5 (90d) | −0.3% | −0.01 | 46.6% | +2.7% | 0.09 | 34.1% | −3.7%/yr | 0.348 |
| Vol-managed equal weight | −7.8% | −0.11 | 84.7% | −15.9% | −0.24 | 72.9% | −28.7%/yr | 0.466 |
| Reversal 7d + 200d trend | −9.3% | −0.36 | 47.1% | −2.9% | −0.10 | 41.5% | −11.3%/yr | 0.491 |
| XSMOM long-short top5 | −36.8% | −0.67 | 72.0% | −4.2% | −0.12 | 31.0% | +3.5%/yr | 0.452 |

**Multiple testing:** 9 strategies tested. The Bonferroni threshold at 5% is
`0.05 / 9 = 0.0056`. The best test p-value is **0.115** (BTC+ETH). **Nothing
survives correction.** The best *train* strategy (BTC-only) delivered only
+6.5% CAGR / 0.54 Sharpe out of sample, so train-based selection did not hold.

## BTC+ETH robustness

The only candidate with positive test performance is BTC+ETH trend. Sweeping
its parameters:

| Lookback / vol target | Train CAGR | Train Sharpe | Test CAGR | Test Sharpe | Test MDD |
|---|---:|---:|---:|---:|---:|
| 30 / 10% | +10.8% | 0.97 | +6.6% | 0.65 | 14.8% |
| 30 / 15% | +15.9% | 0.96 | +9.6% | 0.63 | 21.5% |
| 30 / 20% | +20.7% | 0.93 | +12.3% | 0.61 | 27.9% |
| 50 / 10% | +9.6% | 0.88 | +9.8% | 0.86 | 13.0% |
| **50 / 15%** | +14.0% | 0.86 | **+14.4%** | **0.84** | 19.1% |
| 50 / 20% | +18.1% | 0.83 | +18.8% | 0.83 | 24.9% |
| 75 / 10% | −1.9% | −0.19 | +7.6% | 0.68 | 17.3% |
| 75 / 15% | −3.2% | −0.21 | +11.0% | 0.67 | 25.0% |
| 75 / 20% | −4.8% | −0.23 | +14.2% | 0.65 | 32.0% |
| 100 / 10% | +0.1% | 0.01 | +4.9% | 0.43 | 21.7% |
| 100 / 15% | −0.3% | −0.02 | +6.8% | 0.40 | 31.1% |
| 100 / 20% | −0.9% | −0.04 | +8.4% | 0.37 | 39.4% |
| 150 / 10% | +3.7% | 0.34 | +4.0% | 0.36 | 14.0% |
| 150 / 15% | +5.1% | 0.31 | +5.6% | 0.34 | 20.4% |
| 150 / 20% | +6.2% | 0.29 | +6.9% | 0.32 | 26.5% |

Rebalance frequency (lookback 50, 15% target):

| Rebalance | Train CAGR | Train Sharpe | Test CAGR | Test Sharpe | Test MDD |
|---|---:|---:|---:|---:|---:|
| 1 day | +12.3% | 0.75 | +16.9% | 0.98 | 16.7% |
| 7 days | +14.0% | 0.86 | +14.4% | 0.84 | 19.1% |
| 14 days | +0.1% | 0.01 | +19.2% | 1.07 | 20.1% |
| 30 days | −1.6% | −0.10 | +1.3% | 0.06 | 42.9% |

This is **not a clean plateau**. Lookbacks 30 and 50 work; 75 and 100 have
negative or near-zero *train* CAGR; 150 is weak; monthly rebalancing fails.
The test result at lookback 50 is the best cell in the table, which is a
warning sign, not a validation.

## Honest conclusion

- **No robust, statistically significant edge was found.** Every
  cross-sectional, reversal, vol-managed, and long-short variant failed
  out-of-sample; several lost money.
- **BTC+ETH trend is the least-bad candidate**, with +14.4% CAGR / 0.84 Sharpe
  / 19.1% MDD in the test window. But its p-value is 0.115, it does not survive
  multiple-testing correction, and it is fragile across lookbacks.
- It also **underperformed simply holding BTC** in the test window (+14.4% vs
  +22.7% CAGR). Its value is **lower drawdown** (19.1% vs 53.1%), not higher
  return. It is a lower-vol way to hold crypto beta, not a proven alpha engine.

## Decision

- Keep TrendVol **paper-only, BTC+ETH**, with the causal drawdown overlay.
- Do **not** add capital and do **not** deploy any of the new strategies.
- Re-run this protocol after another 6–12 months of live data. If BTC+ETH
  trend is still positive out-of-sample then, the evidence is stronger; if
  not, retire it.
