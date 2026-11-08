# 📊 Institutional Paper Trading Execution History

> **Automated 24/7 Trading Engine**  
> Tracks execution history, position sizing, leverage, PnL, and drawdowns separately for each deployed strategy.

---

## 🔵 Strategy 1: Adaptive Weighted Strategy (Base Adaptive Model)
* **Model:** LiquidityAlpha + Dynamic Confidence Capping + Volatility Switch Leverage (max 4x) + 2% No-Trade Band
* **Initial Capital:** `$10,000.00`
* **Current Portfolio Value:** `$10,000.00`
* **Peak Equity (HWM):** `$10,000.00`
* **Net Return:** `+0.00%`
* **Current Status:** `100% CASH (Neutral)`

### Trade History Log
| # | Timestamp (IST) | Action / Position | Asset | Entry / Exit Price | Size (%) | Period PnL ($) | Period Return | Portfolio Balance | Total Return | Max DD | Leverage | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 29 Oct 2026, 08:55:03 PM IST | LONG BNB-USD (200.0%) @ $750.77 | BNB-USD | $750.77 | 200.0% | **+$78.10** | **+0.77%** | $10,183.16 | +1.83% | 0.00% | 4.0x | 🟢 PROFIT |

---

## 🚀 Strategy 2: Adaptive v3 Strategy (Hyper-Adaptive Optimizer)
* **Model:** 50d Trend Score (15 assets) + BTC 150d Regime Filter + Square-Root Lasso (λ=0.05) + Elastic Hysteresis + 2% No-Trade Band + Ledoit-Wolf (60d) 15% Vol Target
* **Initial Capital:** `$10,000.00`
* **Current Portfolio Value:** `$10,000.00`
* **Peak Equity (HWM):** `$10,000.00`
* **Net Return:** `+0.00%`
* **Current Status:** `100% CASH (fresh $10,000 forward test; 15-asset v3)`

### Trade History Log
| # | Timestamp (IST) | Action / Position | Asset | Entry / Exit Price | Size (%) | Period PnL ($) | Period Return | Portfolio Balance | Total Return | Max DD | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|

---

## 📉 Strategy 3: TrendVol (Redesigned Signal)
* **Model:** Long-only 50-day time-series momentum on BTC + ETH, inverse-vol sizing, 15% volatility target, 2% No-Trade Band
* **Initial Capital:** `$10,000.00`
* **Current Portfolio Value:** `$10,000.00`
* **Peak Equity (HWM):** `$10,000.00`
* **Net Return:** `+0.00%`
* **Current Status:** `100% CASH (Neutral)`

### Trade History Log
| # | Timestamp (IST) | Action / Position | Asset | Entry / Exit Price | Size (%) | Period PnL ($) | Period Return | Portfolio Balance | Total Return | Max DD | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 07 Nov 2026, 07:38:33 PM IST | 100% CASH (Neutral) | - | - | 0.0% | **+$0.00** | **+0.00%** | $10,000.00 | +0.00% | 0.00% | ⚪ CASH EXIT |

---

## ⚙️ Strategy Comparison & Specifications

| Feature | Adaptive Weighted (Base) | Adaptive v3 (Hyper-Adaptive) | TrendVol (Redesigned) |
|---|---|---|---|
| **Signal** | LiquidityAlpha | 50d trend score across 15 assets + BTC 150d regime | 50d time-series momentum across 15 crypto assets |
| **Turnover Control** | 2% no-trade band + drift-aware turnover | Elastic Hysteresis + 2% no-trade band | 2% no-trade band + drift-aware turnover |
| **Risk Architecture** | Volatility Switch (max 4x), 365-day annualisation, HWM drawdown tracking | 60d Ledoit-Wolf + 15% vol target, max 2x + BTC 150d regime | Inverse-vol + 15% vol target, max 2x, HWM drawdown tracking, drawdown overlay |
| **Position Sizing** | Proportional to signal | Probabilistic (5%-50% per asset) | Inverse-vol, gross normalised |
| **Shorting** | Yes, 3 bp/day financing | Yes (cash below BTC 150d SMA), 3 bp/day financing | No (long or cash) |
| **Universe** | 9 assets | 15 assets | 15 assets |
| **Out-of-sample** | Loses after look-ahead fix; cost controls improve net, not the signal | 2024-26: +21.1%/yr, Sharpe 1.57 CAGR/vol (1.49 mean/std), 10.5% MDD (fixed 15); PIT liquidity universe: +11.3%/yr, 0.81 Sharpe | 2020-26: +58%/yr, Sharpe 2.55, MDD 17.4%; test 2024-26: +33%/yr, Sharpe 1.57, MDD 13.1% (15-asset, post-cost, survivorship-biased) |
| **Execution Cost Model** | Stealth Iceberg (0.05% + 0.014% slip) + 3 bp/day short financing | Stealth Iceberg (0.05% + 0.014% slip) + 3 bp/day short financing | Stealth Iceberg (0.05% + 0.014% slip) |

*v3 backtest note:* the fixed 15-asset result is survivorship-biased because those assets survived to today. A point-in-time top-15-by-liquidity backtest, which includes assets that later collapsed, gives +11.3%/yr and 0.81 Sharpe in 2024-26; its 95% block-bootstrap Sharpe interval includes zero. Treat the fixed-15 figure as an upper bound, not a guarantee. The test window was also observed during tuning, so the edge is not statistically overwhelming after multiple-testing adjustment.
*TrendVol note:* the 15-asset backtest also uses a survivorship-biased universe (assets that exist today). Drawdown overlay is applied (floor 25% exposure beyond 15% drawdown). Treat backtest figures as upper bounds, not guarantees. Forward paper-trading evidence is still limited.
