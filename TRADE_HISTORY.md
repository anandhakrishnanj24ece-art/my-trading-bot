# 📊 Institutional Paper Trading Execution History

> **Automated 24/7 Trading Engine**  
> Tracks execution history, position sizing, leverage, PnL, and drawdowns separately for each deployed strategy.

---

## 🚀 Strategy 1: Adaptive v3 Strategy (Hyper-Adaptive Optimizer)
* **Model:** LiquidityAlpha + Square-Root Lasso + Elastic Hysteresis + Ledoit-Wolf Volatility Target
* **Initial Capital:** `$10,000.00`
* **Current Portfolio Value:** `$9,594.14`
* **Peak Equity (HWM):** `$10,063.07`
* **Net Return:** `-4.06%`
* **Current Status:** `100% CASH (Neutral)`

### Trade History Log
| # | Timestamp (IST) | Action / Position | Asset | Entry / Exit Price | Size (%) | Period PnL ($) | Period Return | Portfolio Balance | Total Return | Max DD | Result |
|---|-----------------|-------------------|-------|--------------------|----------|----------------|---------------|-------------------|--------------|--------|--------|
| 1 | 29 Oct 2026, 08:55:18 PM IST | LONG BNB-USD (67.0%) @ $750.77 | BNB-USD | $750.77 | 67.0% | **+$25.76** | **+0.26%** | $10,063.07 | +0.63% | 0.00% | 🟢 PROFIT |
| 2 | 30 Oct 2026, 12:57:57 AM IST | LONG BNB-USD (65.5%) @ $743.06 | BNB-USD | $743.06 | 65.5% | **-$69.32** | **-0.69%** | $9,993.74 | -0.06% | 0.69% | 🔴 LOSS |
| 3 | 30 Oct 2026, 05:21:28 AM IST | 100% CASH (Neutral) | - | - | 0.0% | **-$399.61** | **-3.96%** | $9,594.14 | -4.06% | 4.66% | ⚪ CASH EXIT |
| 4 | 30 Oct 2026, 10:30:35 AM IST | 100% CASH (Neutral) | - | - | 0.0% | **+$0.00** | **+0.00%** | $9,594.14 | -4.06% | 4.66% | ⚪ CASH EXIT |
| 5 | 30 Oct 2026, 05:56:20 PM IST | 100% CASH (Neutral) | - | - | 0.0% | **+$0.00** | **+0.00%** | $9,594.14 | -4.06% | 4.66% | ⚪ CASH EXIT |

---

## 📈 Strategy 2: Adaptive Weighted Strategy (Base Adaptive Model)
* **Model:** LiquidityAlpha + Dynamic Confidence Capping + Volatility Switch Leverage (4.0x)
* **Initial Capital:** `$10,000.00`
* **Current Portfolio Value:** `$8,755.33`
* **Net Return:** `-12.45%`
* **Current Status:** `100% CASH (Neutral)`

### Trade History Log
| # | Timestamp (IST) | Action / Position | Asset | Entry / Exit Price | Size (%) | Period PnL ($) | Period Return | Portfolio Balance | Total Return | Leverage | Result |
|---|-----------------|-------------------|-------|--------------------|----------|----------------|---------------|-------------------|--------------|----------|--------|
| 1 | 29 Oct 2026, 08:55:03 PM IST | LONG BNB-USD (200.0%) @ $750.77 | BNB-USD | $750.77 | 200.0% | **+$78.10** | **+0.77%** | $10,183.16 | +1.83% | 4.0x | 🟢 PROFIT |
| 2 | 30 Oct 2026, 12:57:54 AM IST | LONG BNB-USD (200.0%) @ $743.06 | BNB-USD | $743.06 | 200.0% | **-$209.15** | **-2.05%** | $9,974.01 | -0.26% | 4.0x | 🔴 LOSS |
| 3 | 30 Oct 2026, 05:21:26 AM IST | 100% CASH (Neutral) | - | - | 0.0% | **-$1,218.68** | **-12.09%** | $8,755.33 | -12.45% | 4.0x | ⚪ CASH EXIT |
| 4 | 30 Oct 2026, 10:30:33 AM IST | 100% CASH (Neutral) | - | - | 0.0% | **+$0.00** | **+0.00%** | $8,755.33 | -12.45% | 4.0x | ⚪ CASH EXIT |
| 5 | 30 Oct 2026, 05:56:18 PM IST | 100% CASH (Neutral) | - | - | 0.0% | **+$0.00** | **+0.00%** | $8,755.33 | -12.45% | 4.0x | ⚪ CASH EXIT |

---

## ⚙️ Strategy Comparison & Specifications

| Feature | Adaptive v3 (Hyper-Adaptive) | Adaptive Weighted (Base) |
|---|---|---|
| **Noise Reduction** | Square-Root Lasso ($\lambda_{\text{adj}}$) | Confidence-based clipping |
| **Turnover Control** | Elastic Band Hysteresis ($\tau$-band) | None (Rebalances every run) |
| **Risk Architecture** | Ledoit-Wolf Shrinkage + 15% Vol Target | Volatility Switch (4.0x $\to$ 1.0x) |
| **Position Sizing** | Probabilistic ($5\% - 50\%$ per asset) | Proportional to signal strength |
| **Execution Cost Model** | Stealth Iceberg ($0.05\%$ fee + $0.014\%$ slip) | Stealth Iceberg ($0.05\%$ fee + $0.014\%$ slip) |
