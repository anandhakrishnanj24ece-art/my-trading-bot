import os
import json
import re

def parse_action(action_str):
    """Render every position in a multi-asset action string (not just the first)."""
    matches = re.findall(r'(LONG|SHORT)\s+([A-Z0-9-]+)\s+\(([\d\.]+)%\)\s+@\s+\$([\d\.,]+)', action_str)
    if not matches:
        return "-", "-", "0.0%"
    tickers = ", ".join(m[1] for m in matches)
    prices = ", ".join(f"${m[3]}" for m in matches)
    sizes = ", ".join(f"{m[2]}%" for m in matches)
    return tickers, prices, sizes

def generate_markdown(adaptive_state, v3_state, trend_state, output_path):
    v3_trades = v3_state.get('trade_history', [])
    v3_curr_val = v3_state.get('portfolio_value', 10000.0)
    v3_peak_val = v3_state.get('peak_equity', 10000.0)
    v3_ret = ((v3_curr_val - 10000.0) / 10000.0) * 100.0
    v3_action = v3_state.get('last_action', '100% CASH (Neutral)')

    adaptive_trades = adaptive_state.get('trade_history', [])
    adaptive_curr_val = adaptive_state.get('portfolio_value', 10000.0)
    adaptive_ret = ((adaptive_curr_val - 10000.0) / 10000.0) * 100.0
    adaptive_action = adaptive_state.get('last_action', '100% CASH (Neutral)')

    trend_trades = trend_state.get('trade_history', [])
    trend_curr_val = trend_state.get('portfolio_value', 10000.0)
    trend_peak_val = trend_state.get('peak_equity', 10000.0)
    trend_ret = ((trend_curr_val - 10000.0) / 10000.0) * 100.0
    trend_action = trend_state.get('last_action', '100% CASH (Neutral)')

    lines = [
        "# 📊 Institutional Paper Trading Execution History",
        "",
        "> **Automated 24/7 Trading Engine**  ",
        "> Tracks execution history, position sizing, leverage, PnL, and drawdowns separately for each deployed strategy.",
        "",
        "---",
        "",
        "## 🚀 Strategy 1: Adaptive v3 Strategy (Hyper-Adaptive Optimizer)",
        "* **Model:** 50d Trend Score (15 assets) + BTC 150d Regime Filter + Square-Root Lasso (λ=0.05) + Elastic Hysteresis + 2% No-Trade Band + Ledoit-Wolf (60d) 15% Vol Target",
        "* **Initial Capital:** `$10,000.00`",
        f"* **Current Portfolio Value:** `${v3_curr_val:,.2f}`",
        f"* **Peak Equity (HWM):** `${v3_peak_val:,.2f}`",
        f"* **Net Return:** `{v3_ret:+.2f}%`",
        f"* **Current Status:** `{v3_action}`",
        "",
        "### Trade History Log",
        "| # | Timestamp (IST) | Action / Position | Asset | Entry / Exit Price | Size (%) | Period PnL ($) | Period Return | Portfolio Balance | Total Return | Max DD | Result |",
        "|---|-----------------|-------------------|-------|--------------------|----------|----------------|---------------|-------------------|--------------|--------|--------|"
    ]

    for idx, t in enumerate(v3_trades, 1):
        action = t.get('action', '')
        asset, price, size = parse_action(action)
        pnl = t.get('period_pnl_usd', 0.0)
        ret = t.get('period_return_pct', 0.0)
        balance = t.get('portfolio_value', 0.0)
        tot_ret = t.get('total_return_pct', 0.0)
        max_dd = t.get('max_dd', 0.0) * 100.0

        if "CASH" in action:
            res = "⚪ CASH EXIT"
        elif pnl > 0:
            res = "🟢 PROFIT"
        else:
            res = "🔴 LOSS"

        pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
        ret_str = f"+{ret:.2f}%" if ret >= 0 else f"{ret:.2f}%"
        tot_ret_str = f"+{tot_ret:.2f}%" if tot_ret >= 0 else f"{tot_ret:.2f}%"

        lines.append(
            f"| {idx} | {t.get('timestamp_ist', '')} | {action} | {asset} | {price} | {size} | **{pnl_str}** | **{ret_str}** | ${balance:,.2f} | {tot_ret_str} | {max_dd:.2f}% | {res} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 📈 Strategy 2: Adaptive Weighted Strategy (Base Adaptive Model)",
        "* **Model:** LiquidityAlpha + Dynamic Confidence Capping + Volatility Switch Leverage (4.0x)",
        "* **Initial Capital:** `$10,000.00`",
        f"* **Current Portfolio Value:** `${adaptive_curr_val:,.2f}`",
        f"* **Net Return:** `{adaptive_ret:+.2f}%`",
        f"* **Current Status:** `{adaptive_action}`",
        "",
        "### Trade History Log",
        "| # | Timestamp (IST) | Action / Position | Asset | Entry / Exit Price | Size (%) | Period PnL ($) | Period Return | Portfolio Balance | Total Return | Leverage | Result |",
        "|---|-----------------|-------------------|-------|--------------------|----------|----------------|---------------|-------------------|--------------|----------|--------|"
    ])

    for idx, t in enumerate(adaptive_trades, 1):
        action = t.get('action', '')
        asset, price, size = parse_action(action)
        pnl = t.get('period_pnl_usd', 0.0)
        ret = t.get('period_return_pct', 0.0)
        balance = t.get('portfolio_value', 0.0)
        tot_ret = t.get('total_return_pct', 0.0)
        lev = t.get('leverage', 4.0)

        if "CASH" in action:
            res = "⚪ CASH EXIT"
        elif pnl > 0:
            res = "🟢 PROFIT"
        else:
            res = "🔴 LOSS"

        pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
        ret_str = f"+{ret:.2f}%" if ret >= 0 else f"{ret:.2f}%"
        tot_ret_str = f"+{tot_ret:.2f}%" if tot_ret >= 0 else f"{tot_ret:.2f}%"

        lines.append(
            f"| {idx} | {t.get('timestamp_ist', '')} | {action} | {asset} | {price} | {size} | **{pnl_str}** | **{ret_str}** | ${balance:,.2f} | {tot_ret_str} | {lev:.1f}x | {res} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 📉 Strategy 3: TrendVol (Redesigned Signal)",
        "* **Model:** Long-only 50-day time-series momentum on BTC + ETH, inverse-vol sizing, 15% volatility target",
        "* **Initial Capital:** `$10,000.00`",
        f"* **Current Portfolio Value:** `${trend_curr_val:,.2f}`",
        f"* **Peak Equity (HWM):** `${trend_peak_val:,.2f}`",
        f"* **Net Return:** `{trend_ret:+.2f}%`",
        f"* **Current Status:** `{trend_action}`",
        "",
        "### Trade History Log",
        "| # | Timestamp (IST) | Action / Position | Asset | Entry / Exit Price | Size (%) | Period PnL ($) | Period Return | Portfolio Balance | Total Return | Max DD | Result |",
        "|---|-----------------|-------------------|-------|--------------------|----------|----------------|---------------|-------------------|--------------|--------|--------|"
    ])

    for idx, t in enumerate(trend_trades, 1):
        action = t.get('action', '')
        asset, price, size = parse_action(action)
        pnl = t.get('period_pnl_usd', 0.0)
        ret = t.get('period_return_pct', 0.0)
        balance = t.get('portfolio_value', 0.0)
        tot_ret = t.get('total_return_pct', 0.0)
        max_dd = t.get('max_dd', 0.0) * 100.0

        if "CASH" in action:
            res = "⚪ CASH EXIT"
        elif pnl > 0:
            res = "🟢 PROFIT"
        else:
            res = "🔴 LOSS"

        pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
        ret_str = f"+{ret:.2f}%" if ret >= 0 else f"{ret:.2f}%"
        tot_ret_str = f"+{tot_ret:.2f}%" if tot_ret >= 0 else f"{tot_ret:.2f}%"

        lines.append(
            f"| {idx} | {t.get('timestamp_ist', '')} | {action} | {asset} | {price} | {size} | **{pnl_str}** | **{ret_str}** | ${balance:,.2f} | {tot_ret_str} | {max_dd:.2f}% | {res} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## ⚙️ Strategy Comparison & Specifications",
        "",
        "| Feature | Adaptive v3 (Hyper-Adaptive) | Adaptive Weighted (Base) | TrendVol (Redesigned) |",
        "|---|---|---|---|",
        "| **Signal** | 50d trend score across 15 assets + BTC 150d regime | LiquidityAlpha | 50d time-series momentum on BTC + ETH |",
        "| **Turnover Control** | Elastic Hysteresis + 2% no-trade band | None | Daily, ~4%/bar |",
        "| **Risk Architecture** | 60d Ledoit-Wolf + 15% vol target, max 2x + BTC 150d regime | Volatility Switch (4x -> 1x) | Inverse-vol + 15% vol target, max 2x |",
        "| **Position Sizing** | Probabilistic (5%-50% per asset) | Proportional to signal | Inverse-vol, gross normalised |",
        "| **Shorting** | Yes (cash below BTC 150d SMA) | Yes | No (long or cash) |",
        "| **Out-of-sample** | 2024-26: +21.1%/yr, Sharpe 1.57 CAGR/vol (1.49 mean/std), 10.5% MDD (fixed 15); PIT liquidity universe: +11.3%/yr, 0.81 Sharpe | Loses after look-ahead fix | Sharpe ~1.0, +10%/yr alpha vs BTC |",
        "| **Execution Cost Model** | Stealth Iceberg (0.05% + 0.014% slip) + 3 bp/day short financing | Stealth Iceberg (0.05% + 0.014% slip) | Stealth Iceberg (0.05% + 0.014% slip) |",
        "",
        "*v3 backtest note:* the fixed 15-asset result is survivorship-biased because those assets survived to today. A point-in-time top-15-by-liquidity backtest, which includes assets that later collapsed, gives +11.3%/yr and 0.81 Sharpe in 2024-26; its 95% block-bootstrap Sharpe interval includes zero. Treat the fixed-15 figure as an upper bound, not a guarantee. The test window was also observed during tuning, so the edge is not statistically overwhelming after multiple-testing adjustment.",
        ""
    ])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    s1_path = os.path.join(base_dir, 'paper_state_adaptive.json')
    s2_path = os.path.join(base_dir, 'paper_state_v3.json')
    s3_path = os.path.join(base_dir, 'paper_state_trend.json')
    out_path = os.path.abspath(os.path.join(base_dir, '..', 'TRADE_HISTORY.md'))

    s1, s2, s3 = {}, {}, {}
    if os.path.exists(s1_path):
        try:
            with open(s1_path, 'r', encoding='utf-8') as f:
                s1 = json.load(f)
        except Exception:
            pass
    if os.path.exists(s2_path):
        try:
            with open(s2_path, 'r', encoding='utf-8') as f:
                s2 = json.load(f)
        except Exception:
            pass
    if os.path.exists(s3_path):
        try:
            with open(s3_path, 'r', encoding='utf-8') as f:
                s3 = json.load(f)
        except Exception:
            pass

    generate_markdown(s1, s2, s3, out_path)
    print(f"Successfully generated {out_path}")
