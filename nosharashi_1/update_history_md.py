import os
import json
import re

def parse_action(action_str):
    match = re.search(r'(LONG|SHORT)\s+([A-Z0-9-]+)\s+\(([\d\.]+)%\)\s+@\s+\$([\d\.,]+)', action_str)
    if match:
        side, ticker, pct, price = match.groups()
        return ticker, f"${price}", f"{pct}%"
    return "-", "-", "0.0%"

def generate_markdown(adaptive_state, v3_state, output_path):
    v3_trades = v3_state.get('trade_history', [])
    v3_curr_val = v3_state.get('portfolio_value', 10000.0)
    v3_peak_val = v3_state.get('peak_equity', 10000.0)
    v3_ret = ((v3_curr_val - 10000.0) / 10000.0) * 100.0
    v3_action = v3_state.get('last_action', '100% CASH (Neutral)')

    adaptive_trades = adaptive_state.get('trade_history', [])
    adaptive_curr_val = adaptive_state.get('portfolio_value', 10000.0)
    adaptive_ret = ((adaptive_curr_val - 10000.0) / 10000.0) * 100.0
    adaptive_action = adaptive_state.get('last_action', '100% CASH (Neutral)')

    lines = [
        "# 📊 Institutional Paper Trading Execution History",
        "",
        "> **Automated 24/7 Trading Engine**  ",
        "> Tracks execution history, position sizing, leverage, PnL, and drawdowns separately for each deployed strategy.",
        "",
        "---",
        "",
        "## 🚀 Strategy 1: Adaptive v3 Strategy (Hyper-Adaptive Optimizer)",
        "* **Model:** LiquidityAlpha + Square-Root Lasso + Elastic Hysteresis + Ledoit-Wolf Volatility Target",
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
        "## ⚙️ Strategy Comparison & Specifications",
        "",
        "| Feature | Adaptive v3 (Hyper-Adaptive) | Adaptive Weighted (Base) |",
        "|---|---|---|",
        "| **Noise Reduction** | Square-Root Lasso ($\\lambda_{\\text{adj}}$) | Confidence-based clipping |",
        "| **Turnover Control** | Elastic Band Hysteresis ($\\tau$-band) | None (Rebalances every run) |",
        "| **Risk Architecture** | Ledoit-Wolf Shrinkage + 15% Vol Target | Volatility Switch (4.0x $\\to$ 1.0x) |",
        "| **Position Sizing** | Probabilistic ($5\\% - 50\\%$ per asset) | Proportional to signal strength |",
        "| **Execution Cost Model** | Stealth Iceberg ($0.05\\%$ fee + $0.014\\%$ slip) | Stealth Iceberg ($0.05\\%$ fee + $0.014\\%$ slip) |",
        ""
    ])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    s1_path = os.path.join(base_dir, 'paper_state_adaptive.json')
    s2_path = os.path.join(base_dir, 'paper_state_v3.json')
    out_path = os.path.abspath(os.path.join(base_dir, '..', 'TRADE_HISTORY.md'))

    s1, s2 = {}, {}
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

    generate_markdown(s1, s2, out_path)
    print(f"Successfully generated {out_path}")
