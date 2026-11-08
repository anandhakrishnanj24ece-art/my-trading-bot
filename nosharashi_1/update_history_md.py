import json
import os
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


def result_label(action, pnl):
    if "CASH" in action:
        return "⚪ CASH EXIT"
    if pnl > 0:
        return "🟢 PROFIT"
    return "🔴 LOSS"


def signed_money(value):
    return f"+${value:,.2f}" if value >= 0 else f"-${abs(value):,.2f}"


def signed_pct(value):
    return f"+{value:.2f}%" if value >= 0 else f"{value:.2f}%"


def render_trade_rows(trades, include_drawdown=False, include_leverage=False):
    rows = []
    for idx, trade in enumerate(trades, 1):
        action = trade.get('action', '')
        asset, price, size = parse_action(action)
        pnl = trade.get('period_pnl_usd', 0.0)
        ret = trade.get('period_return_pct', 0.0)
        balance = trade.get('portfolio_value', 0.0)
        total_ret = trade.get('total_return_pct', 0.0)

        cells = [
            str(idx),
            trade.get('timestamp_ist', ''),
            action,
            asset,
            price,
            size,
            f"**{signed_money(pnl)}**",
            f"**{signed_pct(ret)}**",
            f"${balance:,.2f}",
            signed_pct(total_ret),
        ]
        if include_drawdown:
            cells.append(f"{trade.get('max_dd', 0.0) * 100.0:.2f}%")
        if include_leverage:
            cells.append(f"{trade.get('leverage', 0.0):.1f}x")
        cells.append(result_label(action, pnl))
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def render_strategy(number, title, model, state, include_drawdown=False, include_leverage=False):
    trades = state.get('trade_history', [])
    current_value = state.get('portfolio_value', 10000.0)
    peak_value = state.get('peak_equity', 10000.0)
    total_return = ((current_value - 10000.0) / 10000.0) * 100.0
    action = state.get('last_action', '100% CASH (Neutral)')

    lines = [
        f"## {title.format(number=number)}",
        f"* **Model:** {model}",
        "* **Initial Capital:** `$10,000.00`",
        f"* **Current Portfolio Value:** `${current_value:,.2f}`",
    ]
    if include_drawdown:
        lines.append(f"* **Peak Equity (HWM):** `${peak_value:,.2f}`")
    lines.extend([
        f"* **Net Return:** `{total_return:+.2f}%`",
        f"* **Current Status:** `{action}`",
        "",
        "### Trade History Log",
    ])

    headers = [
        "#", "Timestamp (IST)", "Action / Position", "Asset",
        "Entry / Exit Price", "Size (%)", "Period PnL ($)",
        "Period Return", "Portfolio Balance", "Total Return",
    ]
    separators = ["---"] * len(headers)
    if include_drawdown:
        headers.append("Max DD")
        separators.append("---")
    if include_leverage:
        headers.append("Leverage")
        separators.append("---")
    headers.append("Result")
    separators.append("---")

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(separators) + "|")
    lines.extend(render_trade_rows(
        trades,
        include_drawdown=include_drawdown,
        include_leverage=include_leverage,
    ))
    lines.extend(["", "---", ""])
    return lines


def generate_markdown(adaptive_state, v3_state, trend_state, output_path):
    lines = [
        "# 📊 Institutional Paper Trading Execution History",
        "",
        "> **Automated 24/7 Trading Engine**  ",
        "> Tracks execution history, position sizing, leverage, PnL, and drawdowns separately for each deployed strategy.",
        "",
        "---",
        "",
    ]

    # Keep this order identical to the web dashboard: S1 Adaptive, S2 v3, S3 TrendVol.
    lines.extend(render_strategy(
        1,
        "🔵 Strategy {number}: Adaptive Weighted Strategy (Base Adaptive Model)",
        "LiquidityAlpha + Dynamic Confidence Capping + Volatility Switch Leverage (max 4x) + 2% No-Trade Band",
        adaptive_state,
        include_drawdown=True,
        include_leverage=True,
    ))
    lines.extend(render_strategy(
        2,
        "🚀 Strategy {number}: Adaptive v3 Strategy (Hyper-Adaptive Optimizer)",
        "50d Trend Score (15 assets) + BTC 150d Regime Filter + Square-Root Lasso (λ=0.05) + Elastic Hysteresis + 2% No-Trade Band + Ledoit-Wolf (60d) 15% Vol Target",
        v3_state,
        include_drawdown=True,
    ))
    lines.extend(render_strategy(
        3,
        "📉 Strategy {number}: TrendVol (Redesigned Signal)",
        "Long-only 50-day time-series momentum on BTC + ETH, inverse-vol sizing, 15% volatility target, 2% No-Trade Band",
        trend_state,
        include_drawdown=True,
    ))

    lines.extend([
        "## ⚙️ Strategy Comparison & Specifications",
        "",
        "| Feature | Adaptive Weighted (Base) | Adaptive v3 (Hyper-Adaptive) | TrendVol (Redesigned) |",
        "|---|---|---|---|",
        "| **Signal** | LiquidityAlpha | 50d trend score across 15 assets + BTC 150d regime | 50d time-series momentum on BTC + ETH |",
        "| **Turnover Control** | 2% no-trade band + drift-aware turnover | Elastic Hysteresis + 2% no-trade band | 2% no-trade band + drift-aware turnover |",
        "| **Risk Architecture** | Volatility Switch (max 4x), 365-day annualisation, HWM drawdown tracking | 60d Ledoit-Wolf + 15% vol target, max 2x + BTC 150d regime | Inverse-vol + 15% vol target, max 2x, HWM drawdown tracking, drawdown overlay |",
        "| **Position Sizing** | Proportional to signal | Probabilistic (5%-50% per asset) | Inverse-vol, gross normalised |",
        "| **Shorting** | Yes, 3 bp/day financing | Yes (cash below BTC 150d SMA), 3 bp/day financing | No (long or cash) |",
        "| **Universe** | 9 assets | 15 assets | 2 assets (BTC, ETH) |",
        "| **Out-of-sample** | Loses after look-ahead fix; cost controls improve net, not the signal | 2024-26: +21.1%/yr, Sharpe 1.57 CAGR/vol (1.49 mean/std), 10.5% MDD (fixed 15); PIT liquidity universe: +11.3%/yr, 0.81 Sharpe | 2024-26 test: +16.8%/yr, Sharpe 0.98, MDD 16.4% (no overlay); +10.2%/yr, Sharpe 0.76, MDD 13.5% (causal overlay). Naive t≈1.7 — not statistically significant. |",
        "| **Execution Cost Model** | Stealth Iceberg (0.05% + 0.014% slip) + 3 bp/day short financing | Stealth Iceberg (0.05% + 0.014% slip) + 3 bp/day short financing | Stealth Iceberg (0.05% + 0.014% slip) |",
        "",
        "*v3 backtest note:* the fixed 15-asset result is survivorship-biased because those assets survived to today. A point-in-time top-15-by-liquidity backtest, which includes assets that later collapsed, gives +11.3%/yr and 0.81 Sharpe in 2024-26; its 95% block-bootstrap Sharpe interval includes zero. Treat the fixed-15 figure as an upper bound, not a guarantee. The test window was also observed during tuning, so the edge is not statistically overwhelming after multiple-testing adjustment.",
        "*TrendVol note:* an earlier 15-asset backtest reported +58%/yr and Sharpe 2.55. That figure was inflated by look-ahead in the drawdown overlay and is retracted. The corrected causal result for 15 assets is +28%/yr in-sample but only +4.9%/yr, Sharpe 0.29, out-of-sample; the 15-asset expansion is therefore rejected. The live TrendVol runner trades BTC+ETH. Out-of-sample BTC+ETH is +16.8%/yr, Sharpe 0.98 without the overlay, or +10.2%/yr, Sharpe 0.76 with it. Neither is statistically strong (naive t≈1.7). Forward paper-trading evidence is still limited.",
        "",
    ])

    with open(output_path, 'w', encoding='utf-8', newline='\n') as handle:
        handle.write("\n".join(lines))


def load_state(path):
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    adaptive_state = load_state(os.path.join(base_dir, 'paper_state_adaptive.json'))
    v3_state = load_state(os.path.join(base_dir, 'paper_state_v3.json'))
    trend_state = load_state(os.path.join(base_dir, 'paper_state_trend.json'))
    out_path = os.path.abspath(os.path.join(base_dir, '..', 'TRADE_HISTORY.md'))

    generate_markdown(adaptive_state, v3_state, trend_state, out_path)
    print(f"Successfully generated {out_path}")
