"""
================================================================================
AMD STRATEGY — PURE OUT-OF-SAMPLE: HighVol + Sideways Only
One full year of UNSEEN data, no training, fees + slippage + funding included,
10,000-iteration Monte Carlo.
================================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
import time

from amd_futures_backtest import (
    monte_carlo_simulation, _compute_trade_pnls,
    FUTURES_ASSETS, INITIAL_CAPITAL, RISK_PER_TRADE_PCT,
    MAKER_FEE, TAKER_FEE, SLIPPAGE_BPS, FUNDING_RATE_8H,
    fetch_history, amd_signals
)


def detect_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Detect market regime: Bull, Bear, Sideways, HighVol, LowVol."""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    sma_200 = close.rolling(200, min_periods=1).mean()

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean()
    atr_pct = atr / close

    atr_percentile = atr_pct.rolling(252, min_periods=1).rank(pct=True) * 100

    price_vs_sma = (close - sma_200) / sma_200

    feat = pd.DataFrame(index=df.index)
    feat["sma_200"] = sma_200
    feat["atr_percentile"] = atr_percentile
    feat["price_vs_sma200"] = price_vs_sma

    regime = pd.Series("Sideways", index=df.index)

    bull = (price_vs_sma > 0.02) & (atr_percentile <= 80)
    bear = (price_vs_sma < -0.02) & (atr_percentile <= 80)
    high_vol = atr_percentile > 80
    low_vol = (atr_percentile < 20) & (price_vs_sma.abs() <= 0.02)
    sideways = ~(bull | bear | high_vol | low_vol)

    regime[bull] = "Bull"
    regime[bear] = "Bear"
    regime[high_vol] = "HighVol"
    regime[low_vol] = "LowVol"
    regime[sideways] = "Sideways"

    feat["regime"] = regime
    return feat


def run_backtest_on_df(df: pd.DataFrame, sig: pd.DataFrame,
                       initial_capital: float,
                       risk_per_trade_pct: float,
                       taker_fee: float, slippage_bps: float,
                       funding_rate_8h: float) -> dict:
    """Event-driven backtest on a given dataframe + signals."""
    df = df.loc[sig.index]
    close = df["close"]

    equity = initial_capital
    position_shares = 0.0
    position_side = 0
    position_entry = 0.0
    position_notional = 0.0
    sl = 0.0
    tp = 0.0
    partial_taken = False

    equity_curve = []
    trade_log = []
    daily_pnl = 0.0
    last_date = None

    for i in range(1, len(df)):
        ts = df.index[i]
        px = float(close.iloc[i])

        if last_date is not None and ts.date() != last_date.date():
            daily_pnl = 0.0
            partial_taken = False
        last_date = ts

        position_value = position_shares * px if position_shares != 0 else 0.0
        total_equity = equity + position_value

        # --- Exit management ---
        exit_reason = None
        if position_shares != 0:
            if position_side > 0:
                if px <= sl:
                    exit_reason = "STOP_LOSS"
                elif px >= tp and not partial_taken:
                    exit_reason = "TAKE_PROFIT"

                if not partial_taken:
                    partial_px = position_entry + abs(position_entry - sl) * 1.5
                    if px >= partial_px:
                        close_qty = position_shares * 0.5
                        exit_notional = close_qty * px
                        cost = exit_notional * (taker_fee + slippage_bps / 10000)
                        realized = exit_notional - cost - (close_qty * position_entry)
                        equity += realized
                        daily_pnl += realized
                        trade_log.append({
                            "timestamp": ts, "side": "PARTIAL_TP",
                            "price": px, "notional": exit_notional, "cost": cost,
                            "qty": close_qty, "equity_after": equity,
                            "reason": "PARTIAL_TP_1.5R"
                        })
                        position_shares -= close_qty
                        partial_taken = True
                        sl = position_entry

                s_now = float(sig["signal"].iloc[i])
                if s_now != 0 and np.sign(s_now) != position_side:
                    exit_reason = "SIGNAL_FLIP"

            else:
                if px >= sl:
                    exit_reason = "STOP_LOSS"
                elif px <= tp and not partial_taken:
                    exit_reason = "TAKE_PROFIT"

                if not partial_taken:
                    partial_px = position_entry - abs(position_entry - sl) * 1.5
                    if px <= partial_px:
                        close_qty = position_shares * 0.5
                        exit_notional = close_qty * px
                        cost = exit_notional * (taker_fee + slippage_bps / 10000)
                        realized = (close_qty * position_entry) - (exit_notional + cost)
                        equity += realized
                        daily_pnl += realized
                        trade_log.append({
                            "timestamp": ts, "side": "PARTIAL_TP",
                            "price": px, "notional": exit_notional, "cost": cost,
                            "qty": close_qty, "equity_after": equity,
                            "reason": "PARTIAL_TP_1.5R"
                        })
                        position_shares -= close_qty
                        partial_taken = True
                        sl = position_entry

                s_now = float(sig["signal"].iloc[i])
                if s_now != 0 and np.sign(s_now) != position_side:
                    exit_reason = "SIGNAL_FLIP"

            if exit_reason is not None:
                exit_notional = position_shares * px
                cost = exit_notional * (taker_fee + slippage_bps / 10000)
                if position_side > 0:
                    realized = exit_notional - cost - (position_shares * position_entry)
                else:
                    realized = (position_shares * position_entry) - (exit_notional + cost)
                equity += realized
                daily_pnl += realized
                trade_log.append({
                    "timestamp": ts, "side": "EXIT",
                    "price": px, "notional": exit_notional, "cost": cost,
                    "qty": position_shares, "equity_after": equity, "reason": exit_reason
                })
                position_shares = 0.0
                position_side = 0
                position_entry = 0.0
                sl = 0.0
                tp = 0.0
                partial_taken = False

        # --- Entry ---
        s = float(sig["signal"].iloc[i])
        if s != 0 and position_shares == 0:
            entry_px = px
            entry_sl = float(sig["stop_loss"].iloc[i]) if not np.isnan(sig["stop_loss"].iloc[i]) else px
            entry_tp = float(sig["take_profit"].iloc[i]) if not np.isnan(sig["take_profit"].iloc[i]) else px

            if (s > 0 and entry_sl >= entry_px) or (s < 0 and entry_sl <= entry_px):
                equity_curve.append(total_equity)
                continue

            risk_amount = total_equity * risk_per_trade_pct
            price_risk = abs(entry_px - entry_sl)
            if price_risk <= 0:
                equity_curve.append(total_equity)
                continue

            notional = risk_amount / price_risk * entry_px
            cap_notional = total_equity * 0.20
            notional = min(notional, cap_notional)
            if notional <= 0:
                equity_curve.append(total_equity)
                continue

            cost = notional * (taker_fee + slippage_bps / 10000)
            equity -= cost

            shares = notional / entry_px
            position_shares = shares
            position_side = int(s)
            position_entry = entry_px
            position_notional = notional
            sl = entry_sl
            tp = entry_tp
            partial_taken = False

            trade_log.append({
                "timestamp": ts,
                "side": "BUY" if s > 0 else "SELL",
                "price": entry_px, "notional": notional, "cost": cost,
                "qty": shares, "sl": sl, "tp": tp,
                "confluence": float(sig["confluence"].iloc[i]),
                "equity_after": equity
            })

        # Funding
        if position_shares != 0:
            funding = abs(position_notional) * funding_rate_8h / 3
            equity -= funding
            daily_pnl -= funding

        total_equity = equity + (position_shares * px if position_shares != 0 else 0.0)
        equity_curve.append(total_equity)

    equity_curve = pd.Series(equity_curve, index=df.index[1:])
    return equity_curve, trade_log


def compute_metrics_from_curve(equity_curve: pd.Series, trade_log: list[dict],
                               initial_capital: float) -> dict:
    if equity_curve is None or len(equity_curve) == 0:
        return {"error": "No equity data"}

    returns_ = equity_curve.pct_change().dropna()
    n = len(returns_)
    if n == 0:
        return {"error": "No returns"}

    periods_per_year = 252
    mu = np.mean(returns_)
    sigma = np.std(returns_, ddof=1)
    sharpe = (mu * periods_per_year) / (sigma * np.sqrt(periods_per_year) + 1e-8)

    peaks = np.maximum.accumulate(equity_curve.values)
    dd_series = (peaks - equity_curve.values) / (peaks + 1e-8)
    max_dd = float(np.max(dd_series))

    total_return = float(equity_curve.iloc[-1] / initial_capital - 1)

    trade_pnls = _compute_trade_pnls(trade_log)
    if trade_pnls:
        trade_pnls = np.array(trade_pnls)
        win_rate = len(trade_pnls[trade_pnls > 0]) / len(trade_pnls)
        gross_wins = float(trade_pnls[trade_pnls > 0].sum())
        gross_losses = abs(float(trade_pnls[trade_pnls < 0].sum()))
        profit_factor = gross_wins / (gross_losses + 1e-8)
        expectancy = float(np.mean(trade_pnls))
        n_trades = len(trade_pnls)
    else:
        win_rate = 0.0
        profit_factor = 0.0
        expectancy = 0.0
        n_trades = 0

    return {
        "n_trades": n_trades,
        "total_return": total_return,
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "expectancy": float(expectancy),
    }


def run_oos_highvol_sideways(symbol: str,
                             initial_capital: float = INITIAL_CAPITAL,
                             risk_per_trade_pct: float = RISK_PER_TRADE_PCT,
                             rr_target: float = 3.0,
                             lookback_years: str = "2y") -> dict | None:
    """
    Pure out-of-sample test: fetch 2 years, use the SECOND year (most recent)
    as completely unseen data. Only trade HighVol and Sideways regimes.
    """
    try:
        df = fetch_history(symbol, period=lookback_years, interval="1h")
    except Exception as e:
        print(f"  [skip] {symbol}: {e}")
        return None

    # Detect regimes on full data
    regime_feat = detect_regime(df)
    df = df.join(regime_feat[["regime", "atr_percentile", "price_vs_sma200"]])
    df = df.dropna(subset=["regime"])

    # Generate signals on full data (rule-based, no training)
    sig = amd_signals(df, rr_target=rr_target)

    # --- PURE OUT-OF-SAMPLE: Use only the second half (most recent year) ---
    split_idx = len(df) // 2
    oos_df = df.iloc[split_idx:].copy()
    oos_sig = sig.iloc[split_idx:]

    # Filter to HighVol and Sideways only
    allowed_regimes = ["HighVol", "Sideways"]
    oos_df = oos_df[oos_df["regime"].isin(allowed_regimes)]
    oos_sig = oos_sig[oos_sig.index.isin(oos_df.index)]

    print(f"    Full data: {len(df):,} bars ({df.index[0]} to {df.index[-1]})")
    print(f"    OOS period: {len(oos_df):,} bars ({oos_df.index[0]} to {oos_df.index[-1]})")
    print(f"    Regimes traded: HighVol + Sideways only")

    # Run backtest on OOS filtered data
    equity_curve, trade_log = run_backtest_on_df(
        oos_df, oos_sig,
        initial_capital=initial_capital,
        risk_per_trade_pct=risk_per_trade_pct,
        taker_fee=TAKER_FEE,
        slippage_bps=SLIPPAGE_BPS,
        funding_rate_8h=FUNDING_RATE_8H,
    )

    m = compute_metrics_from_curve(equity_curve, trade_log, initial_capital)

    # Monte Carlo on OOS trades
    mc = monte_carlo_simulation(
        trade_log,
        initial_capital=initial_capital,
        iterations=10000,
        num_trades_per_sim=max(50, m["n_trades"] * 2),
        risk_per_trade_pct=risk_per_trade_pct,
    )

    # Regime breakdown within OOS
    hv_mask = oos_df["regime"] == "HighVol"
    sw_mask = oos_df["regime"] == "Sideways"

    hv_trades = [t for t in trade_log if t["timestamp"] in oos_df[hv_mask].index]
    sw_trades = [t for t in trade_log if t["timestamp"] in oos_df[sw_mask].index]

    hv_m = compute_metrics_from_curve(
        equity_curve.loc[[t["timestamp"] for t in hv_trades]] if hv_trades else pd.Series(),
        hv_trades, initial_capital
    ) if hv_trades else {"n_trades": 0, "total_return": 0, "sharpe": 0, "max_drawdown": 0,
                         "win_rate": 0, "profit_factor": 0, "expectancy": 0}

    sw_m = compute_metrics_from_curve(
        equity_curve.loc[[t["timestamp"] for t in sw_trades]] if sw_trades else pd.Series(),
        sw_trades, initial_capital
    ) if sw_trades else {"n_trades": 0, "total_return": 0, "sharpe": 0, "max_drawdown": 0,
                         "win_rate": 0, "profit_factor": 0, "expectancy": 0}

    return {
        "symbol": symbol,
        "oos_bars": len(oos_df),
        "oos_start": str(oos_df.index[0]),
        "oos_end": str(oos_df.index[-1]),
        "hv_bars": int(hv_mask.sum()),
        "sw_bars": int(sw_mask.sum()),
        "overall": m,
        "monte_carlo": mc,
        "hv_metrics": hv_m,
        "sw_metrics": sw_m,
        "equity_curve": equity_curve,
    }


def print_report(results: dict):
    """Print formatted report."""
    print(f"\n{'='*70}")
    print(f" PURE OUT-OF-SAMPLE — {results['symbol']}")
    print(f" Regimes: HighVol + Sideways Only | No Training")
    print(f"{'='*70}")
    print(f"  OOS Period   : {results['oos_start']} to {results['oos_end']}")
    print(f"  OOS Bars     : {results['oos_bars']:,}")
    print(f"  HighVol bars : {results['hv_bars']:,}")
    print(f"  Sideways bars: {results['sw_bars']:,}")
    print()

    m = results["overall"]
    print(f"  OVERALL (HighVol + Sideways, Unseen Data):")
    print(f"    Return     : {m['total_return']*100:+.2f}%")
    print(f"    Sharpe     : {m['sharpe']:.3f}")
    print(f"    Max DD     : {m['max_drawdown']*100:.2f}%")
    print(f"    Win Rate   : {m['win_rate']*100:.2f}%")
    print(f"    Profit Fct : {m['profit_factor']:.3f}")
    print(f"    Expectancy : {m['expectancy']:+.4f} R")
    print(f"    N Trades   : {m['n_trades']}")
    print()

    mc = results["monte_carlo"]
    if "error" not in mc:
        print(f"  MONTE CARLO (10,000 iterations):")
        print(f"    Median Equity : ${mc['median_equity']:,.2f}")
        print(f"    5th Pctile    : ${mc['p5_equity']:,.2f}")
        print(f"    95th Pctile   : ${mc['p95_equity']:,.2f}")
        print(f"    Median Max DD : {mc['median_max_dd']*100:.2f}%")
        print(f"    Risk of Ruin  : {mc['risk_of_ruin_pct']:.2f}%")
        print()

    print(f"  REGIME BREAKDOWN:")
    print(f"    {'Regime':<12} {'Trades':>7} {'Return':>10} {'Sharpe':>8} {'WR':>7} {'PF':>7}")
    print(f"    {'-'*56}")
    print(f"    {'HighVol':<12} {results['hv_metrics']['n_trades']:>7} "
          f"{results['hv_metrics']['total_return']*100:>+9.2f}% "
          f"{results['hv_metrics']['sharpe']:>8.2f} "
          f"{results['hv_metrics']['win_rate']*100:>6.1f}% "
          f"{results['hv_metrics']['profit_factor']:>6.2f}")
    print(f"    {'Sideways':<12} {results['sw_metrics']['n_trades']:>7} "
          f"{results['sw_metrics']['total_return']*100:>+9.2f}% "
          f"{results['sw_metrics']['sharpe']:>8.2f} "
          f"{results['sw_metrics']['win_rate']*100:>6.1f}% "
          f"{results['sw_metrics']['profit_factor']:>6.2f}")


def main():
    print("=" * 70)
    print(" AMD STRATEGY — PURE OUT-OF-SAMPLE (HighVol + Sideways Only)")
    print(" One Full Year UNSEEN Data | No Training | Fees + Slippage + Funding")
    print(" Monte Carlo: 10,000 iterations")
    print("=" * 70)

    for symbol in FUTURES_ASSETS:
        print(f"\nAnalyzing {symbol}...")
        results = run_oos_highvol_sideways(symbol)
        if results is None:
            continue
        print_report(results)


if __name__ == "__main__":
    main()
