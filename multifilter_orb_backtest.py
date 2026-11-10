"""Python port of the Multi-Filter Opening Range Breakout Pine Script.

Ported logic (approximated for backtesting):
- Instrument: NQ=F futures (MNQ price-equivalent; $20/point per NQ contract).
- Timeframe: 5-minute bars, America/New_York regular session.
- Opening range: 09:30-10:00 ET (orbHigh/orbLow).
- Entry window: 10:00-15:55 ET.
- Max 2 trades per NY day.
- Filters: HTF 60m EMA(200), local EMA(50), VWAP, RSI(14)>50/<50, ADX(14)>=18,
  relative volume >= 1.0 x 20-bar average.
- Entry on close of breakout bar (process_orders_on_close).
- Stop = 1.5 * ATR(14) points; target = 2R.
- Daily loss limit: 2% of day-start equity.
- Commission: 0.05% of trade value; slippage: 2 ticks (0.5 points).
- Starting capital: $100,000; fixed 1 contract per trade.
"""

from __future__ import annotations

import time

import pandas as pd
import numpy as np
import yfinance as yf


SYMBOL = "NQ=F"
INTERVAL = "5m"
PERIOD = "60d"
TZ = "America/New_York"
POINT_VALUE = 20.0  # USD per NQ point per contract
STARTING_CAPITAL = 100_000.0
COMMISSION_PCT = 0.0005  # 0.05%
SLIPPAGE_POINTS = 0.5  # 2 ticks of 0.25


def download() -> pd.DataFrame:
    last_err = None
    for attempt in range(5):
        try:
            raw = yf.download(
                SYMBOL,
                period=PERIOD,
                interval=INTERVAL,
                auto_adjust=False,
                progress=False,
                prepost=False,
            )
            if not raw.empty:
                break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(3)
    else:
        raise RuntimeError(f"no data: {last_err}")
    if raw.empty:
        raise RuntimeError("no data")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy().dropna()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(TZ)
    # Regular cash session only
    df = df.between_time("09:30", "16:00")
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Local EMA 50
    d["ema50"] = d["Close"].ewm(span=50, adjust=False).mean()
    # VWAP (typical price * volume, cumulative per day)
    tp = (d["High"] + d["Low"] + d["Close"]) / 3.0
    d["vwap"] = (tp * d["Volume"]).groupby(d.index.date).cumsum() / d[
        "Volume"
    ].groupby(d.index.date).cumsum()
    # RSI 14
    delta = d["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss
    d["rsi"] = 100 - (100 / (1 + rs))
    # ATR 14 (Wilder-style via EMA of TR)
    prev_close = d["Close"].shift(1)
    tr = pd.concat(
        [
            d["High"] - d["Low"],
            (d["High"] - prev_close).abs(),
            (d["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    # Relative volume vs 20-bar SMA
    d["vol_avg20"] = d["Volume"].rolling(20).mean()
    d["rel_vol"] = d["Volume"] / d["vol_avg20"]
    # ADX 14 (Wilder DMI)
    prev_high = d["High"].shift(1)
    prev_low = d["Low"].shift(1)
    up_move = d["High"] - prev_high
    down_move = prev_low - d["Low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_tr = d["atr"]
    plus_di = 100 * pd.Series(plus_dm, index=d.index).ewm(alpha=1 / 14, adjust=False).mean() / atr_tr
    minus_di = 100 * pd.Series(minus_dm, index=d.index).ewm(alpha=1 / 14, adjust=False).mean() / atr_tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    d["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    # HTF 60m EMA 200 aligned forward (non-repainting: use last completed 60m bar)
    htf = (
        d[["Open", "High", "Low", "Close", "Volume"]]
        .resample("60min", label="right", closed="right")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
    )
    htf["ema200"] = htf["Close"].ewm(span=200, adjust=False).mean()
    htf_close = htf["Close"]
    htf_ema = htf["ema200"]
    # Map each 5m bar to the most recent completed 60m bar (lookahead-free)
    htf_index = htf.index
    d["htf_close"] = np.nan
    d["htf_ema"] = np.nan
    # For each 5m timestamp, find the latest 60m bar <= that timestamp
    pos = htf_index.searchsorted(d.index, side="right") - 1
    valid = pos >= 0
    d.loc[valid, "htf_close"] = htf_close.values[pos[valid]]
    d.loc[valid, "htf_ema"] = htf_ema.values[pos[valid]]
    return d


def backtest(df: pd.DataFrame, reward_risk: float = 2.0) -> tuple[pd.DataFrame, dict]:
    d = compute_indicators(df)
    capital = STARTING_CAPITAL
    equity = capital
    trades: list[dict] = []
    equity_curve: list[dict] = []

    days = d.groupby(d.index.date)
    for day, day_df in days:
        day_df = day_df.sort_index()
        if day_df.empty:
            continue
        day_start_equity = equity
        daily_loss_limit = day_start_equity * (1 - 0.02)
        orb_bars = day_df.between_time("09:30", "09:55")
        if len(orb_bars) < 2:
            continue
        orb_high = float(orb_bars["High"].max())
        orb_low = float(orb_bars["Low"].min())

        position = None  # dict with direction, entry, stop, target, entry_time
        trades_today = 0

        prev_close = None
        for ts, bar in day_df.iterrows():
            close = float(bar["Close"])
            high = float(bar["High"])
            low = float(bar["Low"])
            vol = float(bar["Volume"])
            atr = float(bar["atr"]) if pd.notna(bar["atr"]) else np.nan
            rsi = float(bar["rsi"]) if pd.notna(bar["rsi"]) else np.nan
            adx = float(bar["adx"]) if pd.notna(bar["adx"]) else np.nan
            rel_vol = float(bar["rel_vol"]) if pd.notna(bar["rel_vol"]) else np.nan
            htf_close = float(bar["htf_close"]) if pd.notna(bar["htf_close"]) else np.nan
            htf_ema = float(bar["htf_ema"]) if pd.notna(bar["htf_ema"]) else np.nan
            vwap = float(bar["vwap"]) if pd.notna(bar["vwap"]) else np.nan
            ema50 = float(bar["ema50"]) if pd.notna(bar["ema50"]) else np.nan

            in_trade_window = ts.time() >= pd.Timestamp("10:00").time() and ts.time() <= pd.Timestamp("15:55").time()
            in_orb_window = ts.time() <= pd.Timestamp("09:55").time()

            # Manage open position: check stop/target on this bar (close-processing convention)
            if position is not None:
                dir_ = position["direction"]
                entry = position["entry"]
                stop = position["stop"]
                target = position["target"]
                exit_price = None
                reason = ""
                if dir_ == "long":
                    if low <= stop:
                        exit_price = stop
                        reason = "stop"
                    elif high >= target:
                        exit_price = target
                        reason = "target"
                else:
                    if high >= stop:
                        exit_price = stop
                        reason = "stop"
                    elif low <= target:
                        exit_price = target
                        reason = "target"

                # End-of-window flat
                if exit_price is None and ts.time() >= pd.Timestamp("15:55").time():
                    exit_price = close
                    reason = "window_close"

                if exit_price is not None:
                    gross_pts = (exit_price - entry) if dir_ == "long" else (entry - exit_price)
                    # slippage on entry + exit (2 ticks each way = 1 point total round trip in points)
                    commission_usd = (entry * 1 + exit_price * 1) * COMMISSION_PCT
                    slippage_usd = SLIPPAGE_POINTS * 2 * POINT_VALUE
                    net_pts = gross_pts - (slippage_usd / POINT_VALUE)
                    net_usd = net_pts * POINT_VALUE - commission_usd
                    equity += net_usd
                    trades.append(
                        {
                            "day": str(day),
                            "direction": dir_,
                            "entry_time": str(position["entry_time"]),
                            "exit_time": str(ts),
                            "entry": entry,
                            "exit": exit_price,
                            "stop": stop,
                            "target": target,
                            "gross_pts": gross_pts,
                            "net_pts": net_pts,
                            "net_usd": net_usd,
                            "reason": reason,
                        }
                    )
                    position = None
                    equity_curve.append({"timestamp": str(ts), "equity": equity})
                    continue

            # Entry logic (only when flat)
            if (
                position is None
                and in_trade_window
                and prev_close is not None
                and trades_today < 2
                and equity > daily_loss_limit
                and not np.isnan(atr)
                and atr > 0
            ):
                long_breakout = close > orb_high and prev_close <= orb_high
                short_breakout = close < orb_low and prev_close >= orb_low

                long_ok = long_breakout
                short_ok = short_breakout

                if not np.isnan(htf_close) and not np.isnan(htf_ema):
                    long_ok = long_ok and htf_close > htf_ema
                    short_ok = short_ok and htf_close < htf_ema
                if not np.isnan(ema50):
                    long_ok = long_ok and close > ema50
                    short_ok = short_ok and close < ema50
                if not np.isnan(vwap):
                    long_ok = long_ok and close > vwap
                    short_ok = short_ok and close < vwap
                if not np.isnan(rsi):
                    long_ok = long_ok and rsi > 50
                    short_ok = short_ok and rsi < 50
                if not np.isnan(adx):
                    long_ok = long_ok and adx >= 18
                    short_ok = short_ok and adx >= 18
                if not np.isnan(rel_vol):
                    long_ok = long_ok and rel_vol >= 1.0
                    short_ok = short_ok and rel_vol >= 1.0

                stop_pts = max(1.0, 1.5 * atr)
                target_pts = float(reward_risk) * stop_pts

                if long_ok:
                    entry = close
                    position = {
                        "direction": "long",
                        "entry": entry,
                        "stop": entry - stop_pts,
                        "target": entry + target_pts,
                        "entry_time": ts,
                    }
                    trades_today += 1
                elif short_ok:
                    entry = close
                    position = {
                        "direction": "short",
                        "entry": entry,
                        "stop": entry + stop_pts,
                        "target": entry - target_pts,
                        "entry_time": ts,
                    }
                    trades_today += 1

            prev_close = close

        equity_curve.append({"timestamp": str(day_df.index[-1]), "equity": equity})

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        wins = trades_df[trades_df["net_usd"] > 0]
        losses = trades_df[trades_df["net_usd"] < 0]
        gross_profit = wins["net_usd"].sum()
        gross_loss = abs(losses["net_usd"].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else np.inf
    else:
        gross_profit = gross_loss = 0.0
        pf = np.inf

    peak = -np.inf
    max_dd = 0.0
    for p in equity_curve:
        e = p["equity"]
        peak = max(peak, e)
        if peak > 0:
            max_dd = max(max_dd, (peak - e) / peak)

    stats = {
        "trades": len(trades_df),
        "wins": int((trades_df["net_usd"] > 0).sum()) if not trades_df.empty else 0,
        "losses": int((trades_df["net_usd"] < 0).sum()) if not trades_df.empty else 0,
        "net_usd": equity - STARTING_CAPITAL,
        "gross_profit_usd": gross_profit,
        "gross_loss_usd": gross_loss,
        "profit_factor": pf,
        "max_dd_pct": max_dd * 100,
        "final_equity": equity,
        "return_pct": (equity - STARTING_CAPITAL) / STARTING_CAPITAL * 100,
    }
    return trades_df, stats


def main() -> None:
    df = download()
    print(
        f"Downloaded {len(df)} 5m bars from {df.index.min()} to {df.index.max()} "
        f"({pd.Index(df.index.date).nunique()} sessions)"
    )
    for rr in (2.0, 2.5, 3.0):
        trades_df, stats = backtest(df, reward_risk=rr)
        print(f"\n=== R:R = 1:{rr} ===")
        print(f"  Trades:       {stats['trades']}")
        print(f"  Wins:         {stats['wins']}  Losses: {stats['losses']}")
        print(f"  Win rate:     {stats['wins'] / stats['trades'] * 100 if stats['trades'] else 0:.1f}%")
        print(f"  Gross profit: ${stats['gross_profit_usd']:+,.2f}")
        print(f"  Gross loss:   ${stats['gross_loss_usd']:+,.2f}")
        print(f"  Net P&L:      ${stats['net_usd']:+,.2f}")
        print(f"  Profit factor:{stats['profit_factor']:.2f}")
        print(f"  Max DD:       {stats['max_dd_pct']:.2f}%")
        print(f"  Return:       {stats['return_pct']:+.2f}%")
        print(f"  Final equity: ${stats['final_equity']:,.2f}")
        if not trades_df.empty:
            out = f"multifilter_orb_backtest_rr{rr}_{SYMBOL.replace('=', '_')}.csv"
            trades_df.to_csv(out, index=False)
            print(f"Wrote {out}")
    print("\nMulti-Filter ORB Backtest Results (NQ=F, 5m, 1 contract):")
    print(f"  Trades:       {stats['trades']}")
    print(f"  Wins:         {stats['wins']}  Losses: {stats['losses']}")
    print(f"  Win rate:     {stats['wins'] / stats['trades'] * 100 if stats['trades'] else 0:.1f}%")
    print(f"  Gross profit: ${stats['gross_profit_usd']:+,.2f}")
    print(f"  Gross loss:   ${stats['gross_loss_usd']:+,.2f}")
    print(f"  Net P&L:      ${stats['net_usd']:+,.2f}")
    print(f"  Profit factor:{stats['profit_factor']:.2f}")
    print(f"  Max DD:       {stats['max_dd_pct']:.2f}%")
    print(f"  Return:       {stats['return_pct']:+.2f}%")
    print(f"  Final equity: ${stats['final_equity']:,.2f}")
    if not trades_df.empty:
        out = f"multifilter_orb_backtest_{SYMBOL.replace('=', '_')}.csv"
        trades_df.to_csv(out, index=False)
        print(f"\nWrote {out}")
        print(trades_df.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
