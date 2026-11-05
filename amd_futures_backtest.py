"""
================================================================================
AMD (Accumulation, Manipulation, Distribution) Perpetual Futures Backtest Engine
Real-market-data version with fees, slippage, funding, and 10,000-iteration
Monte Carlo simulation.

Uses yfinance for OHLCV data (free, no API key required).
================================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
import time

# --------------------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------------------
INITIAL_CAPITAL = 10000.0
RISK_PER_TRADE_PCT = 0.02          # 2% equity per trade ($1R)

# VIP 0 Standard Perpetual Futures Rates
MAKER_FEE = 0.0002                 # 0.02%
TAKER_FEE = 0.0005                 # 0.05%
SLIPPAGE_BPS = 1.0                 # 1 basis point slippage per side
FUNDING_RATE_8H = 0.0001           # avg funding per 8h

FUTURES_ASSETS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"
]

# Trade setup parameters
RR_TARGET = 3.0                    # 1:3 reward-to-risk
WIN_PROB = 0.46                    # baseline win rate


def fetch_history(symbol: str, period: str = "1y", interval: str = "1h",
                  retries: int = 3) -> pd.DataFrame:
    """Download historical OHLCV data for a symbol with retry."""
    last_err = None
    for attempt in range(retries):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=True)
            if df.empty:
                raise ValueError(f"No data returned for {symbol}")
            df = df.rename(columns=str.lower)
            df = df[["open", "high", "low", "close", "volume"]].copy()
            df["symbol"] = symbol
            df.index = pd.to_datetime(df.index)
            df = df.dropna()
            return df
        except Exception as e:
            last_err = e
            time.sleep(1)
    raise ValueError(f"Failed to fetch {symbol} after {retries} attempts: {last_err}")


# --------------------------------------------------------------------------
# 2. AMD / SMC STRATEGY LOGIC
# --------------------------------------------------------------------------
def detect_accumulation(df: pd.DataFrame, lookback: int = 24) -> pd.DataFrame:
    """Detect accumulation ranges: tight consolidation periods."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    roll_high = high.rolling(lookback, min_periods=1).max()
    roll_low = low.rolling(lookback, min_periods=1).min()
    roll_range_pct = (roll_high - roll_low) / (roll_low + 1e-8)

    is_accum = (
        (roll_range_pct >= 0.002) &
        (roll_range_pct <= 0.05)
    )

    feat = pd.DataFrame(index=df.index)
    feat["accum_range_high"] = roll_high
    feat["accum_range_low"] = roll_low
    feat["accum_range_pct"] = roll_range_pct
    feat["is_accumulation"] = is_accum.astype(float)
    return feat


def detect_manipulation(df: pd.DataFrame, lookback: int = 24,
                        buffer_pct: float = 0.0005) -> pd.DataFrame:
    """Detect liquidity sweeps beyond accumulation range (stop hunts)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    accum_high = df["accum_range_high"].ffill()
    accum_low = df["accum_range_low"].ffill()

    accum_high_prev = accum_high.shift(1)
    accum_low_prev = accum_low.shift(1)

    sweep_below_raw = low < (accum_low_prev * (1 - buffer_pct))
    sweep_above_raw = high > (accum_high_prev * (1 + buffer_pct))
    reclaim_above = close > accum_low_prev
    reject_below = close < accum_high_prev

    bullish_manip = sweep_below_raw.shift(1).fillna(False) & reclaim_above
    bearish_manip = sweep_above_raw.shift(1).fillna(False) & reject_below

    feat = pd.DataFrame(index=df.index)
    feat["sweep_below"] = sweep_below_raw.astype(float)
    feat["sweep_above"] = sweep_above_raw.astype(float)
    feat["bullish_manipulation"] = bullish_manip.astype(float)
    feat["bearish_manipulation"] = bearish_manip.astype(float)
    feat["accum_high"] = accum_high
    feat["accum_low"] = accum_low
    return feat


def detect_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Detect fair value gaps and order blocks for distribution entries."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    open_ = df["open"]

    fvg_bullish = pd.Series(False, index=df.index)
    fvg_bearish = pd.Series(False, index=df.index)
    fvg_top = pd.Series(np.nan, index=df.index)
    fvg_bottom = pd.Series(np.nan, index=df.index)

    for i in range(2, len(df)):
        if low.iloc[i - 2] > high.iloc[i]:
            fvg_bullish.iloc[i] = True
            fvg_top.iloc[i] = low.iloc[i - 2]
            fvg_bottom.iloc[i] = high.iloc[i]
        elif high.iloc[i - 2] < low.iloc[i]:
            fvg_bearish.iloc[i] = True
            fvg_top.iloc[i] = low.iloc[i]
            fvg_bottom.iloc[i] = high.iloc[i - 2]

    bullish_ob = pd.Series(False, index=df.index)
    bearish_ob = pd.Series(False, index=df.index)
    ob_high = pd.Series(np.nan, index=df.index)
    ob_low = pd.Series(np.nan, index=df.index)

    for i in range(1, len(df)):
        is_bullish_impulse = close.iloc[i] > high.iloc[i - 1]
        is_bearish_impulse = close.iloc[i] < low.iloc[i - 1]

        if is_bullish_impulse:
            j = i - 1
            while j >= max(0, i - 20):
                if close.iloc[j] < open_.iloc[j]:
                    bullish_ob.iloc[j] = True
                    ob_high.iloc[j] = max(open_.iloc[j], close.iloc[j])
                    ob_low.iloc[j] = min(open_.iloc[j], close.iloc[j])
                    break
                j -= 1

        if is_bearish_impulse:
            j = i - 1
            while j >= max(0, i - 20):
                if close.iloc[j] > open_.iloc[j]:
                    bearish_ob.iloc[j] = True
                    ob_high.iloc[j] = max(open_.iloc[j], close.iloc[j])
                    ob_low.iloc[j] = min(open_.iloc[j], close.iloc[j])
                    break
                j -= 1

    feat = pd.DataFrame(index=df.index)
    feat["fvg_bullish"] = fvg_bullish.astype(float)
    feat["fvg_bearish"] = fvg_bearish.astype(float)
    feat["fvg_top"] = fvg_top
    feat["fvg_bottom"] = fvg_bottom
    feat["ob_bullish"] = bullish_ob.astype(float)
    feat["ob_bearish"] = bearish_ob.astype(float)
    feat["ob_high"] = ob_high
    feat["ob_low"] = ob_low
    return feat


def amd_signals(df: pd.DataFrame, rr_target: float = 3.0) -> pd.DataFrame:
    """
    Generate AMD trading signals.
    Logic: Accumulation -> Manipulation (liquidity sweep) -> Distribution entry.
    """
    acc = detect_accumulation(df)
    manip = detect_manipulation(df.join(acc[["accum_range_high", "accum_range_low"]]))
    dist = detect_distribution(df)

    feat = pd.concat([df, acc, manip, dist], axis=1)
    feat = feat.dropna(subset=["open", "high", "low", "close"])
    feat["accum_range_high"] = feat["accum_range_high"].ffill()
    feat["accum_range_low"] = feat["accum_range_low"].ffill()

    sig = pd.DataFrame(index=feat.index)
    sig["signal"] = 0.0
    sig["confluence"] = 0
    sig["entry_price"] = np.nan
    sig["stop_loss"] = np.nan
    sig["take_profit"] = np.nan
    sig["rr"] = np.nan

    close = feat["close"]
    high = feat["high"]
    low = feat["low"]
    open_ = feat["open"]

    last_accum_high = np.nan
    last_accum_low = np.nan
    last_manip_dir = 0
    last_manip_idx = -1
    manip_window = 20

    for i in range(1, len(feat)):
        ts = feat.index[i]
        c = float(close.iloc[i])
        h = float(high.iloc[i])
        l = float(low.iloc[i])
        o = float(open_.iloc[i])

        is_accum = float(feat.get("is_accumulation", pd.Series(0, index=feat.index)).iloc[i])
        bull_manip = float(feat.get("bullish_manipulation", pd.Series(0, index=feat.index)).iloc[i])
        bear_manip = float(feat.get("bearish_manipulation", pd.Series(0, index=feat.index)).iloc[i])

        fvg_bull = float(feat.get("fvg_bullish", pd.Series(0, index=feat.index)).iloc[i])
        fvg_bear = float(feat.get("fvg_bearish", pd.Series(0, index=feat.index)).iloc[i])
        fvg_top = float(feat.get("fvg_top", pd.Series(0, index=feat.index)).iloc[i])
        fvg_bottom = float(feat.get("fvg_bottom", pd.Series(0, index=feat.index)).iloc[i])
        ob_bull = float(feat.get("ob_bullish", pd.Series(0, index=feat.index)).iloc[i])
        ob_bear = float(feat.get("ob_bearish", pd.Series(0, index=feat.index)).iloc[i])
        ob_h = float(feat.get("ob_high", pd.Series(0, index=feat.index)).iloc[i])
        ob_l = float(feat.get("ob_low", pd.Series(0, index=feat.index)).iloc[i])

        if is_accum > 0:
            last_accum_high = float(feat["accum_range_high"].iloc[i])
            last_accum_low = float(feat["accum_range_low"].iloc[i])
        if np.isnan(last_accum_high):
            last_accum_high = float(feat["accum_range_high"].iloc[i]) if not np.isnan(feat["accum_range_high"].iloc[i]) else np.nan
        if np.isnan(last_accum_low):
            last_accum_low = float(feat["accum_range_low"].iloc[i]) if not np.isnan(feat["accum_range_low"].iloc[i]) else np.nan

        if bull_manip > 0 or bear_manip > 0:
            last_manip_dir = 1 if bull_manip > 0 else -1
            last_manip_idx = i

        manip_recent = (i - last_manip_idx) <= manip_window and last_manip_idx >= 0

        # --- Long: bearish manipulation (swept lows) -> distribution up ---
        long_trigger = (last_manip_dir == -1) and manip_recent
        at_bullish_fvg = fvg_bull > 0 and fvg_bottom > 0 and l <= fvg_top * 1.002 and h >= fvg_bottom * 0.998
        at_bullish_ob = ob_bull > 0 and ob_l > 0 and l <= ob_l * 1.002 and h >= ob_l * 0.998

        if long_trigger and (at_bullish_fvg or at_bullish_ob):
            confluence = (1 if at_bullish_fvg else 0) + (1 if at_bullish_ob else 0)
            entry = c
            sl = last_accum_low if not np.isnan(last_accum_low) else l * 0.995
            sl = min(sl, l * 0.995)
            risk = entry - sl
            if risk > 0:
                tp = entry + risk * rr_target
                sig.iloc[i, sig.columns.get_loc("signal")] = 1.0
                sig.iloc[i, sig.columns.get_loc("confluence")] = confluence
                sig.iloc[i, sig.columns.get_loc("entry_price")] = entry
                sig.iloc[i, sig.columns.get_loc("stop_loss")] = sl
                sig.iloc[i, sig.columns.get_loc("take_profit")] = tp
                sig.iloc[i, sig.columns.get_loc("rr")] = rr_target

        # --- Short: bullish manipulation (swept highs) -> distribution down ---
        short_trigger = (last_manip_dir == 1) and manip_recent
        at_bearish_fvg = fvg_bear > 0 and fvg_top > 0 and h >= fvg_top * 0.998 and l <= fvg_bottom * 1.002
        at_bearish_ob = ob_bear > 0 and ob_h > 0 and h >= ob_h * 0.998 and l <= ob_h * 1.002

        if short_trigger and (at_bearish_fvg or at_bearish_ob):
            confluence = (1 if at_bearish_fvg else 0) + (1 if at_bearish_ob else 0)
            entry = c
            sl = last_accum_high if not np.isnan(last_accum_high) else h * 1.005
            sl = max(sl, h * 1.005)
            risk = sl - entry
            if risk > 0:
                tp = entry - risk * rr_target
                sig.iloc[i, sig.columns.get_loc("signal")] = -1.0
                sig.iloc[i, sig.columns.get_loc("confluence")] = confluence
                sig.iloc[i, sig.columns.get_loc("entry_price")] = entry
                sig.iloc[i, sig.columns.get_loc("stop_loss")] = sl
                sig.iloc[i, sig.columns.get_loc("take_profit")] = tp
                sig.iloc[i, sig.columns.get_loc("rr")] = rr_target

    return sig


# --------------------------------------------------------------------------
# 3. BACKTEST ENGINE (fees, slippage, funding included)
# --------------------------------------------------------------------------
def run_amd_backtest(symbol: str, initial_capital: float = INITIAL_CAPITAL,
                     risk_per_trade_pct: float = RISK_PER_TRADE_PCT,
                     rr_target: float = RR_TARGET,
                     maker_fee: float = MAKER_FEE,
                     taker_fee: float = TAKER_FEE,
                     slippage_bps: float = SLIPPAGE_BPS,
                     funding_rate_8h: float = FUNDING_RATE_8H) -> dict | None:
    """Run a single-symbol AMD backtest with realistic costs."""
    try:
        df = fetch_history(symbol)
    except Exception as e:
        print(f"  [skip] {symbol}: {e}")
        return None

    sig = amd_signals(df, rr_target=rr_target)
    df = df.loc[sig.index]
    close = df["close"]

    equity = initial_capital  # cash only
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
            if position_side > 0:  # long
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
                            "timestamp": ts, "symbol": symbol, "side": "PARTIAL_TP",
                            "price": px, "notional": exit_notional, "cost": cost,
                            "qty": close_qty, "equity_after": equity,
                            "reason": "PARTIAL_TP_1.5R"
                        })
                        position_shares -= close_qty
                        partial_taken = True
                        sl = position_entry  # move to breakeven

                s_now = float(sig["signal"].iloc[i])
                if s_now != 0 and np.sign(s_now) != position_side:
                    exit_reason = "SIGNAL_FLIP"

            else:  # short
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
                            "timestamp": ts, "symbol": symbol, "side": "PARTIAL_TP",
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
                    "timestamp": ts, "symbol": symbol, "side": "EXIT",
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
                "timestamp": ts, "symbol": symbol,
                "side": "BUY" if s > 0 else "SELL",
                "price": entry_px, "notional": notional, "cost": cost,
                "qty": shares, "sl": sl, "tp": tp,
                "confluence": float(sig["confluence"].iloc[i]),
                "equity_after": equity
            })

        # Funding (per 8h, prorated to hourly bars)
        if position_shares != 0:
            funding = abs(position_notional) * funding_rate_8h / 3
            equity -= funding
            daily_pnl -= funding

        total_equity = equity + (position_shares * px if position_shares != 0 else 0.0)
        equity_curve.append(total_equity)

    equity_curve = pd.Series(equity_curve, index=df.index[1:])
    metrics = compute_metrics(equity_curve, trade_log, initial_capital)

    return {
        "symbol": symbol,
        "equity_curve": equity_curve,
        "trades": trade_log,
        "metrics": metrics,
    }


# --------------------------------------------------------------------------
# 4. METRICS
# --------------------------------------------------------------------------
def _compute_trade_pnls(trades: list[dict]) -> list[float]:
    pnls = []
    i = 0
    while i < len(trades):
        t = trades[i]
        if t.get("side") in ("BUY", "SELL"):
            entry_price = t["price"]
            entry_qty = t["qty"]
            entry_side = t["side"]
            entry_cost = t.get("cost", 0.0)
            position_side = 1 if entry_side == "BUY" else -1
            total_realized = -entry_cost
            remaining_qty = entry_qty
            j = i + 1
            while j < len(trades):
                nt = trades[j]
                nside = nt.get("side", "")
                if nside == "PARTIAL_TP":
                    px = nt["price"]; qty = nt["qty"]; cost = nt.get("cost", 0.0)
                    if position_side > 0:
                        realized = (px * qty) - cost - (qty * entry_price)
                    else:
                        realized = (qty * entry_price) - (px * qty) - cost
                    total_realized += realized
                    remaining_qty -= qty
                elif nside == "EXIT":
                    px = nt["price"]; cost = nt.get("cost", 0.0)
                    if position_side > 0:
                        realized = (px * remaining_qty) - cost - (remaining_qty * entry_price)
                    else:
                        realized = (remaining_qty * entry_price) - (px * remaining_qty) - cost
                    total_realized += realized
                    remaining_qty = 0
                    j += 1
                    break
                elif nside in ("BUY", "SELL"):
                    break
                j += 1
            pnls.append(total_realized)
            i = j
        else:
            i += 1
    return pnls


def compute_metrics(equity_curve: pd.Series, trades: list[dict],
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
    downside = returns_[returns_ < 0]
    down_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-8

    ann_ret = mu * periods_per_year
    ann_vol = sigma * np.sqrt(periods_per_year)
    ann_down = down_std * np.sqrt(periods_per_year)

    sharpe = ann_ret / (ann_vol + 1e-8)
    sortino = ann_ret / (ann_down + 1e-8)

    peaks = np.maximum.accumulate(equity_curve.values)
    dd_series = (peaks - equity_curve.values) / (peaks + 1e-8)
    max_dd = float(np.max(dd_series))
    calmar = ann_ret / (max_dd + 1e-8)

    trade_pnls = _compute_trade_pnls(trades)
    if trade_pnls:
        trade_pnls = np.array(trade_pnls)
        wins = trade_pnls[trade_pnls > 0]
        losses = trade_pnls[trade_pnls < 0]
        win_rate = len(wins) / len(trade_pnls)
        gross_wins = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_losses = float(abs(losses.sum())) if len(losses) > 0 else 0.0
        profit_factor = gross_wins / (gross_losses + 1e-8)
        expectancy = float(np.mean(trade_pnls))
        n_trades = len(trade_pnls)
    else:
        win_rate = 0.0
        profit_factor = 0.0
        expectancy = 0.0
        n_trades = 0

    var_95 = float(np.percentile(returns_, 5))
    cvar_95 = float(np.mean(returns_[returns_ <= var_95])) if np.any(returns_ <= var_95) else var_95
    total_return = float(equity_curve.iloc[-1] / initial_capital - 1)

    return {
        "initial_capital": float(initial_capital),
        "final_equity": float(equity_curve.iloc[-1]),
        "total_return": total_return,
        "annualized_return": float(ann_ret),
        "annualized_volatility": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_drawdown": max_dd,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "expectancy": float(expectancy),
        "n_trades": int(n_trades),
    }


# --------------------------------------------------------------------------
# 5. MONTE CARLO (10,000 iterations)
# --------------------------------------------------------------------------
def monte_carlo_simulation(trades: list[dict], initial_capital: float,
                           iterations: int = 10000, num_trades_per_sim: int = 200,
                           risk_per_trade_pct: float = RISK_PER_TRADE_PCT) -> dict:
    trade_pnls = _compute_trade_pnls(trades)
    if len(trade_pnls) == 0:
        return {"error": "No trades to simulate"}

    trade_pnls = np.array(trade_pnls)
    avg_risk = np.mean(np.abs(trade_pnls)) + 1e-8
    r_multiples = trade_pnls / avg_risk

    final_balances = np.zeros(iterations)
    max_drawdowns = np.zeros(iterations)

    for i in range(iterations):
        sampled_r = np.random.choice(r_multiples, size=num_trades_per_sim, replace=True)
        equity = np.zeros(num_trades_per_sim + 1)
        equity[0] = initial_capital
        for t in range(num_trades_per_sim):
            risk_amt = equity[t] * risk_per_trade_pct
            pnl = risk_amt * sampled_r[t]
            equity[t + 1] = max(0.0, equity[t] + pnl)
        final_balances[i] = equity[-1]
        peaks = np.maximum.accumulate(equity)
        dds = (peaks - equity) / (peaks + 1e-8)
        max_drawdowns[i] = np.max(dds)

    median_equity = float(np.median(final_balances))
    p5_equity = float(np.percentile(final_balances, 5))
    p95_equity = float(np.percentile(final_balances, 95))
    median_dd = float(np.median(max_drawdowns))
    p95_dd = float(np.percentile(max_drawdowns, 95))
    risk_of_ruin = float((max_drawdowns >= 0.50).mean() * 100)

    return {
        "iterations": iterations,
        "trades_per_sim": num_trades_per_sim,
        "initial_capital": float(initial_capital),
        "median_equity": median_equity,
        "p5_equity": p5_equity,
        "p95_equity": p95_equity,
        "median_max_dd": median_dd,
        "p95_max_dd": p95_dd,
        "risk_of_ruin_pct": risk_of_ruin,
        "expectancy_dollars_per_trade": float(np.mean(trade_pnls)),
        "expectancy_r_per_trade": float(np.mean(r_multiples)),
        "win_rate_pct": float((trade_pnls > 0).mean() * 100),
        "profit_factor": float(trade_pnls[trade_pnls > 0].sum() / (abs(trade_pnls[trade_pnls < 0].sum()) + 1e-8)),
        "final_balances": final_balances,
        "max_drawdowns": max_drawdowns,
    }


# --------------------------------------------------------------------------
# 6. MAIN
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print(" AMD PERPETUAL FUTURES BACKTEST (Real Market Data)")
    print(" Fees + Slippage + Funding | 10,000-Iteration Monte Carlo")
    print("=" * 60)

    for symbol in FUTURES_ASSETS:
        print(f"\n--- {symbol} ---")
        result = run_amd_backtest(symbol)
        if result is None:
            continue
        m = result["metrics"]
        print(f"  Initial Capital : ${m['initial_capital']:,.2f}")
        print(f"  Final Equity    : ${m['final_equity']:,.2f}")
        print(f"  Total Return    : {m['total_return']*100:+.2f}%")
        print(f"  Sharpe Ratio    : {m['sharpe']:.3f}")
        print(f"  Max Drawdown    : {m['max_drawdown']*100:.2f}%")
        print(f"  Win Rate        : {m['win_rate']*100:.2f}%")
        print(f"  Profit Factor   : {m['profit_factor']:.3f}")
        print(f"  N Trades        : {m['n_trades']}")

        mc = monte_carlo_simulation(
            result["trades"],
            initial_capital=INITIAL_CAPITAL,
            iterations=10000,
            num_trades_per_sim=max(100, m["n_trades"] * 2),
            risk_per_trade_pct=RISK_PER_TRADE_PCT,
        )
        if "error" not in mc:
            print(f"  --- Monte Carlo (10,000 runs) ---")
            print(f"  Median Equity   : ${mc['median_equity']:,.2f}")
            print(f"  5th Pctile      : ${mc['p5_equity']:,.2f}")
            print(f"  95th Pctile     : ${mc['p95_equity']:,.2f}")
            print(f"  Median Max DD   : {mc['median_max_dd']*100:.2f}%")
            print(f"  Risk of Ruin    : {mc['risk_of_ruin_pct']:.2f}%")
