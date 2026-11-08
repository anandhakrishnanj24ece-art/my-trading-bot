import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import yfinance as yf
import pandas as pd
import numpy as np

from neural_flow.alpha.v3_core import (AdaptiveWeightedV3, ProRiskManagerV3,
                                        trend_score, btc_regime_series)
from neural_flow.execution.model import StealthExecution
from neural_flow.risk.model import drawdown_multiplier

LIVE_DIR = Path(__file__).resolve().parent.parent
STATE_FILE = str(LIVE_DIR / 'paper_state_v3.json')
BALANCE_FILE = str(LIVE_DIR / 'paper_balance_v3.txt')
LEGACY_BALANCE_FILE = str(LIVE_DIR / 'paper_balance.txt')
SNAPSHOT_DIR = LIVE_DIR / 'price_snapshots'
BASE_CAPITAL = 10000.0

# Improved v3 configuration (validated out-of-sample).
SIGNAL_MODEL = 'trend50'
UNIVERSE_TAG = 'v3_15'
LOOKBACK = 50
REGIME_LOOKBACK = 150
TARGET_VOL = 0.15
BASE_LEVERAGE = 2.0
COV_WINDOW = 60
LAMBDA_BASE = 0.05
TAU_BASE = 0.02
CRITICAL_THRESHOLD = 3.0
MAX_ASSET_WEIGHT = 1.0
NO_TRADE_BAND = 0.02
# Conservative daily financing on the short book (perp funding/borrow).
SHORT_COST_BPS_PER_DAY = 3.0
# 150d regime + 60d covariance + 50d trend, with a small buffer.
MIN_HISTORY_BARS = max(REGIME_LOOKBACK, COV_WINDOW, LOOKBACK) + 10
# Protective drawdown overlay: de-risk from 5% drawdown, floor at 25% from 15%.
DD_CAUTION = 0.05
DD_HALT = 0.15
DD_FLOOR = 0.25


def run_adaptive_v3():
    tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD',
               'DOT-USD', 'LINK-USD', 'LTC-USD', 'DOGE-USD', 'AVAX-USD',
               'ATOM-USD', 'NEAR-USD', 'ALGO-USD', 'FIL-USD']
    print("Downloading latest market data for Adaptive v3...")
    data_raw = pd.DataFrame()
    for _ in range(3):
        try:
            data_raw = yf.download(tickers, period='1y', interval='1d', progress=False, threads=False)
            if not data_raw.empty:
                break
        except Exception:
            pass

    if data_raw.empty:
        print("Warning: Market data download returned empty. Retrying with fallback...")
        data_raw = yf.download(tickers, period='1y', interval='1d', progress=False)

    if isinstance(data_raw.columns, pd.MultiIndex):
        close_prices = data_raw['Close'].ffill().dropna(how='all')
        volumes = data_raw['Volume'].ffill().dropna(how='all')
    else:
        close_prices = pd.DataFrame({tickers[0]: data_raw['Close']}).ffill().dropna(how='all')
        volumes = pd.DataFrame({tickers[0]: data_raw['Volume']}).ffill().dropna(how='all')

    # Forward-fill real gaps only.  Do NOT back-fill: leading values belong to
    # the future and would leak information into the early signal history.
    close_prices = close_prices.dropna(axis=1, how='all').ffill()
    volumes = volumes.dropna(axis=1, how='all').ffill()
    valid_tickers = [t for t in tickers if t in close_prices.columns and t in volumes.columns]
    close_prices = close_prices[valid_tickers]
    volumes = volumes[valid_tickers]

    tickers = [t for t in close_prices.columns
               if np.isfinite(float(close_prices[t].iloc[-1]))]
    if not tickers:
        print("Warning: no ticker has a usable latest price. Skipping run.")
        return
    close_prices = close_prices[tickers]
    volumes = volumes[tickers]

    returns = close_prices.pct_change().dropna()
    common_index = returns.index.intersection(close_prices.index).intersection(volumes.index)
    returns = returns.loc[common_index]
    close_prices = close_prices.loc[common_index]
    volumes = volumes.loc[common_index]

    if len(close_prices) < MIN_HISTORY_BARS:
        print(f"Warning: only {len(close_prices)} bars available; "
              f"need at least {MIN_HISTORY_BARS}. Skipping run.")
        return

    # Candle-close safety check (Crypto daily closes at 00:00 UTC; allow run after 01:00 UTC)
    now_utc = datetime.now(timezone.utc)
    last_bar_date = close_prices.index[-1].date() if hasattr(close_prices.index[-1], 'date') else pd.to_datetime(close_prices.index[-1]).date()

    if last_bar_date == now_utc.date() and now_utc.hour < 1:
        print(f"[{now_utc.isoformat()}] Daily candle not closed yet (current hour: {now_utc.hour} UTC, last bar date: {last_bar_date}). Skipping trade.")
        return

    execution_engine = StealthExecution()

    # 1. Load state if available
    state = None
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
        except Exception:
            state = None

    # A configuration change (new signal model or universe) is a new book even
    # on the same bar, so it must be allowed to rebalance the transition.
    same_book = (state is not None
                 and state.get('signal_model') == SIGNAL_MODEL
                 and state.get('universe') == UNIVERSE_TAG)

    # Idempotency guard: never process the same daily bar twice for the same book.
    if state and same_book and state.get('last_bar_date') == str(last_bar_date):
        print(f"[{now_utc.isoformat()}] No new daily bar since last run ({last_bar_date}). Skipping.")
        return

    prev_weights_arr = None
    if same_book and 'weights_raw' in state:
        prev_weights_arr = np.array([state['weights_raw'].get(t, 0.0) for t in tickers])

    optimizer = AdaptiveWeightedV3(
        lambda_base=LAMBDA_BASE,
        tau_base=TAU_BASE,
        critical_threshold=CRITICAL_THRESHOLD,
        prev_weights=prev_weights_arr,
    )
    risk_manager = ProRiskManagerV3(
        target_vol=TARGET_VOL,
        base_leverage=BASE_LEVERAGE,
        cov_window=COV_WINDOW,
        ann=365,
    )

    # 2. Compute current signals: causal 50d trend score (in daily-vol units).
    score_df, conf_df = trend_score(close_prices, lookback=LOOKBACK)
    asset_signals = [float(x) for x in score_df.iloc[-1].reindex(tickers).fillna(0.0).tolist()]
    asset_confidences = [float(x) for x in conf_df.iloc[-1].reindex(tickers).fillna(0.0).tolist()]

    mkt_vol = returns.tail(COV_WINDOW).std().mean() * np.sqrt(365)
    weights = optimizer.optimize(np.array(asset_signals), np.array(asset_confidences), market_vol=mkt_vol)
    lev = risk_manager.get_portfolio_leverage(weights, returns)
    desired_weights = np.clip(weights * lev, -MAX_ASSET_WEIGHT, MAX_ASSET_WEIGHT)

    latest_prices = {t: float(close_prices[t].iloc[-1]) for t in tickers}

    # 3. Mark the existing book to market and recover the *actual* weights.
    # The state stores the previous target. Between runs prices move, so the
    # live book drifts. The band and the trade must use those actual weights,
    # not the old target, or the band is comparing two different things.
    peak_equity = BASE_CAPITAL
    period_return = 0.0
    short_gross_prev = 0.0
    stale_closing = 0.0
    actual_weights = np.zeros(len(tickers))
    days_held = 1
    risk_multiplier = 1.0
    risk_drawdown = 0.0

    if state is not None and 'portfolio_value' in state and 'last_prices' in state:
        prev_portfolio_val = float(state['portfolio_value'])
        prev_prices = state['last_prices']
        prev_weights = state.get('weights', {})
        # High-water mark before this period's trading.
        peak_equity = max(float(state.get('peak_equity', BASE_CAPITAL)),
                          prev_portfolio_val)

        for i, t in enumerate(tickers):
            w = float(prev_weights.get(t, 0.0))
            short_gross_prev += max(-w, 0.0)
            if t in prev_prices and float(prev_prices[t]) > 0:
                asset_ret = (latest_prices[t] - float(prev_prices[t])) / float(prev_prices[t])
                period_return += w * asset_ret

        denom = 1.0 + period_return
        if denom > 1e-9:
            for i, t in enumerate(tickers):
                w = float(prev_weights.get(t, 0.0))
                if t in prev_prices and float(prev_prices[t]) > 0:
                    actual_weights[i] = (w * (latest_prices[t] / float(prev_prices[t]))
                                         / denom)

        # Protective drawdown overlay: scale the signal target down as the
        # portfolio drifts from its high-water mark, before the no-trade band.
        risk_drawdown = (peak_equity - prev_portfolio_val) / peak_equity if peak_equity > 0 else 0.0
        risk_multiplier = drawdown_multiplier(
            peak_equity, prev_portfolio_val,
            caution=DD_CAUTION, halt=DD_HALT, floor=DD_FLOOR)
        desired_weights = np.array(desired_weights, dtype=float) * risk_multiplier

        # Positions that fell out of the download must be closed, not silently
        # dropped from the book.
        stale = [t for t in prev_weights if t not in tickers and abs(float(prev_weights[t])) > 0]
        if stale:
            stale_closing = sum(abs(float(prev_weights[t])) for t in stale)
            print(f"Warning: closing {len(stale)} stale position(s) missing from data: {', '.join(stale)}")

        try:
            # Zero on a same-bar configuration transition; >1 after a missed run.
            days_held = max(0, (last_bar_date - pd.to_datetime(state['last_bar_date']).date()).days)
        except Exception:
            days_held = 1
    else:
        # Initial run from cash: the entry trade must pay its own cost.
        prev_portfolio_val = BASE_CAPITAL
        prev_prices = {}
        period_return = 0.0

    # 4. No-trade band: compare the new target with the actual, drifted book.
    # The band only suppresses small changes in the same direction; a full
    # exit or a long/short reversal must always execute.  Applied before the
    # regime filter so a regime exit always goes to cash.
    if NO_TRADE_BAND > 0:
        band_delta = desired_weights - actual_weights
        same_direction = np.sign(desired_weights) == np.sign(actual_weights)
        material = np.abs(band_delta) >= NO_TRADE_BAND
        keep_position = same_direction & ~material & (np.abs(desired_weights) > 1e-12)
        target_weights = np.where(keep_position, actual_weights, desired_weights)
    else:
        target_weights = desired_weights

    # Portfolio-level regime filter: only carry risk when BTC is above its
    # 150-day SMA. Below it, the book goes to cash.
    regime_on = bool(btc_regime_series(close_prices, REGIME_LOOKBACK).iloc[-1])
    if not regime_on:
        target_weights = np.zeros(len(tickers))

    # 5. Charge turnover from the actual book, plus short financing on the
    # short exposure that was held over the period.
    turnover = float(np.abs(target_weights - actual_weights).sum()) + stale_closing
    trading_costs = execution_engine.calculate_cost(turnover)
    short_financing = short_gross_prev * (SHORT_COST_BPS_PER_DAY / 1e4) * days_held
    costs = trading_costs + short_financing
    current_portfolio_val = prev_portfolio_val * (1.0 + period_return - costs)
    peak_equity = max(peak_equity, current_portfolio_val)
    net_period_return = ((current_portfolio_val / prev_portfolio_val - 1.0)
                         if prev_portfolio_val > 0 else 0.0)

    total_return_pct = ((current_portfolio_val - BASE_CAPITAL) / BASE_CAPITAL) * 100.0
    max_dd = max(0.0, (peak_equity - current_portfolio_val) / peak_equity) if peak_equity > 0 else 0.0

    # 4. Generate IST Timestamp and Action Summary
    now_utc = datetime.now(timezone.utc)
    ist_zone = timezone(timedelta(hours=5, minutes=30))
    now_ist = now_utc.astimezone(ist_zone)
    ist_str = now_ist.strftime("%d %b %Y, %I:%M:%S %p IST")

    active_positions = []
    for i, t in enumerate(tickers):
        w = float(target_weights[i])
        if abs(w) > 0.01:
            side = "LONG" if w > 0 else "SHORT"
            active_positions.append(f"{side} {t} ({abs(w)*100:.1f}%) @ ${latest_prices[t]:,.2f}")

    if not regime_on:
        action_summary = f"100% CASH (BTC below {REGIME_LOOKBACK}d SMA)"
    else:
        action_summary = ", ".join(active_positions) if active_positions else "100% CASH (Neutral)"

    # Trade History Persistence.  Suppress consecutive identical flat-cash
    # cycles so the log stays readable; a cash entry after any real trade is
    # still recorded.
    def _is_flat_cash(entry):
        return (str(entry.get('action', '')).startswith('100% CASH')
                and abs(float(entry.get('period_pnl_usd', 0.0))) < 1e-9)

    trade_history = state.get('trade_history', []) if state else []
    period_pnl_usd = current_portfolio_val - prev_portfolio_val

    trade_entry = {
        'timestamp_ist': ist_str,
        'action': action_summary,
        'gross_return_pct': round(period_return * 100.0, 3),
        'period_return_pct': round(net_period_return * 100.0, 3),
        'period_pnl_usd': round(period_pnl_usd, 2),
        'portfolio_value': round(current_portfolio_val, 2),
        'total_return_pct': round(total_return_pct, 2),
        'max_dd': round(max_dd, 4),
        'leverage': round(float(lev), 2),
        'risk_multiplier': round(risk_multiplier, 4),
        'risk_drawdown': round(risk_drawdown, 4),
        'turnover': round(turnover, 4),
        'costs_pct': round(costs * 100.0, 3),
    }

    if (not trade_history
            or not (_is_flat_cash(trade_entry) and _is_flat_cash(trade_history[-1])
                    and trade_history[-1].get('portfolio_value') == trade_entry['portfolio_value'])):
        trade_history.append(trade_entry)
    if len(trade_history) > 100:
        trade_history = trade_history[-100:]

    new_state = {
        'portfolio_value': current_portfolio_val,
        'peak_equity': peak_equity,
        'max_dd': round(max_dd, 6),
        'weights': {t: float(target_weights[i]) for i, t in enumerate(tickers)},
        'weights_raw': {t: float(weights[i]) for i, t in enumerate(tickers)},
        'signal_model': SIGNAL_MODEL,
        'universe': UNIVERSE_TAG,
        'no_trade_band': NO_TRADE_BAND,
        'regime_on': regime_on,
        'regime_sma': REGIME_LOOKBACK,
        'leverage': round(float(lev), 2),
        'last_gross_return_pct': round(period_return * 100.0, 3),
        'last_turnover': round(turnover, 4),
        'last_trading_costs': round(trading_costs, 6),
        'last_short_financing': round(short_financing, 6),
        'last_costs': round(costs, 6),
        'last_days_held': days_held,
        'last_risk_multiplier': round(risk_multiplier, 4),
        'last_risk_drawdown': round(risk_drawdown, 4),
        'last_prices': latest_prices,
        'last_bar_date': str(last_bar_date),
        'last_updated': now_utc.isoformat(),
        'last_updated_ist': ist_str,
        'last_action': action_summary,
        'last_period_return_pct': round(net_period_return * 100.0, 3),
        'last_period_pnl_usd': round(period_pnl_usd, 2),
        'trade_history': trade_history
    }

    # Save price snapshot after each run (anchored to the package, not CWD).
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "strategy": "adaptive_v3",
        "signal_model": SIGNAL_MODEL,
        "universe": UNIVERSE_TAG,
        "no_trade_band": NO_TRADE_BAND,
        "timestamp": now_utc.isoformat(),
        "tickers": tickers,
        "dates": [str(d) for d in close_prices.index],
        "close": {t: close_prices[t].tolist() for t in tickers},
        "volume": {t: volumes[t].tolist() for t in tickers},
        "signals": asset_signals,
        "confidences": asset_confidences,
        "weights": {t: float(target_weights[i]) for i, t in enumerate(tickers)},
        "weights_raw": {t: float(weights[i]) for i, t in enumerate(tickers)},
        "leverage": float(lev),
        "max_dd": float(max_dd),
        "regime_on": regime_on,
        "regime_sma": REGIME_LOOKBACK,
        "last_prices": latest_prices,
        "portfolio_value": current_portfolio_val,
        "risk_multiplier": float(risk_multiplier),
        "risk_drawdown": float(risk_drawdown),
    }

    snap_file = SNAPSHOT_DIR / f"v3_{now_utc.strftime('%Y%m%d_%H%M%S')}.json"
    with open(snap_file, "w") as f:
        json.dump(snapshot, f)

    with open(STATE_FILE, 'w') as f:
        json.dump(new_state, f, indent=2)

    with open(BALANCE_FILE, 'w') as f:
        f.write(f"{current_portfolio_val:.2f}")

    with open(LEGACY_BALANCE_FILE, 'w') as f:
        f.write(f"{current_portfolio_val:.2f}")

    print("\n--- ADAPTIVE v3 RESULTS ---")
    print(f"Signal model: {SIGNAL_MODEL} | Universe: {len(tickers)} assets | "
          f"No-trade band: {NO_TRADE_BAND:.0%}")
    print(f"Regime: "
          f"{'ON (BTC > %dd SMA)' % REGIME_LOOKBACK if regime_on else 'OFF (BTC < %dd SMA -> CASH)' % REGIME_LOOKBACK}")
    print(f"Portfolio leverage: {lev:.2f}x")
    print(f"Current Portfolio Value: ${current_portfolio_val:.2f}")
    print(f"Total Return from Base: {total_return_pct:.2f}%")
    print(f"Max Drawdown: {max_dd:.4f}  |  Risk DD: {risk_drawdown:.4f}  "
          f"| Risk Multiplier: {risk_multiplier:.2f}x")
    print("\nActive Portfolio Allocations:")
    for t in tickers:
        w = float(target_weights[tickers.index(t)])
        status = "LONG" if w > 0.01 else ("SHORT" if w < -0.01 else "CASH")
        print(f"  {t:<10}: {w*100:>6.1f}%  [{status}]  @ ${latest_prices[t]:,.2f}")


if __name__ == '__main__':
    run_adaptive_v3()
