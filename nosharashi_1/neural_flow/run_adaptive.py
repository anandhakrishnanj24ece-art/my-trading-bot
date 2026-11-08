import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import yfinance as yf

from neural_flow.alpha.model import LiquidityAlpha
from neural_flow.execution.model import StealthExecution
from neural_flow.risk.model import VolatilitySwitch

LIVE_DIR = Path(__file__).resolve().parent.parent
STATE_FILE = str(LIVE_DIR / 'paper_state_adaptive.json')
BALANCE_FILE = str(LIVE_DIR / 'paper_balance_adaptive.txt')
SNAPSHOT_DIR = LIVE_DIR / 'price_snapshots'
BASE_CAPITAL = 10000.0
UNIVERSE_TAG = 'adaptive_9'

# Cost / risk controls.  A 2% band stops tiny daily rebalances, and short
# exposure pays a conservative daily financing charge.
NO_TRADE_BAND = 0.02
SHORT_COST_BPS_PER_DAY = 3.0
MIN_HISTORY_BARS = 220


def run_adaptive_weighted_strategy():
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD',
               'ADA-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD']

    print("Downloading latest market data for Adaptive Strategy...")
    data_raw = pd.DataFrame()
    for _ in range(3):
        try:
            data_raw = yf.download(tickers, period='1y', interval='1d',
                                   progress=False, threads=False)
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

    # Forward-fill real gaps only.  Never back-fill: leading values belong to
    # the future and would leak information into the early signal history.
    close_prices = close_prices.dropna(axis=1, how='all').ffill()
    volumes = volumes.dropna(axis=1, how='all').ffill()
    valid_tickers = [t for t in tickers if t in close_prices.columns and t in volumes.columns]
    close_prices = close_prices[valid_tickers]
    volumes = volumes[valid_tickers]

    # A stale or missing latest print makes a ticker untradeable today.
    valid_tickers = [t for t in valid_tickers
                     if np.isfinite(float(close_prices[t].iloc[-1]))]
    if not valid_tickers:
        print("Warning: no ticker has a usable latest price. Skipping run.")
        return
    close_prices = close_prices[valid_tickers]
    volumes = volumes[valid_tickers]
    tickers = valid_tickers

    returns = close_prices.pct_change().dropna()
    common_index = returns.index.intersection(close_prices.index).intersection(volumes.index)
    returns = returns.loc[common_index]
    close_prices = close_prices.loc[common_index]
    volumes = volumes.loc[common_index]

    if len(close_prices) < MIN_HISTORY_BARS:
        print(f"Warning: only {len(close_prices)} bars available; "
              f"need at least {MIN_HISTORY_BARS} for a full 200-day trend. Skipping run.")
        return

    # Candle-close safety check (crypto daily closes at 00:00 UTC; allow after 01:00 UTC).
    now_utc = datetime.now(timezone.utc)
    last_bar_date = (close_prices.index[-1].date()
                     if hasattr(close_prices.index[-1], 'date')
                     else pd.to_datetime(close_prices.index[-1]).date())

    if last_bar_date == now_utc.date() and now_utc.hour < 1:
        print(f"[{now_utc.isoformat()}] Daily candle not closed yet "
              f"(current hour: {now_utc.hour} UTC, last bar date: {last_bar_date}). Skipping trade.")
        return

    alpha_engine = LiquidityAlpha()
    execution_engine = StealthExecution()
    risk_engine = VolatilitySwitch(base_leverage=4.0)

    # 1. Compute current signals on the latest market bar.
    asset_signals = []
    asset_confidences = []
    for ticker in tickers:
        asset_df = pd.DataFrame({
            'Close': close_prices[ticker],
            'Volume': volumes[ticker],
        })
        res_df = alpha_engine.generate_signal(asset_df)
        if not res_df.empty:
            res = res_df.iloc[-1]
            asset_signals.append(res['signal'])
            asset_confidences.append(res['confidence'])
        else:
            asset_signals.append(0.0)
            asset_confidences.append(0.0)

    signals_arr = np.array(asset_signals, dtype=float)
    conf_arr = np.array(asset_confidences, dtype=float)

    # 2. Adaptive weighting: confidence caps, gross normalisation, vol switch.
    if np.abs(signals_arr).sum() > 0:
        raw_weights = np.sign(signals_arr) * (np.abs(signals_arr) / np.abs(signals_arr).sum())
    else:
        raw_weights = np.zeros(len(tickers))

    final_weights_pre_lev = []
    for j in range(len(raw_weights)):
        conf = conf_arr[j]
        if conf < 1.0:
            cap = 0.05
        elif conf < 2.0:
            cap = 0.20
        else:
            cap = 0.50
        final_weights_pre_lev.append(np.clip(raw_weights[j], -cap, cap))

    final_weights_pre_lev = np.array(final_weights_pre_lev, dtype=float)
    total_exp = np.sum(np.abs(final_weights_pre_lev))
    if total_exp > 1.0:
        final_weights_pre_lev = final_weights_pre_lev / total_exp

    benchmark_rets = returns.iloc[:, 0]
    lev = risk_engine.get_leverage(benchmark_rets)
    target_weights = final_weights_pre_lev * lev

    latest_prices = {t: float(close_prices[t].iloc[-1]) for t in tickers}

    # 3. Load state and mark the existing book to market.
    state = None
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as handle:
                state = json.load(handle)
        except Exception:
            state = None

    same_book = bool(state) and state.get('universe') == UNIVERSE_TAG

    # Idempotency guard: never process the same daily bar twice for the same book.
    if state and same_book and state.get('last_bar_date') == str(last_bar_date):
        print(f"[{now_utc.isoformat()}] No new daily bar since last run ({last_bar_date}). Skipping.")
        return

    prev_portfolio_val = BASE_CAPITAL
    period_return = 0.0
    turnover = 0.0
    trading_costs = 0.0
    short_financing = 0.0
    short_gross_prev = 0.0
    days_held = 0
    peak_equity = BASE_CAPITAL
    prev_max_dd = 0.0
    actual_weights = {t: 0.0 for t in tickers}

    if state is not None and 'portfolio_value' in state and 'last_prices' in state:
        prev_portfolio_val = float(state['portfolio_value'])
        prev_prices = state['last_prices']
        prev_weights = state.get('weights', {})
        peak_equity = max(float(state.get('peak_equity', BASE_CAPITAL)),
                          prev_portfolio_val)
        prev_max_dd = float(state.get('max_dd', 0.0))

        # Mark-to-market return and the *actual* drifted weights.
        for t in tickers:
            w = float(prev_weights.get(t, 0.0))
            short_gross_prev += max(-w, 0.0)
            if t in prev_prices and float(prev_prices[t]) > 0:
                asset_ret = (latest_prices[t] - float(prev_prices[t])) / float(prev_prices[t])
                period_return += w * asset_ret

        denom = 1.0 + period_return
        if denom > 1e-9:
            for t in tickers:
                if t in prev_prices and float(prev_prices[t]) > 0:
                    actual_weights[t] = (float(prev_weights.get(t, 0.0))
                                         * (latest_prices[t] / float(prev_prices[t]))
                                         / denom)

        # Positions that fell out of the download must be closed, not silently
        # dropped from the book.
        stale = [t for t in prev_weights
                 if t not in tickers and abs(float(prev_weights[t])) > 0]
        stale_closing = sum(abs(float(prev_weights[t])) for t in stale)
        if stale:
            print(f"Warning: closing {len(stale)} stale position(s) missing from data: "
                  f"{', '.join(stale)}")

        # A no-trade band reduces churn, but never blocks a full exit or a
        # long/short reversal.
        target = np.array(target_weights, dtype=float)
        actual = np.array([actual_weights[t] for t in tickers], dtype=float)
        if NO_TRADE_BAND > 0:
            delta = target - actual
            same_direction = np.sign(target) == np.sign(actual)
            material = np.abs(delta) >= NO_TRADE_BAND
            keep_position = same_direction & ~material & (np.abs(target) > 1e-12)
            target = np.where(keep_position, actual, target)
        target_weights = target

        try:
            days_held = max(0, (last_bar_date
                                - pd.to_datetime(state['last_bar_date']).date()).days)
        except Exception:
            days_held = 0

        turnover = float(np.abs(target_weights - actual).sum()) + stale_closing
        trading_costs = execution_engine.calculate_cost(turnover)
        short_financing = (short_gross_prev
                           * (SHORT_COST_BPS_PER_DAY / 1e4)
                           * days_held)
        costs = trading_costs + short_financing
        current_portfolio_val = prev_portfolio_val * (1.0 + period_return - costs)
    else:
        # Initial run from cash: the entry trade must pay its own cost.
        turnover = float(np.abs(target_weights).sum())
        trading_costs = execution_engine.calculate_cost(turnover)
        costs = trading_costs
        current_portfolio_val = BASE_CAPITAL * (1.0 - costs)

    peak_equity = max(peak_equity, current_portfolio_val)
    drawdown = ((peak_equity - current_portfolio_val) / peak_equity
                if peak_equity > 0 else 0.0)
    max_dd = max(prev_max_dd, drawdown)

    net_period_return = ((current_portfolio_val / prev_portfolio_val - 1.0)
                         if prev_portfolio_val > 0 else 0.0)
    total_return_pct = ((current_portfolio_val - BASE_CAPITAL) / BASE_CAPITAL) * 100.0

    # 4. Generate IST timestamp and action summary.
    now_utc = datetime.now(timezone.utc)
    ist_zone = timezone(timedelta(hours=5, minutes=30))
    now_ist = now_utc.astimezone(ist_zone)
    ist_str = now_ist.strftime("%d %b %Y, %I:%M:%S %p IST")

    active_positions = []
    for i, t in enumerate(tickers):
        w = float(target_weights[i])
        if abs(w) > 0.01:
            side = "LONG" if w > 0 else "SHORT"
            active_positions.append(
                f"{side} {t} ({abs(w) * 100:.1f}%) @ ${latest_prices[t]:,.2f}"
            )

    action_summary = ", ".join(active_positions) if active_positions else "100% CASH (Neutral)"

    # Trade history persistence.
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
        'leverage': round(lev, 2),
        'turnover': round(turnover, 4),
        'costs_pct': round(costs * 100.0, 3),
    }
    trade_history.append(trade_entry)
    if len(trade_history) > 100:
        trade_history = trade_history[-100:]

    new_state = {
        'portfolio_value': current_portfolio_val,
        'peak_equity': peak_equity,
        'max_dd': round(max_dd, 6),
        'weights': {t: float(target_weights[i]) for i, t in enumerate(tickers)},
        'last_prices': latest_prices,
        'last_bar_date': str(last_bar_date),
        'last_updated': now_utc.isoformat(),
        'last_updated_ist': ist_str,
        'last_action': action_summary,
        'leverage': round(lev, 2),
        'universe': UNIVERSE_TAG,
        'no_trade_band': NO_TRADE_BAND,
        'last_gross_return_pct': round(period_return * 100.0, 3),
        'last_period_return_pct': round(net_period_return * 100.0, 3),
        'last_period_pnl_usd': round(period_pnl_usd, 2),
        'last_turnover': round(turnover, 4),
        'last_trading_costs': round(trading_costs, 6),
        'last_short_financing': round(short_financing, 6),
        'last_costs': round(costs, 6),
        'last_days_held': days_held,
        'trade_history': trade_history,
    }

    # Save a price snapshot for the snapshot backtester.
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "strategy": "adaptive",
        "timestamp": now_utc.isoformat(),
        "tickers": tickers,
        "dates": [str(d) for d in close_prices.index],
        "close": {t: close_prices[t].tolist() for t in tickers},
        "volume": {t: volumes[t].tolist() for t in tickers},
        "signals": asset_signals,
        "confidences": asset_confidences,
        "weights": {t: float(target_weights[i]) for i, t in enumerate(tickers)},
        "leverage": float(lev),
        "universe": UNIVERSE_TAG,
        "last_prices": latest_prices,
        "portfolio_value": current_portfolio_val,
        "max_dd": float(max_dd),
    }

    snap_file = SNAPSHOT_DIR / f"adaptive_{now_utc.strftime('%Y%m%d_%H%M%S')}.json"
    with open(snap_file, "w", encoding='utf-8') as handle:
        json.dump(snapshot, handle)

    with open(STATE_FILE, 'w', encoding='utf-8') as handle:
        json.dump(new_state, handle, indent=2)

    with open(BALANCE_FILE, 'w', encoding='utf-8') as handle:
        handle.write(f"{current_portfolio_val:.2f}")

    print("\n--- ADAPTIVE WEIGHTED STRATEGY RESULTS ---")
    print(f"Current Portfolio Value: ${current_portfolio_val:.2f}")
    print(f"Period Return: {net_period_return * 100:.2f}%  "
          f"(gross {period_return * 100:.2f}%, "
          f"trading {trading_costs * 100:.3f}%, "
          f"short financing {short_financing * 100:.3f}%)")
    print(f"Total Return from Base: {total_return_pct:.2f}%")
    print(f"Max Drawdown: {max_dd * 100:.2f}%")
    print(f"Current Leverage: {lev:.2f}x  |  Turnover: {turnover:.2f}")
    print("\nActive Portfolio Allocations:")
    for t in tickers:
        w = float(target_weights[tickers.index(t)])
        status = "LONG" if w > 0.01 else ("SHORT" if w < -0.01 else "CASH")
        print(f"  {t:<10}: {w * 100:>6.1f}%  [{status}]  @ ${latest_prices[t]:,.2f}")


if __name__ == '__main__':
    run_adaptive_weighted_strategy()
