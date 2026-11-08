"""
Live paper-trading runner for the redesigned TrendVol signal.

Strategy
--------
Long-only time-series momentum on BTC and ETH:
  * hold an asset when its close is above its 50-day SMA, otherwise cash;
  * size survivors by inverse realised volatility;
  * scale the whole book to a 15% annualised volatility target (max 2x).

The weights are computed from the latest daily close only, so the same bar is
never processed twice (guarded by `last_bar_date`).

State:   paper_state_trend.json
Balance: paper_balance_trend.txt
"""
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import yfinance as yf

from neural_flow.alpha.trend import TrendVolSignal
from neural_flow.execution.model import StealthExecution
from neural_flow.risk.model import drawdown_multiplier

LIVE_DIR = Path(__file__).resolve().parent.parent
STATE_FILE = str(LIVE_DIR / 'paper_state_trend.json')
BALANCE_FILE = str(LIVE_DIR / 'paper_balance_trend.txt')
SNAPSHOT_DIR = LIVE_DIR / 'price_snapshots'
BASE_CAPITAL = 10000.0
TICKERS = ['BTC-USD', 'ETH-USD']

LOOKBACK = 50
VOL_WINDOW = 20
TARGET_VOL = 0.15
MAX_LEVERAGE = 2.0
NO_TRADE_BAND = 0.02
# Protective drawdown overlay: de-risk from 5% drawdown, floor at 25% from 15%.
DD_CAUTION = 0.05
DD_HALT = 0.15
DD_FLOOR = 0.25


def download_close(tickers=TICKERS, period='1y', attempts=5):
    for attempt in range(attempts):
        try:
            raw = yf.download(tickers, period=period, interval='1d', progress=False,
                              threads=False, auto_adjust=True)
            if raw is None or len(raw) == 0 or 'Close' not in raw:
                raise RuntimeError('empty download')
            close = raw['Close']
            if isinstance(close, pd.Series):
                close = close.to_frame(tickers[0])
            close = close.ffill().dropna(how='all').dropna(axis=1, how='all')
            close = close.ffill().dropna()
            if len(close) < LOOKBACK + VOL_WINDOW + 5:
                raise RuntimeError(f'not enough history ({len(close)} bars)')
            return close
        except Exception as exc:  # noqa: BLE001
            print(f"Download attempt {attempt + 1} failed: {exc}")
            time.sleep(3)
    raise RuntimeError('market data download failed')


def load_state():
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not read state file ({exc}); starting fresh.")
        return None


def main():
    now_utc = datetime.now(timezone.utc)
    close = download_close()
    tickers = [t for t in TICKERS if t in close.columns]
    tickers = [t for t in tickers if np.isfinite(float(close[t].iloc[-1]))]
    if not tickers:
        raise RuntimeError('none of the configured tickers have a usable latest price')
    close = close[tickers]

    last_ts = close.index[-1]
    last_bar_date = last_ts.date() if hasattr(last_ts, 'date') else pd.to_datetime(last_ts).date()
    if last_bar_date == now_utc.date() and now_utc.hour < 1:
        print(f"[{now_utc.isoformat()}] Daily candle not closed yet "
              f"(hour={now_utc.hour} UTC, last bar={last_bar_date}). Skipping.")
        return

    state = load_state()
    if state and state.get('last_bar_date') == str(last_bar_date):
        print(f"[{now_utc.isoformat()}] No new daily bar since last run "
              f"({last_bar_date}). Skipping.")
        return

    signal = TrendVolSignal(lookback=LOOKBACK, vol_window=VOL_WINDOW,
                            target_vol=TARGET_VOL, max_leverage=MAX_LEVERAGE)
    target = signal.latest_weights(close).reindex(tickers).fillna(0.0)
    target_weights = {t: float(target[t]) for t in tickers}
    latest_prices = {t: float(close[t].iloc[-1]) for t in tickers}

    # ---- mark-to-market on the previous book, then charge turnover costs ----
    execution_engine = StealthExecution()
    days_held = 0
    peak_equity = BASE_CAPITAL
    prev_max_dd = 0.0
    risk_multiplier = 1.0
    risk_drawdown = 0.0
    if state and 'portfolio_value' in state and 'last_prices' in state:
        prev_val = float(state['portfolio_value'])
        prev_prices = state['last_prices']
        prev_weights = state.get('weights', {})
        # High-water mark before this period's trading.
        peak_equity = max(float(state.get('peak_equity', BASE_CAPITAL)), prev_val)
        prev_max_dd = float(state.get('max_dd', 0.0))

        period_return = 0.0
        for t in tickers:
            w = float(prev_weights.get(t, 0.0))
            p0 = prev_prices.get(t)
            if p0 and p0 > 0:
                period_return += w * (latest_prices[t] / float(p0) - 1.0)

        # Recover the actual, price-drifted book.  Turnover must compare the
        # new target with what is actually held, not with the old target.
        actual_weights = {t: 0.0 for t in tickers}
        denom = 1.0 + period_return
        if denom > 1e-9:
            for t in tickers:
                p0 = prev_prices.get(t)
                if p0 and p0 > 0:
                    actual_weights[t] = (float(prev_weights.get(t, 0.0))
                                         * (latest_prices[t] / float(p0))
                                         / denom)

        # Protective drawdown overlay: scale the signal target down as the
        # portfolio drifts from its high-water mark, before the no-trade band.
        risk_drawdown = (peak_equity - prev_val) / peak_equity if peak_equity > 0 else 0.0
        risk_multiplier = drawdown_multiplier(
            peak_equity, prev_val,
            caution=DD_CAUTION, halt=DD_HALT, floor=DD_FLOOR)
        for t in tickers:
            target_weights[t] = float(target_weights[t]) * risk_multiplier

        stale = [t for t in prev_weights
                 if t not in tickers and abs(float(prev_weights[t])) > 0]
        stale_closing = sum(abs(float(prev_weights[t])) for t in stale)
        if stale:
            print(f"Warning: closing {len(stale)} stale position(s) missing from data: "
                  f"{', '.join(stale)}")

        # Suppress only small same-direction rebalances; never block a full
        # exit or a reversal.
        if NO_TRADE_BAND > 0:
            for t in tickers:
                desired = target_weights[t]
                actual = actual_weights[t]
                same_direction = np.sign(desired) == np.sign(actual)
                material = abs(desired - actual) >= NO_TRADE_BAND
                if same_direction and not material and abs(desired) > 1e-12:
                    target_weights[t] = actual

        turnover = sum(abs(target_weights[t] - actual_weights[t])
                       for t in tickers) + stale_closing
        costs = execution_engine.calculate_cost(turnover)
        current_val = prev_val * (1.0 + period_return - costs)
        period_pnl_usd = current_val - prev_val

        try:
            days_held = max(0, (last_bar_date
                                - pd.to_datetime(state['last_bar_date']).date()).days)
        except Exception:
            days_held = 0
    else:
        prev_val = BASE_CAPITAL
        period_return = 0.0
        # Match the backtest: entering the first book is not free.
        turnover = sum(abs(target_weights[t]) for t in tickers)
        costs = execution_engine.calculate_cost(turnover)
        current_val = BASE_CAPITAL * (1.0 - costs)
        period_pnl_usd = current_val - BASE_CAPITAL

    peak_equity = max(peak_equity, current_val)
    drawdown = ((peak_equity - current_val) / peak_equity
                if peak_equity > 0 else 0.0)
    max_dd = max(prev_max_dd, drawdown)

    net_period_return = ((current_val / prev_val - 1.0)
                         if prev_val > 0 else 0.0)
    total_return_pct = (current_val / BASE_CAPITAL - 1.0) * 100.0

    ist = now_utc.astimezone(timezone(timedelta(hours=5, minutes=30)))
    ist_str = ist.strftime("%d %b %Y, %I:%M:%S %p IST")

    active = []
    for t in tickers:
        w = target_weights[t]
        if abs(w) > 0.01:
            side = 'LONG' if w > 0 else 'SHORT'
            active.append(f"{side} {t} ({abs(w) * 100:.1f}%) @ ${latest_prices[t]:,.2f}")
    action_summary = ', '.join(active) if active else '100% CASH (Neutral)'

    history = state.get('trade_history', []) if state else []

    # Suppress consecutive identical flat-cash cycles so the log stays
    # readable; a cash entry after any real trade is still recorded.
    def _is_flat_cash(entry):
        return (str(entry.get('action', '')).startswith('100% CASH')
                and abs(float(entry.get('period_pnl_usd', 0.0))) < 1e-9)

    new_entry = {
        'timestamp_ist': ist_str,
        'action': action_summary,
        'gross_return_pct': round(period_return * 100.0, 3),
        'period_return_pct': round(net_period_return * 100.0, 3),
        'period_pnl_usd': round(period_pnl_usd, 2),
        'portfolio_value': round(current_val, 2),
        'total_return_pct': round(total_return_pct, 2),
        'max_dd': round(max_dd, 4),
        'risk_multiplier': round(risk_multiplier, 4),
        'risk_drawdown': round(risk_drawdown, 4),
        'turnover': round(turnover, 4),
        'costs_pct': round(costs * 100.0, 3),
    }
    if (not history
            or not (_is_flat_cash(new_entry) and _is_flat_cash(history[-1])
                    and history[-1].get('portfolio_value') == new_entry['portfolio_value'])):
        history.append(new_entry)
    history = history[-100:]

    new_state = {
        'strategy': 'trend_vol',
        'portfolio_value': round(current_val, 2),
        'peak_equity': round(peak_equity, 2),
        'max_dd': round(max_dd, 6),
        'weights': target_weights,
        'last_prices': latest_prices,
        'last_bar_date': str(last_bar_date),
        'last_updated': now_utc.isoformat(),
        'last_updated_ist': ist_str,
        'last_action': action_summary,
        'no_trade_band': NO_TRADE_BAND,
        'last_gross_return_pct': round(period_return * 100.0, 3),
        'last_period_return_pct': round(net_period_return * 100.0, 3),
        'last_period_pnl_usd': round(period_pnl_usd, 2),
        'last_turnover': round(turnover, 4),
        'last_costs': round(costs, 6),
        'last_days_held': days_held,
        'last_risk_multiplier': round(risk_multiplier, 4),
        'last_risk_drawdown': round(risk_drawdown, 4),
        'trade_history': history,
    }

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = {
        'strategy': 'trend_vol',
        'timestamp': now_utc.isoformat(),
        'tickers': tickers,
        'dates': [str(d) for d in close.index],
        'close': {t: close[t].tolist() for t in tickers},
        'weights': target_weights,
        'last_prices': latest_prices,
        'portfolio_value': current_val,
        'max_dd': float(max_dd),
        'risk_multiplier': float(risk_multiplier),
        'risk_drawdown': float(risk_drawdown),
    }
    with open(SNAPSHOT_DIR / f"trend_{now_utc.strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(snapshot, f)

    with open(STATE_FILE, 'w') as f:
        json.dump(new_state, f, indent=2)
    with open(BALANCE_FILE, 'w') as f:
        f.write(f"{current_val:.2f}")

    print('\n--- TREND VOL RESULTS ---')
    print(f"Portfolio Value: ${current_val:.2f}")
    print(f"Period Return:   {net_period_return * 100:.2f}%  "
          f"(gross {period_return * 100:.2f}%, costs {costs * 100:.3f}%)")
    print(f"Total Return:    {total_return_pct:.2f}%")
    print(f"Max Drawdown:    {max_dd * 100:.2f}%  |  Risk DD: {risk_drawdown * 100:.2f}%  "
          f"| Risk Multiplier: {risk_multiplier:.2f}x")
    print('Allocations:')
    for t in tickers:
        w = target_weights[t]
        status = 'LONG' if w > 0.01 else ('SHORT' if w < -0.01 else 'CASH')
        print(f"  {t:<10}: {w * 100:>6.1f}%  [{status}]  @ ${latest_prices[t]:,.2f}")


if __name__ == '__main__':
    main()
