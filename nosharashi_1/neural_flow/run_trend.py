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

STATE_FILE = 'paper_state_trend.json'
BALANCE_FILE = 'paper_balance_trend.txt'
BASE_CAPITAL = 10000.0
TICKERS = ['BTC-USD', 'ETH-USD']

LOOKBACK = 50
VOL_WINDOW = 20
TARGET_VOL = 0.15
MAX_LEVERAGE = 2.0


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
    if not tickers:
        raise RuntimeError('none of the configured tickers are available')
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
    if state and 'portfolio_value' in state and 'last_prices' in state:
        prev_val = float(state['portfolio_value'])
        prev_prices = state['last_prices']
        prev_weights = state.get('weights', {})
        peak_equity = max(float(state.get('peak_equity', BASE_CAPITAL)), prev_val)

        period_return = 0.0
        for t in tickers:
            w = float(prev_weights.get(t, 0.0))
            p0 = prev_prices.get(t)
            if p0 and p0 > 0:
                period_return += w * (latest_prices[t] / float(p0) - 1.0)

        turnover = sum(abs(target_weights[t] - float(prev_weights.get(t, 0.0)))
                       for t in tickers)
        costs = StealthExecution().calculate_cost(turnover)
        current_val = prev_val * (1.0 + period_return - costs)
        period_pnl_usd = current_val - prev_val
        peak_equity = max(peak_equity, current_val)
    else:
        prev_val = BASE_CAPITAL
        period_return = 0.0
        # Match the backtest: entering the first book is not free.
        turnover = sum(abs(target_weights[t]) for t in tickers)
        costs = StealthExecution().calculate_cost(turnover)
        current_val = BASE_CAPITAL * (1.0 - costs)
        period_pnl_usd = current_val - BASE_CAPITAL
        peak_equity = BASE_CAPITAL

    total_return_pct = (current_val / BASE_CAPITAL - 1.0) * 100.0
    max_dd = max(0.0, (peak_equity - current_val) / peak_equity) if peak_equity > 0 else 0.0

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
    history.append({
        'timestamp_ist': ist_str,
        'action': action_summary,
        'period_return_pct': round(period_return * 100.0, 3),
        'period_pnl_usd': round(period_pnl_usd, 2),
        'portfolio_value': round(current_val, 2),
        'total_return_pct': round(total_return_pct, 2),
        'max_dd': round(max_dd, 4),
    })
    history = history[-100:]

    new_state = {
        'strategy': 'trend_vol',
        'portfolio_value': round(current_val, 2),
        'peak_equity': round(peak_equity, 2),
        'weights': target_weights,
        'last_prices': latest_prices,
        'last_bar_date': str(last_bar_date),
        'last_updated': now_utc.isoformat(),
        'last_updated_ist': ist_str,
        'last_action': action_summary,
        'last_period_return_pct': round(period_return * 100.0, 3),
        'last_period_pnl_usd': round(period_pnl_usd, 2),
        'last_turnover': round(turnover, 4),
        'last_costs': round(costs, 6),
        'trade_history': history,
    }

    snapshot_dir = Path(__file__).resolve().parent.parent / 'price_snapshots'
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {
        'strategy': 'trend_vol',
        'timestamp': now_utc.isoformat(),
        'tickers': tickers,
        'dates': [str(d) for d in close.index],
        'close': {t: close[t].tolist() for t in tickers},
        'weights': target_weights,
        'last_prices': latest_prices,
        'portfolio_value': current_val,
    }
    with open(snapshot_dir / f"trend_{now_utc.strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
        json.dump(snapshot, f)

    with open(STATE_FILE, 'w') as f:
        json.dump(new_state, f, indent=2)
    with open(BALANCE_FILE, 'w') as f:
        f.write(f"{current_val:.2f}")

    print('\n--- TREND VOL RESULTS ---')
    print(f"Portfolio Value: ${current_val:.2f}")
    print(f"Period Return:   {period_return * 100:.2f}%  (costs {costs * 100:.3f}%)")
    print(f"Total Return:    {total_return_pct:.2f}%")
    print(f"Max Drawdown:    {max_dd * 100:.2f}%")
    print('Allocations:')
    for t in tickers:
        w = target_weights[t]
        status = 'LONG' if w > 0.01 else ('SHORT' if w < -0.01 else 'CASH')
        print(f"  {t:<10}: {w * 100:>6.1f}%  [{status}]  @ ${latest_prices[t]:,.2f}")


if __name__ == '__main__':
    main()
