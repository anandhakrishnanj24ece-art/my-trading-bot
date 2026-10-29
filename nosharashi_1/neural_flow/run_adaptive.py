import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from datetime import datetime, timezone, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from neural_flow.alpha.model import LiquidityAlpha
from neural_flow.execution.model import StealthExecution
from neural_flow.risk.model import VolatilitySwitch

STATE_FILE = 'paper_state_adaptive.json'
BALANCE_FILE = 'paper_balance_adaptive.txt'
BASE_CAPITAL = 10000.0

def run_adaptive_weighted_strategy():
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD']

    print("Downloading latest market data for Adaptive Strategy...")
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

    close_prices = close_prices.dropna(axis=1, how='all').ffill().bfill()
    volumes = volumes.dropna(axis=1, how='all').ffill().bfill()
    valid_tickers = [t for t in tickers if t in close_prices.columns and t in volumes.columns]
    close_prices = close_prices[valid_tickers]
    volumes = volumes[valid_tickers]

    tickers = close_prices.columns.tolist()
    returns = close_prices.pct_change().dropna()
    common_index = returns.index.intersection(close_prices.index).intersection(volumes.index)
    returns = returns.loc[common_index]
    close_prices = close_prices.loc[common_index]
    volumes = volumes.loc[common_index]

    alpha_engine = LiquidityAlpha()
    execution_engine = StealthExecution()
    risk_engine = VolatilitySwitch(base_leverage=4.0)

    # 1. Compute current signals on latest market bar
    asset_signals = []
    asset_confidences = []
    for ticker in tickers:
        asset_df = pd.DataFrame({
            'Close': close_prices[ticker],
            'Volume': volumes[ticker]
        })
        res_df = alpha_engine.generate_signal(asset_df)
        if not res_df.empty:
            res = res_df.iloc[-1]
            asset_signals.append(res['signal'])
            asset_confidences.append(res['confidence'])
        else:
            asset_signals.append(0.0)
            asset_confidences.append(0.0)

    signals_arr = np.array(asset_signals)
    conf_arr = np.array(asset_confidences)

    # 2. Adaptive Weighting Logic
    if np.abs(signals_arr).sum() > 0:
        raw_weights = np.sign(signals_arr) * (np.abs(signals_arr) / np.abs(signals_arr).sum())
    else:
        raw_weights = np.zeros(len(tickers))

    final_weights_pre_lev = []
    for j in range(len(raw_weights)):
        conf = conf_arr[j]
        if conf < 1.0: cap = 0.05
        elif conf < 2.0: cap = 0.20
        else: cap = 0.50
        final_weights_pre_lev.append(np.clip(raw_weights[j], -cap, cap))

    final_weights_pre_lev = np.array(final_weights_pre_lev)
    total_exp = np.sum(np.abs(final_weights_pre_lev))
    if total_exp > 1.0:
        final_weights_pre_lev = final_weights_pre_lev / total_exp

    benchmark_rets = returns.iloc[:, 0]
    lev = risk_engine.get_leverage(benchmark_rets)
    target_weights = final_weights_pre_lev * lev

    latest_prices = {t: float(close_prices[t].iloc[-1]) for t in tickers}

    # 3. Load or initialize paper trading state
    state = None
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
        except Exception:
            state = None

    if state is not None and 'portfolio_value' in state and 'last_prices' in state:
        prev_portfolio_val = float(state['portfolio_value'])
        prev_prices = state['last_prices']
        prev_weights = state.get('weights', {})

        # Compute period return on held positions
        period_return = 0.0
        for i, t in enumerate(tickers):
            w = prev_weights.get(t, 0.0)
            if t in prev_prices and prev_prices[t] > 0:
                asset_ret = (latest_prices[t] - prev_prices[t]) / prev_prices[t]
                period_return += w * asset_ret

        # Turnover and costs
        turnover = sum(abs(target_weights[i] - prev_weights.get(t, 0.0)) for i, t in enumerate(tickers))
        costs = execution_engine.calculate_cost(turnover)
        current_portfolio_val = prev_portfolio_val * (1.0 + period_return - costs)
    else:
        # Initial run: reset to clean starting baseline $10,000.00
        current_portfolio_val = BASE_CAPITAL
        period_return = 0.0
        costs = 0.0

    total_return_pct = ((current_portfolio_val - BASE_CAPITAL) / BASE_CAPITAL) * 100.0

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

    action_summary = ", ".join(active_positions) if active_positions else "100% CASH (Neutral)"

    # Trade History Persistence
    trade_history = state.get('trade_history', []) if state else []
    period_pnl_usd = (current_portfolio_val - prev_portfolio_val) if (state and 'portfolio_value' in state) else 0.0

    trade_entry = {
        'timestamp_ist': ist_str,
        'action': action_summary,
        'period_return_pct': round(period_return * 100.0, 3),
        'period_pnl_usd': round(period_pnl_usd, 2),
        'portfolio_value': round(current_portfolio_val, 2),
        'total_return_pct': round(total_return_pct, 2),
        'leverage': round(lev, 2)
    }
    trade_history.append(trade_entry)
    if len(trade_history) > 100:
        trade_history = trade_history[-100:]

    new_state = {
        'portfolio_value': current_portfolio_val,
        'weights': {t: float(target_weights[i]) for i, t in enumerate(tickers)},
        'last_prices': latest_prices,
        'last_updated': now_utc.isoformat(),
        'last_updated_ist': ist_str,
        'last_action': action_summary,
        'last_period_return_pct': round(period_return * 100.0, 3),
        'last_period_pnl_usd': round(period_pnl_usd, 2),
        'trade_history': trade_history
    }

    with open(STATE_FILE, 'w') as f:
        json.dump(new_state, f, indent=2)

    with open(BALANCE_FILE, 'w') as f:
        f.write(f"{current_portfolio_val:.2f}")

    print("\n--- ADAPTIVE WEIGHTED STRATEGY RESULTS ---")
    print(f"Current Portfolio Value: ${current_portfolio_val:.2f}")
    print(f"Total Return from Base: {total_return_pct:.2f}%")
    print(f"Current Leverage: {lev:.2f}x")
    print("\nActive Portfolio Allocations:")
    for t in tickers:
        w = float(target_weights[tickers.index(t)])
        status = "LONG" if w > 0.01 else ("SHORT" if w < -0.01 else "CASH")
        print(f"  {t:<10}: {w*100:>6.1f}%  [{status}]  @ ${latest_prices[t]:,.2f}")

if __name__ == '__main__':
    run_adaptive_weighted_strategy()
