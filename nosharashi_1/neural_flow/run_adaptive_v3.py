import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from datetime import datetime, timezone, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
from neural_flow.alpha.model import LiquidityAlpha
from neural_flow.execution.model import StealthExecution
from neural_flow.risk.model import VolatilitySwitch

STATE_FILE = 'paper_state_v3.json'
BALANCE_FILE = 'paper_balance_v3.txt'
LEGACY_BALANCE_FILE = 'paper_balance.txt'
BASE_CAPITAL = 10000.0

class AdaptiveWeightedV3:
    """
    Hyper-Adaptive v3 Optimizer.
    Implements Square-Root Lasso, Elastic Hysteresis, and Probabilistic Sizing.
    """
    def __init__(self, lambda_base=0.005, tau_base=0.01, critical_threshold=3.0, prev_weights=None):
        self.lambda_base = lambda_base
        self.tau_base = tau_base
        self.critical_threshold = critical_threshold
        self.prev_weights = prev_weights

    def optimize(self, signals, confidences, market_vol=0.15):
        # 1. Square-Root Lasso
        snr = np.mean(np.abs(signals)) / (np.std(signals) + 1e-6)
        lambda_adj = self.lambda_base / (1.0 + snr)

        abs_sigs = np.abs(signals)
        if abs_sigs.sum() == 0:
            return np.zeros(len(signals))

        weights = np.sign(signals) * (abs_sigs / abs_sigs.sum())
        weights = np.where(np.abs(weights) > lambda_adj, weights, 0)

        if np.abs(weights).sum() > 0:
            weights = weights / np.abs(weights).sum()
        else:
            weights = np.zeros(len(signals))

        # 2. Probabilistic Sizing
        final_weights = []
        for w, conf in zip(weights, confidences):
            cap = 0.05 + (np.clip(conf, 0, 3) / 3.0) * (0.50 - 0.05)
            final_weights.append(np.clip(w, -cap, cap))
        weights = np.array(final_weights)

        total_exp = np.sum(np.abs(weights))
        if total_exp > 1.0:
            weights = weights / total_exp

        # 3. Elastic Band Hysteresis
        if self.prev_weights is not None and len(self.prev_weights) == len(weights):
            new_weights = np.copy(self.prev_weights)
            vol_scale = 1.0 / (1.0 + market_vol * 10)
            tau_adj = self.tau_base * vol_scale

            for i in range(len(weights)):
                if abs(signals[i]) > self.critical_threshold:
                    new_weights[i] = weights[i]
                    continue

                diff = weights[i] - self.prev_weights[i]
                if diff < 0 and abs(diff) > (0.5 * tau_adj):
                    new_weights[i] = weights[i]
                elif diff > 0 and abs(diff) > (1.5 * tau_adj):
                    new_weights[i] = weights[i]
                else:
                    new_weights[i] = self.prev_weights[i]
            weights = new_weights

        self.prev_weights = weights
        return weights

class ProRiskManagerV3:
    """
    Institutional Risk Manager v3.
    Implements Ledoit-Wolf and Portfolio Vol Targeting with Convexity scaling.
    """
    def __init__(self, target_vol=0.15, base_leverage=4.0):
        self.target_vol = target_vol
        self.base_leverage = base_leverage

    def get_portfolio_leverage(self, weights, returns_df):
        if len(returns_df) < 20:
            return 1.0

        recent_rets = returns_df.tail(20)
        try:
            lw = LedoitWolf().fit(recent_rets)
            cov_matrix = lw.covariance_ * 252
        except Exception:
            cov_matrix = recent_rets.cov() * 252

        port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        port_vol = np.sqrt(max(0, port_var))

        if port_vol == 0:
            return self.base_leverage

        lev = self.target_vol / port_vol
        return min(lev, self.base_leverage)

def run_adaptive_v3():
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD']
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
    risk_manager = ProRiskManagerV3()

    # 1. Load state if available
    state = None
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
        except Exception:
            state = None

    prev_weights_arr = None
    if state and 'weights' in state:
        prev_weights_arr = np.array([state['weights'].get(t, 0.0) for t in tickers])

    optimizer = AdaptiveWeightedV3(prev_weights=prev_weights_arr)

    # 2. Compute current signals
    asset_signals = []
    asset_confidences = []
    for ticker in tickers:
        asset_df = pd.DataFrame({'Close': close_prices[ticker], 'Volume': volumes[ticker]})
        res_df = alpha_engine.generate_signal(asset_df)
        if not res_df.empty:
            res = res_df.iloc[-1]
            asset_signals.append(res['signal'])
            asset_confidences.append(res['confidence'])
        else:
            asset_signals.append(0.0)
            asset_confidences.append(0.0)

    mkt_vol = returns.iloc[:, 0].std() * np.sqrt(252)
    weights = optimizer.optimize(np.array(asset_signals), np.array(asset_confidences), market_vol=mkt_vol)
    lev = risk_manager.get_portfolio_leverage(weights, returns)
    target_weights = weights * lev

    latest_prices = {t: float(close_prices[t].iloc[-1]) for t in tickers}

    # 3. Calculate PnL
    peak_equity = BASE_CAPITAL
    if state is not None and 'portfolio_value' in state and 'last_prices' in state:
        prev_portfolio_val = float(state['portfolio_value'])
        prev_prices = state['last_prices']
        prev_weights = state.get('weights', {})
        peak_equity = max(float(state.get('peak_equity', BASE_CAPITAL)), prev_portfolio_val)

        period_return = 0.0
        for i, t in enumerate(tickers):
            w = prev_weights.get(t, 0.0)
            if t in prev_prices and prev_prices[t] > 0:
                asset_ret = (latest_prices[t] - prev_prices[t]) / prev_prices[t]
                period_return += w * asset_ret

        turnover = sum(abs(target_weights[i] - prev_weights.get(t, 0.0)) for i, t in enumerate(tickers))
        costs = execution_engine.calculate_cost(turnover)
        current_portfolio_val = prev_portfolio_val * (1.0 + period_return - costs)
        peak_equity = max(peak_equity, current_portfolio_val)
    else:
        # Initial run: reset to clean starting baseline $10,000.00
        current_portfolio_val = BASE_CAPITAL
        period_return = 0.0
        costs = 0.0

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
        'max_dd': round(max_dd, 4)
    }
    trade_history.append(trade_entry)
    if len(trade_history) > 100:
        trade_history = trade_history[-100:]

    new_state = {
        'portfolio_value': current_portfolio_val,
        'peak_equity': peak_equity,
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

    with open(LEGACY_BALANCE_FILE, 'w') as f:
        f.write(f"{current_portfolio_val:.2f}")

    print("\n--- ADAPTIVE v3 RESULTS ---")
    print(f"Current Portfolio Value: ${current_portfolio_val:.2f}")
    print(f"Total Return from Base: {total_return_pct:.2f}%")
    print(f"Max Drawdown: {max_dd:.4f}")
    print("\nActive Portfolio Allocations:")
    for t in tickers:
        w = float(target_weights[tickers.index(t)])
        status = "LONG" if w > 0.01 else ("SHORT" if w < -0.01 else "CASH")
        print(f"  {t:<10}: {w*100:>6.1f}%  [{status}]  @ ${latest_prices[t]:,.2f}")

if __name__ == '__main__':
    run_adaptive_v3()
