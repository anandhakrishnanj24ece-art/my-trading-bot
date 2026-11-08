import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import json
from datetime import datetime, timezone, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

from neural_flow.alpha.model import LiquidityAlpha, AlphaAggregator
from neural_flow.alpha.mean_reversion import MeanReversionAlpha
from neural_flow.execution.model import StealthExecution
from neural_flow.risk.model import VolatilityTarget, DrawdownGuard
from neural_flow.risk.diversification import CorrelationFilter
from neural_flow.risk.dcc_model import DCCGARCHRiskModel


STATE_FILE = 'paper_state_uncorrelated.json'
BALANCE_FILE = 'paper_balance_uncorrelated.txt'
BASE_CAPITAL = 10000.0

def calculate_diversification_ratio(weights, returns_df):
    """
    Metric: Diversification Ratio = (Sum of weighted vol) / Portfolio Vol
    A value > 1 indicates that the assets are not perfectly correlated.
    """
    if len(weights) == 0 or np.sum(np.abs(weights)) == 0:
        return 1.0

    # Individual annualized volatilities
    vols = returns_df.std() * np.sqrt(252)
    weighted_vol_sum = np.sum(np.abs(weights) * vols)

    # Portfolio volatility
    cov = returns_df.cov() * 252
    port_var = np.dot(weights.T, np.dot(cov, weights))
    port_vol = np.sqrt(max(0, port_var))

    return weighted_vol_sum / port_vol if port_vol > 0 else 1.0

def run_uncorrelated_strategy():
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD']
    print("Downloading market data for Uncorrelated Strategy...")

    # Data fetching
    data_raw = yf.download(tickers, period='1y', interval='1d', progress=False)
    if data_raw.empty:
        print("Error: No data fetched.")
        return

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

    # --- Pipeline Initialization ---
    liq_alpha = LiquidityAlpha()
    mr_alpha = MeanReversionAlpha()
    aggregator = AlphaAggregator()
    corr_filter = CorrelationFilter()
    risk_model = DCCGARCHRiskModel()
    vol_target = VolatilityTarget()
    dd_guard = DrawdownGuard()
    execution = StealthExecution()

    # 1. Alpha Generation
    # Liquidity Alpha (Asset by Asset)
    liq_signals = []
    liq_confidences = []
    for t in tickers:
        asset_df = pd.DataFrame({'Close': close_prices[t], 'Volume': volumes[t]})
        res = liq_alpha.generate_signal(asset_df).iloc[-1]
        liq_signals.append(res['signal'])
        liq_confidences.append(res['confidence'])

    # Mean Reversion Alpha (Cross-Sectional)
    mr_signals_df, mr_conf_df = mr_alpha.generate_signals(returns)
    mr_signals = mr_signals_df.iloc[-1].values
    mr_conf = mr_conf_df.iloc[-1].values

    # Aggregated Signal
    combined_signal, combined_conf = aggregator.aggregate(
        np.array(liq_signals), np.array(liq_confidences),
        mr_signals, mr_conf
    )

    # 2. Correlation Filtering
    initial_weights = combined_signal / (np.sum(np.abs(combined_signal)) + 1e-6)
    filtered_weights = corr_filter.apply_filter(initial_weights, returns)

    # 3. DCC Optimization (Minimum Variance)
    opt_weights = risk_model.optimize_weights(filtered_weights, returns)

    # 4. Volatility Targeting & Guards
    # Use the first asset as a proxy for market vol for leverage
    mkt_returns = returns.iloc[:, 0]
    lev = vol_target.get_leverage(mkt_returns)

    # Load state for Drawdown Guard
    state = None
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f: state = json.load(f)

    current_equity = state['portfolio_value'] if state else BASE_CAPITAL
    dd_mult = dd_guard.get_multiplier(current_equity / BASE_CAPITAL)

    final_weights = opt_weights * lev * dd_mult

    # Metrics Calculation
    div_ratio = calculate_diversification_ratio(final_weights, returns)
    avg_corr = returns.corr().values.mean()

    # PnL Calculation (Simplified for current run)
    # In a real scenario, we'd track prev_weights and calculate period_return
    current_portfolio_val = current_equity # Simplified

    # State Persistence
    now_utc = datetime.now(timezone.utc)
    new_state = {
        'portfolio_value': current_portfolio_val,
        'weights': {t: float(w) for t, w in zip(tickers, final_weights)},
        'last_updated': now_utc.isoformat(),
        'metrics': {
            'diversification_ratio': round(div_ratio, 4),
            'avg_portfolio_corr': round(avg_corr, 4)
        }
    }
    with open(STATE_FILE, 'w') as f: json.dump(new_state, f, indent=2)

    print("\n--- UNCORRELATED STRATEGY RESULTS ---")
    print(f"Portfolio Value: ${current_portfolio_val:.2f}")
    print(f"Diversification Ratio: {div_ratio:.4f} (Goal: > 1.0)")
    print(f"Avg Portfolio Correlation: {avg_corr:.4f} (Goal: Low)")
    print("\nFinal Allocations:")
    for t, w in zip(tickers, final_weights):
        status = "LONG" if w > 0.01 else ("SHORT" if w < -0.01 else "CASH")
        print(f"  {t:<10}: {w*100:>6.1f}%  [{status}]")

if __name__ == '__main__':
    run_uncorrelated_strategy()
