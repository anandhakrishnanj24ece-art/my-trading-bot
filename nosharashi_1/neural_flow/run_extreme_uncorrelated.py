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

STATE_FILE = 'paper_state_extreme.json'
BALANCE_FILE = 'paper_balance_extreme.txt'
BASE_CAPITAL = 10000.0

def calculate_beta(weights, asset_returns, benchmark_returns):
    """
    Calculates the portfolio beta relative to a benchmark.
    Beta = Cov(Rp, Rb) / Var(Rb)
    """
    port_returns = (asset_returns * weights).sum(axis=1)
    covariance = np.cov(port_returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    return covariance / benchmark_variance if benchmark_variance > 0 else 0.0

def calculate_diversification_ratio(weights, returns_df):
    """
    Metric: Diversification Ratio = (Sum of weighted vol) / Portfolio Vol
    """
    if len(weights) == 0 or np.sum(np.abs(weights)) == 0:
        return 1.0

    vols = returns_df.std() * np.sqrt(252)
    weighted_vol_sum = np.sum(np.abs(weights) * vols)
    cov = returns_df.cov() * 252
    port_var = np.dot(weights.T, np.dot(cov, weights))
    port_vol = np.sqrt(max(0, port_var))

    return weighted_vol_sum / port_vol if port_vol > 0 else 1.0

def run_extreme_uncorrelated_strategy():
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD']
    print("Downloading market data for EXTREME Uncorrelated Strategy...")

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
    corr_filter = CorrelationFilter(lambda_penalty=0.8) # Higher penalty for extreme
    risk_model = DCCGARCHRiskModel()
    vol_target = VolatilityTarget()
    dd_guard = DrawdownGuard()
    execution = StealthExecution()

    # 1. Alpha Generation
    liq_signals = []
    liq_confidences = []
    for t in tickers:
        asset_df = pd.DataFrame({'Close': close_prices[t], 'Volume': volumes[t]})
        res = liq_alpha.generate_signal(asset_df).iloc[-1]
        liq_signals.append(res['signal'])
        liq_confidences.append(res['confidence'])

    mr_signals_df, mr_conf_df = mr_alpha.generate_signals(returns)
    mr_signals = mr_signals_df.iloc[-1].values
    mr_conf = mr_conf_df.iloc[-1].values

    combined_signal, combined_conf = aggregator.aggregate(
        np.array(liq_signals), np.array(liq_confidences),
        mr_signals, mr_conf
    )

    # 2. Market Neutral Adjustment (Beta-Zeroing)
    # We force sum(weights) = 0 to be Market Neutral
    # Shift signals so they center around zero
    signal_mean = np.mean(combined_signal)
    neutral_signal = combined_signal - signal_mean

    # 3. Correlation Filtering
    initial_weights = neutral_signal / (np.sum(np.abs(neutral_signal)) + 1e-6)
    filtered_weights = corr_filter.apply_filter(initial_weights, returns)

    # 4. DCC Optimization
    opt_weights = risk_model.optimize_weights(filtered_weights, returns)

    # 5. Final Risk Scaling
    mkt_returns = returns.iloc[:, 0]
    lev = vol_target.get_leverage(mkt_returns)

    state = None
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f: state = json.load(f)

    current_equity = state['portfolio_value'] if state else BASE_CAPITAL
    dd_mult = dd_guard.get_multiplier(current_equity / BASE_CAPITAL)

    final_weights = opt_weights * lev * dd_mult

    # Metrics
    div_ratio = calculate_diversification_ratio(final_weights, returns)
    avg_corr = returns.corr().values.mean()
    portfolio_beta = calculate_beta(final_weights, returns, mkt_returns)

    # State Persistence
    now_utc = datetime.now(timezone.utc)
    new_state = {
        'portfolio_value': current_equity,
        'weights': {t: float(w) for t, w in zip(tickers, final_weights)},
        'last_updated': now_utc.isoformat(),
        'metrics': {
            'diversification_ratio': round(div_ratio, 4),
            'avg_portfolio_corr': round(avg_corr, 4),
            'portfolio_beta': round(portfolio_beta, 4)
        }
    }
    with open(STATE_FILE, 'w') as f: json.dump(new_state, f, indent=2)

    print("\n--- EXTREME UNCORRELATED RESULTS ---")
    print(f"Portfolio Value: ${current_equity:.2f}")
    print(f"Diversification Ratio: {div_ratio:.4f} (Goal: > 1.5)")
    print(f"Avg Portfolio Correlation: {avg_corr:.4f}")
    print(f"Portfolio Beta (vs BTC): {portfolio_beta:.4f} (Goal: ~ 0.0)")
    print("\nFinal Allocations (Market Neutral):")
    for t, w in zip(tickers, final_weights):
        status = "LONG" if w > 0.01 else ("SHORT" if w < -0.01 else "CASH")
        print(f"  {t:<10}: {w*100:>6.1f}%  [{status}]")

if __name__ == '__main__':
    run_extreme_uncorrelated_strategy()
