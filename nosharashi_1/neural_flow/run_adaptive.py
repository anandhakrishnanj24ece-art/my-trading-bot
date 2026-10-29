import yfinance as yf
import pandas as pd
import numpy as np
from neural_flow.alpha.model import LiquidityAlpha
from neural_flow.execution.model import StealthExecution
from neural_flow.risk.model import VolatilitySwitch

def run_adaptive_weighted_strategy():
    # 1. Setup Data - Using high-liquidity tickers
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD']

    print("Downloading data for Adaptive Strategy...")
    data_raw = yf.download(tickers, period='1y', interval='1d')

    # Handle yfinance multi-index columns
    if isinstance(data_raw.columns, pd.MultiIndex):
        close_prices = data_raw['Close'].ffill().dropna()
        volumes = data_raw['Volume'].ffill().dropna()
    else:
        close_prices = pd.DataFrame({tickers[0]: data_raw['Close']}).ffill().dropna()
        volumes = pd.DataFrame({tickers[0]: data_raw['Volume']}).ffill().dropna()

    # Filter tickers to only those that actually have data
    tickers = close_prices.columns.tolist()

    returns = close_prices.pct_change().dropna()
    common_index = returns.index.intersection(close_prices.index).intersection(volumes.index)
    returns = returns.loc[common_index]
    close_prices = close_prices.loc[common_index]
    volumes = volumes.loc[common_index]

    # 2. System Components
    alpha_engine = LiquidityAlpha()
    execution_engine = StealthExecution()
    risk_engine = VolatilitySwitch(base_leverage=4.0)

    # Persistence: Load starting capital from file
    try:
        with open('paper_balance.txt', 'r') as f:
            start_capital = float(f.read().strip())
    except:
        start_capital = 10000.0

    portfolio_value = [start_capital]
    current_weights = np.zeros(len(tickers))

    print(f"Executing Adaptive Weighted Strategy on {len(tickers)} assets...")
    for i in range(len(returns)):
        asset_signals = []
        asset_confidences = []

        for ticker in tickers:
            asset_df = pd.DataFrame({
                'Close': close_prices[ticker],
                'Volume': volumes[ticker]
            })
            res_df = alpha_engine.generate_signal(asset_df)
            if not res_df.empty and i < len(res_df):
                res = res_df.iloc[i]
                asset_signals.append(res['signal'])
                asset_confidences.append(res['confidence'])
            else:
                asset_signals.append(0.0)
                asset_confidences.append(0.0)

        signals_arr = np.array(asset_signals)
        conf_arr = np.array(asset_confidences)

        # Adaptive Weighting Logic
        if np.abs(signals_arr).sum() > 0:
            weights = np.sign(signals_arr) * (np.abs(signals_arr) / np.abs(signals_arr).sum())
        else:
            weights = np.zeros(len(tickers))

        # Apply Adaptive Caps based on confidence
        final_weights_pre_lev = []
        for j in range(len(weights)):
            conf = conf_arr[j]
            if conf < 1.0: cap = 0.05
            elif conf < 2.0: cap = 0.20
            else: cap = 0.50
            final_weights_pre_lev.append(np.clip(weights[j], -cap, cap))

        final_weights_pre_lev = np.array(final_weights_pre_lev)

        # Normalize exposure to 100%
        total_exp = np.sum(np.abs(final_weights_pre_lev))
        if total_exp > 1.0:
            final_weights_pre_lev = final_weights_pre_lev / total_exp

        # Dynamic Leverage based on the first available ticker (usually BTC)
        benchmark_rets = returns.iloc[:, 0]
        lev = risk_engine.get_leverage(benchmark_rets.iloc[:i+1])
        final_weights = final_weights_pre_lev * lev

        # PnL and Costs
        period_return = np.sum(final_weights * returns.iloc[i].values)
        turnover = np.sum(np.abs(final_weights - current_weights))
        costs = execution_engine.calculate_cost(turnover)

        portfolio_value.append(portfolio_value[-1] * (1 + period_return - costs))
        current_weights = final_weights

    # Metrics
    strategy_rets = pd.Series(portfolio_value).pct_change().dropna()
    final_val = portfolio_value[-1]
    total_ret = (final_val - 10000.0) / 10000.0 * 100
    max_dd = (1 - pd.Series(portfolio_value).cummax() / pd.Series(portfolio_value)).max()

    # SAVE BALANCE FOR PERSISTENCE
    with open('paper_balance.txt', 'w') as f:
        f.write(str(final_val))

    print(f"\n--- ADAPTIVE WEIGHTED STRATEGY RESULTS ---")
    print(f"Current Portfolio Value: ${final_val:.2f}")
    print(f"Total Return from Base: {total_ret:.2f}%")
    print(f"Max Drawdown: {max_dd:.4f}")

if __name__ == '__main__':
    run_adaptive_weighted_strategy()
