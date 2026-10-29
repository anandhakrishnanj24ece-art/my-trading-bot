import yfinance as yf
import pandas as pd
import numpy as np
from neural_flow.alpha.model import LiquidityAlpha
from neural_flow.execution.model import StealthExecution
from neural_flow.risk.model import VolatilitySwitch

def run_neural_flow_adaptive_backtest():
    # 1. Setup Data
    tickers = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT', 'DOT/USDT', 'POL/USDT', 'LINK/USDT', 'AVAX/USDT']
    yf_tickers = [t.replace('/USDT', '-USD') for t in tickers]

    print("Downloading data...")
    data_raw = yf.download(yf_tickers, period='2y', interval='1d')
    close_prices = data_raw['Close'].ffill().dropna()
    volumes = data_raw['Volume'].ffill().dropna()

    returns = close_prices.pct_change().dropna()
    common_index = returns.index.intersection(close_prices.index).intersection(volumes.index)
    returns = returns.loc[common_index]
    close_prices = close_prices.loc[common_index]
    volumes = volumes.loc[common_index]

    # 2. System Components
    alpha_engine = LiquidityAlpha()
    execution_engine = StealthExecution()
    risk_engine = VolatilitySwitch(base_leverage=4.0)

    portfolio_value = [1.0]
    current_weights = np.zeros(len(tickers))

    # 3. Backtest Loop
    for i in range(len(returns)):
        # a. Signal Generation
        asset_signals = []
        asset_confidences = []
        for ticker in tickers:
            yf_t = ticker.replace('/USDT', '-USD')
            asset_df = pd.DataFrame({
                'Close': close_prices[yf_t],
                'Volume': volumes[yf_t]
            })
            res = alpha_engine.generate_signal(asset_df).iloc[i]
            asset_signals.append(res['signal'])
            asset_confidences.append(res['confidence'])

        # b. Adaptive Weighting Logic
        signals_arr = np.array(asset_signals)
        conf_arr = np.array(asset_confidences)

        if np.abs(signals_arr).sum() > 0:
            weights = np.sign(signals_arr) * (np.abs(signals_arr) / np.abs(signals_arr).sum())
        else:
            weights = np.zeros(len(tickers))

        # Apply Adaptive Caps
        final_weights_pre_lev = []
        for j in range(len(weights)):
            conf = conf_arr[j]
            if conf < 1.0: cap = 0.05
            elif conf < 2.0: cap = 0.20
            else: cap = 0.50
            final_weights_pre_lev.append(np.clip(weights[j], -cap, cap))

        final_weights_pre_lev = np.array(final_weights_pre_lev)

        # Ensure total exposure <= 100%
        total_exp = np.sum(np.abs(final_weights_pre_lev))
        if total_exp > 1.0:
            final_weights_pre_lev = final_weights_pre_lev / total_exp

        # Dynamic Leverage
        btc_rets = returns.iloc[:,0]
        lev = risk_engine.get_leverage(btc_rets.iloc[:i+1])
        final_weights = final_weights_pre_lev * lev

        # c. PnL and Costs
        period_return = np.sum(final_weights * returns.iloc[i].values)
        turnover = np.sum(np.abs(final_weights - current_weights))
        costs = execution_engine.calculate_cost(turnover)

        portfolio_value.append(portfolio_value[-1] * (1 + period_return - costs))
        current_weights = final_weights

    # 4. Monte Carlo Validation
    strategy_rets = pd.Series(portfolio_value).pct_change().dropna()
    mc_results = []
    for _ in range(10000):
        sim_rets = np.random.choice(strategy_rets, size=len(strategy_rets), replace=True)
        mc_results.append(np.prod(1 + sim_rets))

    print(f"\n--- NEURAL-FLOW ADAPTIVE BACKTEST RESULTS ---")
    print(f"Final Portfolio Value: {portfolio_value[-1]:.4f}")
    print(f"Total Return: {(portfolio_value[-1]-1)*100:.2f}%")
    print(f"Sharpe Ratio: {(strategy_rets.mean()/strategy_rets.std()) * np.sqrt(252):.4f}")
    print(f"Max Drawdown: {(1 - pd.Series(portfolio_value).cummax()/pd.Series(portfolio_value)).max():.4f}")
    print(f"MC 5% VaR (1y): {np.percentile(mc_results, 5):.4f}")

if __name__ == '__main__':
    run_neural_flow_adaptive_backtest()
