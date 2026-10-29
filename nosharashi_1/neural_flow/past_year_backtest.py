import yfinance as yf
import pandas as pd
import numpy as np
from neural_flow.alpha.model import LiquidityAlpha
from neural_flow.execution.model import StealthExecution
from neural_flow.risk.model import VolatilitySwitch

def run_past_year_backtest():
    print("\n--- RUNNING PAST YEAR ADAPTIVE BACKTEST ---")
    
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOT-USD', 'POL-USD', 'LINK-USD', 'AVAX-USD']
    
    # Fetch 1 year of data
    data_raw = yf.download(tickers, period='1y', interval='1d')
    
    if data_raw.empty:
        print("No data found.")
        return

    # Use only assets that have at least 90% of the data to avoid NaN pollution
    # We'll be aggressive: drop assets with too many NaNs
    close_raw = data_raw['Close']
    vol_raw = data_raw['Volume']
    
    valid_tickers = []
    for t in tickers:
        if t in close_raw.columns and close_raw[t].isna().sum() < (len(close_raw) * 0.1):
            valid_tickers.append(t)
    
    print(f"Using {len(valid_tickers)} valid tickers: {valid_tickers}")
    
    close_prices = close_raw[valid_tickers].ffill().dropna()
    volumes = vol_raw[valid_tickers].ffill().dropna()
    returns = close_prices.pct_change().dropna()
    
    common_index = returns.index.intersection(close_prices.index).intersection(volumes.index)
    returns = returns.loc[common_index]
    close_prices = close_prices.loc[common_index]
    volumes = volumes.loc[common_index]

    print(f"Processing {len(returns)} days of data...")

    alpha_engine = LiquidityAlpha()
    execution_engine = StealthExecution()
    risk_engine = VolatilitySwitch(base_leverage=4.0)
    
    portfolio_value = [1.0]
    current_weights = np.zeros(len(valid_tickers))
    
    for i in range(len(returns)):
        asset_signals = []
        for ticker in valid_tickers:
            asset_data = pd.DataFrame({
                'Close': close_prices[ticker],
                'Volume': volumes[ticker]
            })
            sig_df = alpha_engine.generate_signal(asset_data)
            sig = sig_df['signal'].iloc[i]
            conf = sig_df['confidence'].iloc[i]
            asset_signals.append((sig, conf))
        
        signals_raw, confs_raw = zip(*asset_signals)
        signals = np.array([s if not np.isnan(s) else 0.0 for s in signals_raw])
        confs = np.array([c if not np.isnan(c) else 0.0 for c in confs_raw])
        
        if np.abs(signals).sum() > 0:
            weights = np.sign(signals) * (np.abs(signals) / np.abs(signals).sum())
            overall_conf = np.max(confs)
            cap = 0.05 if overall_conf < 1.0 else (0.20 if overall_conf < 2.0 else 0.50)
            weights = np.clip(weights, -cap, cap)
            total_exp = np.sum(np.abs(weights))
            if total_exp > 0:
                weights = weights / total_exp
        else:
            weights = np.zeros(len(valid_tickers))
        
        # Proxy volatility using first valid ticker (usually BTC)
        proxy_ticker = valid_tickers[0]
        btc_rets = returns[proxy_ticker]
        lev = risk_engine.get_leverage(btc_rets.iloc[:i+1])
        final_weights = weights * lev
        
        period_return = (returns.iloc[i] @ final_weights)
        turnover = np.sum(np.abs(final_weights - current_weights))
        costs = execution_engine.calculate_cost(turnover)
        
        portfolio_value.append(portfolio_value[-1] * (1 + period_return - costs))
        current_weights = final_weights

    strategy_rets = pd.Series(portfolio_value).pct_change().dropna()
    
    print("\n" + "="*40)
    print(f"FINAL RESULTS: PAST YEAR")
    print("="*40)
    print(f"Final Portfolio Value: {portfolio_value[-1]:.4f}")
    print(f"Total Return: {(portfolio_value[-1]-1)*100:.2f}%")
    if len(strategy_rets) > 1 and strategy_rets.std() != 0:
        print(f"Sharpe Ratio: {(strategy_rets.mean()/strategy_rets.std()) * np.sqrt(252):.4f}")
    print(f"Max Drawdown: {(1 - pd.Series(portfolio_value).cummax()/pd.Series(portfolio_value)).max():.4f}")
    
    # Monte Carlo
    mc_results = []
    if len(strategy_rets) > 0:
        for _ in range(10000):
            sim_rets = np.random.choice(strategy_rets, size=len(strategy_rets), replace=True)
            mc_results.append(np.prod(1 + sim_rets))
        print(f"MC 5% VaR (1y): {np.percentile(mc_results, 5):.4f}")
    print("="*40)

if __name__ == '__main__':
    run_past_year_backtest()
