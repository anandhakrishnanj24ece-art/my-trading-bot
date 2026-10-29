import yfinance as yf
import pandas as pd
import numpy as np
from neural_flow.alpha.model import LiquidityAlpha
from neural_flow.execution.model import StealthExecution
from neural_flow.risk.model import VolatilitySwitch

def run_year_backtest(year, tickers):
    print(f"\n--- RUNNING BACKTEST FOR YEAR: {year} ---")
    
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    data_raw = yf.download(tickers, start=start_date, end=end_date, interval='1d')
    
    if data_raw.empty:
        print(f"No data found for {year}")
        return

    # Use a simpler data extraction to avoid multi-index issues with yfinance
    close_prices = data_raw['Close'].ffill().dropna(how='all')
    volumes = data_raw['Volume'].ffill().dropna(how='all')
    
    # Ensure we have returns
    returns = close_prices.pct_change().dropna(how='all')
    
    common_index = returns.index.intersection(close_prices.index).intersection(volumes.index)
    returns = returns.loc[common_index]
    close_prices = close_prices.loc[common_index]
    volumes = volumes.loc[common_index]

    print(f"Processing {len(returns)} days of data...")

    alpha_engine = LiquidityAlpha()
    execution_engine = StealthExecution()
    risk_engine = VolatilitySwitch(base_leverage=4.0)
    
    portfolio_value = [1.0]
    current_weights = np.zeros(len(tickers))
    
    trade_count = 0
    for i in range(len(returns)):
        asset_signals = []
        for ticker in tickers:
            # Create a slice for the asset
            asset_data = pd.DataFrame({
                'Close': close_prices[ticker],
                'Volume': volumes[ticker]
            })
            sig_df = alpha_engine.generate_signal(asset_data)
            # Handle case where sig_df might be empty or all NaN
            if len(sig_df) > i:
                sig = sig_df['signal'].iloc[i]
                conf = sig_df['confidence'].iloc[i]
            else:
                sig, conf = 0.0, 0.0
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
            weights = np.zeros(len(tickers))
        
        btc_rets = returns.iloc[:, 0]
        lev = risk_engine.get_leverage(btc_rets.iloc[:i+1])
        final_weights = weights * lev
        
        period_return = (returns.iloc[i] @ final_weights)
        turnover = np.sum(np.abs(final_weights - current_weights))
        costs = execution_engine.calculate_cost(turnover)
        
        portfolio_value.append(portfolio_value[-1] * (1 + period_return - costs))
        current_weights = final_weights
        if np.abs(final_weights).sum() > 0:
            trade_count += 1

    print(f"Total trades executed: {trade_count}")
    strategy_rets = pd.Series(portfolio_value).pct_change().dropna()
    
    print(f"Final Portfolio Value: {portfolio_value[-1]:.4f}")
    print(f"Total Return: {(portfolio_value[-1]-1)*100:.2f}%")
    if len(strategy_rets) > 1 and strategy_rets.std() != 0:
        print(f"Sharpe Ratio: {(strategy_rets.mean()/strategy_rets.std()) * np.sqrt(252):.4f}")
    print(f"Max Drawdown: {(1 - pd.Series(portfolio_value).cummax()/pd.Series(portfolio_value)).max():.4f}")

if __name__ == '__main__':
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOT-USD', 'POL-USD', 'LINK-USD', 'AVAX-USD']
    run_year_backtest(2021, tickers)
    run_year_backtest(2024, tickers)
