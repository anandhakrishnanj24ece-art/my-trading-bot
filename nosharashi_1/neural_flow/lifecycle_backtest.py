import yfinance as yf
import pandas as pd
import numpy as np
from neural_flow.alpha.model import LiquidityAlpha
from neural_flow.execution.model import StealthExecution
from neural_flow.risk.model import VolatilitySwitch
from datetime import datetime, timedelta

def run_lifecycle_analysis():
    capital = 10000.0
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD'] # Simplified for clarity
    
    # Fetch data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    data_raw = yf.download(tickers, start=start_date, end=end_date, interval='1d')
    
    close_prices = data_raw['Close'].ffill().dropna()
    volumes = data_raw['Volume'].ffill().dropna()
    returns = close_prices.pct_change().dropna()
    
    common_index = returns.index.intersection(close_prices.index).intersection(volumes.index)
    returns = returns.loc[common_index]
    close_prices = close_prices.loc[common_index]
    volumes = volumes.loc[common_index]

    alpha_engine = LiquidityAlpha()
    risk_engine = VolatilitySwitch(base_leverage=4.0)
    
    portfolio_value = [capital]
    current_weights = np.zeros(len(tickers))
    
    print(f"\n{'Date':<12} | {'Ticker':<8} | {'Signal':<8} | {'Conf':<6} | {'Weight':<10} | {'Action'}")
    print("-" * 70)

    # We only simulate the last 15 days to keep the log readable
    test_start_idx = max(0, len(returns) - 15)
    
    for i in range(test_start_idx, len(returns)):
        asset_signals = []
        for ticker in tickers:
            asset_data = pd.DataFrame({'Close': close_prices[ticker], 'Volume': volumes[ticker]})
            sig_df = alpha_engine.generate_signal(asset_data)
            asset_signals.append((sig_df['signal'].iloc[i], sig_df['confidence'].iloc[i]))
        
        signals_raw, confs_raw = zip(*asset_signals)
        signals = np.array([s if not np.isnan(s) else 0.0 for s in signals_raw])
        confs = np.array([c if not np.isnan(c) else 0.0 for c in confs_raw])
        
        # Adaptive Weighting
        if np.abs(signals).sum() > 0:
            weights = np.sign(signals) * (np.abs(signals) / np.abs(signals).sum())
            overall_conf = np.max(confs)
            cap = 0.05 if overall_conf < 1.0 else (0.20 if overall_conf < 2.0 else 0.50)
            weights = np.clip(weights, -cap, cap)
            total_exp = np.sum(np.abs(weights))
            if total_exp > 0: weights = weights / total_exp
        else:
            weights = np.zeros(len(tickers))
        
        lev = risk_engine.get_leverage(returns.iloc[:, 0].iloc[:i+1])
        final_weights = weights * lev
        
        # --- ANALYSIS OF CLOSING TRADES ---
        for idx, ticker in enumerate(tickers):
            old_w = current_weights[idx]
            new_w = final_weights[idx]
            
            if abs(new_w - old_w) > 0.01:
                # Determine action type
                if old_w > 0 and new_w <= 0: action = "CLOSE LONG (Flip)"
                elif old_w < 0 and new_w >= 0: action = "CLOSE SHORT (Flip)"
                elif abs(new_w) < abs(old_w): action = "REDUCE (Risk Off)"
                elif abs(new_w) > abs(old_w): action = "INCREASE (Strike)"
                else: action = "ADJUST"
                
                print(f"{returns.index[i].date():<12} | {ticker[:3]:<8} | {signals[idx]:>8.2f} | {confs[idx]:>6.2f} | {new_w:>9.2%} | {action}")
        
        current_weights = final_weights
        day_return = (returns.iloc[i] @ final_weights)
        portfolio_value.append(portfolio_value[-1] * (1 + day_return))

if __name__ == '__main__':
    run_lifecycle_analysis()
