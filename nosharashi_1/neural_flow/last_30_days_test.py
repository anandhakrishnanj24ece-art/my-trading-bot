import yfinance as yf
import pandas as pd
import numpy as np
from neural_flow.alpha.model import LiquidityAlpha
from neural_flow.execution.model import StealthExecution
from neural_flow.risk.model import VolatilityTarget, StopLoss
from datetime import datetime, timedelta

def run_30_day_test():
    capital = 10000.0
    print(f"\n--- RUNNING LAST 30 DAYS TEST | START CAPITAL: ${capital} ---")
    
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD']
    
    # Need 60 days of data to have a stable 20-day window for the start of the 30-day period
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    data_raw = yf.download(tickers, start=start_date, end=end_date, interval='1d')
    
    if data_raw.empty:
        print("No data found.")
        return

    close_prices = data_raw['Close'].ffill().dropna(how='all')
    volumes = data_raw['Volume'].ffill().dropna(how='all')
    returns = close_prices.pct_change().dropna(how='all')
    
    common_index = returns.index.intersection(close_prices.index).intersection(volumes.index)
    returns = returns.loc[common_index]
    close_prices = close_prices.loc[common_index]
    volumes = volumes.loc[common_index]

    alpha_engine = LiquidityAlpha()
    execution_engine = StealthExecution()
    risk_engine = VolatilityTarget(max_cap=4.0)
    stop_loss_engine = StopLoss(stop_pct=0.10)
    stop_loss_engine = StopLoss(stop_pct=0.10)

    # Simulate only the last 30 days
    test_start_idx = max(0, len(returns) - 30)

    portfolio_value = [capital]
    current_weights = np.zeros(len(tickers))
    # Track entry prices for stop loss
    entry_prices = {ticker: 0.0 for ticker in tickers}

    print(f"\nStarting simulation from {returns.index[test_start_idx].date()} to {returns.index[-1].date()}\n")
    print(f"{'Date':<12} | {'Return %':<12} | {'Equity':<12} | {'Max Conf':<10}")
    print("-" * 50)

    for i in range(test_start_idx, len(returns)):
        # 1. Check Stop Losses
        trades_made_sl = False
        current_prices = close_prices.iloc[i]
        for ticker in tickers:
            if ticker not in close_prices.columns: continue
            price = current_prices[ticker]
            units = current_weights[tickers.index(ticker)] if len(current_weights) == len(tickers) else 0

            if units != 0:
                # Proxy for entry price: the price from 5 days ago
                entry_proxy = close_prices[ticker].iloc[max(0, i-5)]
                side = 'long' if units > 0 else 'short'
                if stop_loss_engine.should_exit(entry_proxy, price, side):
                    # Force close: remove this asset's weight
                    current_weights[tickers.index(ticker)] = 0
                    trades_made_sl = True

        # 2. Normal signal generation
        asset_signals = []
        for ticker in tickers:
            if ticker not in close_prices.columns:
                asset_signals.append((0.0, 0.0))
                continue

            asset_data = pd.DataFrame({
                'Close': close_prices[ticker],
                'Volume': volumes[ticker]
            })
            sig_df = alpha_engine.generate_signal(asset_data)
            sig = sig_df['signal'].iloc[i] if len(sig_df) > i else 0.0
            conf = sig_df['confidence'].iloc[i] if len(sig_df) > i else 0.0
            asset_signals.append((sig, conf))

        signals_raw, confs_raw = zip(*asset_signals)
        signals = np.array([s if not np.isnan(s) else 0.0 for s in signals_raw])
        confs = np.array([c if not np.isnan(c) else 0.0 for c in confs_raw])

        overall_conf = np.max(confs) if len(confs) > 0 else 0.0

        if np.abs(signals).sum() > 0:
            weights = np.sign(signals) * (np.abs(signals) / np.abs(signals).sum())
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

        day_return = (returns.iloc[i] @ final_weights)
        turnover = np.sum(np.abs(final_weights - current_weights))
        costs = execution_engine.calculate_cost(turnover)

        net_return = day_return - costs
        portfolio_value.append(portfolio_value[-1] * (1 + net_return))
        current_weights = final_weights
        
        print(f"{returns.index[i].date():<12} | {net_return*100:>11.2f}% | ${portfolio_value[-1]:>11.2f} | {overall_conf:>10.4f}")

    print("-" * 50)
    final_val = portfolio_value[-1]
    total_ret = ((final_val - capital) / capital) * 100
    print(f"FINAL EQUITY: ${final_val:.2f}")
    print(f"TOTAL 30-DAY RETURN: {total_ret:.2f}%")
    print("="*50)

if __name__ == '__main__':
    run_30_day_test()
