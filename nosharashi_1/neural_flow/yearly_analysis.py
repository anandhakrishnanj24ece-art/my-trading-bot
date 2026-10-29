import yfinance as yf
import pandas as pd
import numpy as np
from neural_flow.alpha.model import LiquidityAlpha
from neural_flow.execution.model import StealthExecution
from neural_flow.risk.model import VolatilitySwitch

def run_yearly_analysis():
    tickers = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT', 'DOT/USDT', 'POL/USDT', 'LINK/USDT', 'AVAX/USDT']
    yf_tickers = [t.replace('/USDT', '-USD') for t in tickers]

    print("Downloading 5 years of data for yearly analysis...")
    data_raw = yf.download(yf_tickers, period='5y', interval='1d')
    close_prices = data_raw['Close'].ffill()
    volumes = data_raw['Volume'].ffill()

    # Get the years present in the data
    all_years = close_prices.index.year.unique()
    all_years = sorted(all_years)

    alpha_engine = LiquidityAlpha()
    execution_engine = StealthExecution()
    risk_engine = VolatilitySwitch(base_leverage=4.0)

    overall_results = []

    for year in all_years:
        # Filter data for the specific year
        year_data_close = close_prices[close_prices.index.year == year].dropna()
        year_data_vol = volumes[volumes.index.year == year].dropna()

        if len(year_data_close) < 30: # Skip years with too little data
            continue

        # Calculate returns for the year
        year_returns = year_data_close.pct_change().dropna()

        # Sync indices
        common_idx = year_returns.index.intersection(year_data_close.index).intersection(year_data_vol.index)
        y_ret = year_returns.loc[common_idx]
        y_close = year_data_close.loc[common_idx]
        y_vol = year_data_vol.loc[common_idx]

        portfolio_value = [1.0]
        current_weights = np.zeros(len(tickers))

        for i in range(len(y_ret)):
            # 1. Signals
            asset_signals = []
            asset_confidences = []
            for ticker in tickers:
                yf_t = ticker.replace('/USDT', '-USD')
                # Use the full dataset for signal generation to maintain rolling windows
                # but we only care about the value at index i
                asset_df = pd.DataFrame({
                    'Close': close_prices[yf_t],
                    'Volume': volumes[yf_t]
                })
                # Align index of full data with current return index
                idx_pos = close_prices.index.get_loc(y_ret.index[i])
                res = alpha_engine.generate_signal(asset_df).iloc[idx_pos]
                asset_signals.append(res['signal'])
                asset_confidences.append(res['confidence'])

            # 2. Adaptive Weighting
            sig_arr = np.array(asset_signals)
            conf_arr = np.array(asset_confidences)
            if np.abs(sig_arr).sum() > 0:
                w = np.sign(sig_arr) * (np.abs(sig_arr) / np.abs(sig_arr).sum())
            else:
                w = np.zeros(len(tickers))

            final_w_pre = []
            for j in range(len(w)):
                c = conf_arr[j]
                cap = 0.05 if c < 1.0 else (0.20 if c < 2.0 else 0.50)
                final_w_pre.append(np.clip(w[j], -cap, cap))

            fw_pre = np.array(final_w_pre)
            total_exp = np.sum(np.abs(fw_pre))
            if total_exp > 1.0: fw_pre = fw_pre / total_exp

            # 3. Leverage
            btc_rets = y_ret.iloc[:,0]
            lev = risk_engine.get_leverage(btc_rets.iloc[:i+1])
            fw = fw_pre * lev

            # 4. PnL
            p_ret = np.sum(fw * y_ret.iloc[i].values)
            turnover = np.sum(np.abs(fw - current_weights))
            costs = execution_engine.calculate_cost(turnover)

            portfolio_value.append(portfolio_value[-1] * (1 + p_ret - costs))
            current_weights = fw

        # Yearly Metrics
        y_strat_rets = pd.Series(portfolio_value).pct_change().dropna()
        final_val = portfolio_value[-1]
        ret = (final_val - 1) * 100
        sharpe = (y_strat_rets.mean() / y_strat_rets.std() * np.sqrt(252)) if y_strat_rets.std() != 0 else 0
        mdd = (1 - pd.Series(portfolio_value).cummax() / pd.Series(portfolio_value)).max()

        overall_results.append({
            'Year': year,
            'Return': ret,
            'Sharpe': sharpe,
            'MDD': mdd,
            'FinalVal': final_val
        })

    # Print results in a table
    print("\n" + "="*60)
    print(f"{'Year':<10} | {'Return %':<15} | {'Sharpe':<10} | {'MaxDD':<10}")
    print("-" * 60)
    for r in overall_results:
        print(f"{r['Year']:<10} | {r['Return']:>13.2f}% | {r['Sharpe']:>9.4f} | {r['MDD']:>9.4f}")
    print("="*60)

if __name__ == '__main__':
    run_yearly_analysis()
