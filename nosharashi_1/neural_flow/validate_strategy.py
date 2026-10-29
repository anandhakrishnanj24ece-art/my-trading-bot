import yfinance as yf
import pandas as pd
import numpy as np
from neural_flow.alpha.model import LiquidityAlpha
from neural_flow.execution.model import StealthExecution
from neural_flow.risk.model import VolatilitySwitch

def run_validation_suite():
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD']
    print("Downloading data...")
    data_raw = yf.download(tickers, period='2y', interval='1d')

    if isinstance(data_raw.columns, pd.MultiIndex):
        close_prices = data_raw['Close'].ffill().dropna()
        volumes = data_raw['Volume'].ffill().dropna()
    else:
        close_prices = data_raw.ffill().dropna()
        volumes = data_raw.ffill().dropna()

    tickers = close_prices.columns.tolist()
    returns = close_prices.pct_change().dropna()
    common_index = returns.index.intersection(close_prices.index).intersection(volumes.index)
    returns = returns.loc[common_index]
    close_prices = close_prices.loc[common_index]
    volumes = volumes.loc[common_index]

    # --- VALIDATION 1: TRADE LOG & BASE RUN ---
    print("\n--- [VALIDATION 1] Generating Trade Log ---")
    alpha_engine = LiquidityAlpha()
    execution_engine = StealthExecution()
    risk_engine = VolatilitySwitch(base_leverage=4.0)

    portfolio_value = [1.0]
    current_weights = np.zeros(len(tickers))
    trade_log = []

    # We run on the last year for the detailed log
    test_returns = returns.tail(365)
    test_close = close_prices.tail(365)
    test_vol = volumes.tail(365)

    for i in range(len(test_returns)):
        asset_signals = []
        asset_confidences = []
        for ticker in tickers:
            asset_df = pd.DataFrame({'Close': test_close[ticker], 'Volume': test_vol[ticker]})
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

        if np.abs(signals_arr).sum() > 0:
            weights = np.sign(signals_arr) * (np.abs(signals_arr) / np.abs(signals_arr).sum())
        else:
            weights = np.zeros(len(tickers))

        final_weights_pre_lev = []
        for j in range(len(weights)):
            conf = conf_arr[j]
            cap = 0.05 if conf < 1.0 else (0.20 if conf < 2.0 else 0.50)
            final_weights_pre_lev.append(np.clip(weights[j], -cap, cap))

        final_weights_pre_lev = np.array(final_weights_pre_lev)
        total_exp = np.sum(np.abs(final_weights_pre_lev))
        if total_exp > 1.0:
            final_weights_pre_lev /= total_exp

        # Dynamic Leverage based on the first available ticker (usually BTC)
        # The risk_engine expects a Series of returns for a SINGLE asset.
        # We provide the returns of the first asset in our tickers list.
        benchmark_rets = test_returns.iloc[:, 0]
        lev = risk_engine.get_leverage(benchmark_rets.iloc[:i+1])
        final_weights = final_weights_pre_lev * lev

        period_return = np.sum(final_weights * test_returns.iloc[i].values)
        costs = execution_engine.calculate_cost(np.sum(np.abs(final_weights - current_weights)))

        portfolio_value.append(portfolio_value[-1] * (1 + period_return - costs))
        current_weights = final_weights

        trade_log.append({
            'date': test_returns.index[i],
            'return': period_return - costs,
            'lev': lev,
            'weights': final_weights
        })

    log_df = pd.DataFrame(trade_log)
    log_df.to_csv('trade_log.csv', index=False)
    print(f"Trade log saved to trade_log.csv. Final Value: {portfolio_value[-1]:.4f}")

    # --- VALIDATION 2: PARAMETER STABILITY ---
    print("\n--- [VALIDATION 2] Parameter Stability (Leverage Sweep) ---")
    leverages = [3.0, 3.5, 4.0, 4.5, 5.0]
    stability_results = {}
    for lev_val in leverages:
        risk_engine_sweep = VolatilitySwitch(base_leverage=lev_val)
        pv_sweep = [1.0]
        cw_sweep = np.zeros(len(tickers))
        for i in range(len(test_returns)):
            # Use benchmark asset returns for the leverage sweep as well
            benchmark_rets_sweep = test_returns.iloc[:, 0]
            lev_sweep = risk_engine_sweep.get_leverage(benchmark_rets_sweep.iloc[:i+1])
            # Using weights from the base run log
            w = trade_log[i]['weights'] / (trade_log[i]['lev'] if trade_log[i]['lev'] != 0 else 1)
            final_w = w * lev_sweep
            p_ret = np.sum(final_w * test_returns.iloc[i].values)
            c = execution_engine.calculate_cost(np.sum(np.abs(final_w - cw_sweep)))
            pv_sweep.append(pv_sweep[-1] * (1 + p_ret - c))
            cw_sweep = final_w
        stability_results[lev_val] = pv_sweep[-1]
        print(f"Leverage {lev_val}x -> Final Value: {pv_sweep[-1]:.4f}")

    # --- VALIDATION 3: WALK-FORWARD ---
    print("\n--- [VALIDATION 3] Walk-Forward (Train 6m / Test 6m) ---")
    mid = len(test_returns) // 2
    train_rets = test_returns.iloc[:mid]
    test_rets_wf = test_returns.iloc[mid:]

    # Note: Since our strategy is non-parametric (fixed rules),
    # walk-forward here means checking performance on the second half
    # using the logic trained/defined on the first half.
    pv_wf = [1.0]
    cw_wf = np.zeros(len(tickers))
    for i in range(len(test_rets_wf)):
        # Re-calculate signals for the a-priori window
        # Using the logic from the loop above
        # Simplified for script: using existing signals from the log
        idx = mid + i
        w = trade_log[idx]['weights'] / (trade_log[idx]['lev'] if trade_log[idx]['lev'] != 0 else 1)

        # FIX: Pass only the benchmark asset returns to the risk engine
        benchmark_rets_wf = test_returns.iloc[:, 0]
        lev_wf = risk_engine.get_leverage(benchmark_rets_wf.iloc[:mid+i])

        final_w = w * lev_wf
        p_ret = np.sum(final_w * test_rets_wf.iloc[i].values)
        c = execution_engine.calculate_cost(np.sum(np.abs(final_w - cw_wf)))
        pv_wf.append(pv_wf[-1] * (1 + p_ret - c))
        cw_wf = final_w
    print(f"Out-of-Sample (Last 6m) Final Value: {pv_wf[-1]:.4f}")

    # --- VALIDATION 4: MONTE CARLO ---
    print("\n--- [VALIDATION 4] Monte Carlo (1000 runs) ---")
    daily_rets = pd.Series(portfolio_value).pct_change().dropna().values
    mc_results = []
    for _ in range(1000):
        sim = np.random.choice(daily_rets, size=len(daily_rets), replace=True)
        mc_results.append(np.prod(1 + sim))

    var_5 = np.percentile(mc_results, 5)
    var_1 = np.percentile(mc_results, 1)
    print(f"Monte Carlo Median: {np.median(mc_results):.4f}")
    print(f"5% VaR (Worst Case): {var_5:.4f}")
    print(f"1% VaR (Extreme Case): {var_1:.4f}")

if __name__ == '__main__':
    run_validation_suite()
