import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
from importlib import reload

# We will implement a function that runs the audit for a specific path
# and returns the metrics as a dictionary.

def run_strategy_audit(version_name, root_path):
    print(f"\n>>> Running Quant Audit for: {version_name}")

    # Clear existing neural_flow from sys.modules to avoid conflicts
    # since we are swapping the implementation at runtime.
    modules_to_remove = [m for m in sys.modules if m.startswith('neural_flow')]
    for m in modules_to_remove:
        del sys.modules[m]

    # Add the specific path to the front of sys.path
    # root_path should be the directory containing the 'neural_flow' package
    sys.path.insert(0, root_path)

    try:
        # Import the components from the selected path
        from neural_flow.alpha.model import LiquidityAlpha
        from neural_flow.execution.model import StealthExecution
        from neural_flow.risk.model import VolatilitySwitch

        # Setup Data (identical for both)
        tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD']
        data_raw = yf.download(tickers, period='1y', interval='1d', progress=False)

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

        # Run Strategy
        alpha_engine = LiquidityAlpha()
        execution_engine = StealthExecution()
        risk_engine = VolatilitySwitch(base_leverage=4.0)

        portfolio_value = [1.0]
        current_weights = np.zeros(len(tickers))
        daily_strategy_rets = []
        all_predictions = []
        all_actuals = []

        # Weights formed on bar i can only earn bar i+1's return.
        for i in range(len(returns) - 1):
            asset_signals = []
            asset_confidences = []
            day_preds = {}
            day_actuals = {}

            for ticker in tickers:
                asset_df = pd.DataFrame({'Close': close_prices[ticker], 'Volume': volumes[ticker]})
                res_df = alpha_engine.generate_signal(asset_df)
                if not res_df.empty and i < len(res_df):
                    res = res_df.iloc[i]
                    sig, conf = res['signal'], res['confidence']
                    asset_signals.append(sig)
                    asset_confidences.append(conf)
                    day_preds[ticker] = sig
                else:
                    asset_signals.append(0.0)
                    asset_confidences.append(0.0)
                    day_preds[ticker] = 0.0
                if i + 1 < len(returns):
                    day_actuals[ticker] = returns.iloc[i+1][ticker]

            all_predictions.append(day_preds)
            all_actuals.append(day_actuals)

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

            benchmark_rets = returns.iloc[:, 0]
            lev = risk_engine.get_leverage(benchmark_rets.iloc[:i+1])
            final_weights = final_weights_pre_lev * lev

            period_return = np.sum(final_weights * returns.iloc[i + 1].values)
            costs = execution_engine.calculate_cost(np.sum(np.abs(final_weights - current_weights)))
            net_return = period_return - costs
            daily_strategy_rets.append(net_return)
            portfolio_value.append(portfolio_value[-1] * (1 + net_return))
            current_weights = final_weights

        # Calculate Metrics
        strat_rets_series = pd.Series(daily_strategy_rets)
        total_ret = (np.prod(1 + strat_rets_series) - 1)
        ann_ret = ((1 + total_ret) ** (252.0 / len(strat_rets_series)) - 1) if len(strat_rets_series) else 0.0
        ann_vol = strat_rets_series.std() * np.sqrt(252)
        sharpe = (ann_ret / ann_vol) if ann_vol != 0 else 0

        downside_rets = strat_rets_series[strat_rets_series < 0]
        downside_vol = downside_rets.std() * np.sqrt(252) if not downside_rets.empty else 0
        sortino = (ann_ret / downside_vol) if downside_vol != 0 else np.inf

        max_dd = (1 - pd.Series(portfolio_value) / pd.Series(portfolio_value).cummax()).max()
        calmar = (ann_ret / max_dd) if max_dd != 0 else np.inf

        hits, total_preds = 0, 0
        ic_list = []
        for day_p, day_a in zip(all_predictions, all_actuals):
            day_p_vals, day_a_vals = [], []
            for t in tickers:
                if t in day_p and t in day_a:
                    p, a = day_p[t], day_a[t]
                    if p != 0:
                        total_preds += 1
                        if np.sign(p) == np.sign(a): hits += 1
                        day_p_vals.append(p)
                        day_a_vals.append(a)
            if len(day_p_vals) > 1:
                corr = np.corrcoef(day_p_vals, day_a_vals)[0, 1]
                if not np.isnan(corr): ic_list.append(corr)

        hit_rate = (hits / total_preds * 100) if total_preds != 0 else 0
        avg_ic = np.mean(ic_list) if ic_list else 0

        btc_rets = returns.iloc[:, 0]
        common_len = min(len(strat_rets_series), len(btc_rets))
        s_rets, b_rets = strat_rets_series.iloc[:common_len], btc_rets.iloc[:common_len]
        covariance = np.cov(s_rets, b_rets)[0, 1]
        btc_variance = np.var(b_rets)
        beta = covariance / btc_variance if btc_variance != 0 else 0
        ann_alpha = (s_rets.mean() - (beta * b_rets.mean())) * 252

        turnovers = []
        current_weights = np.zeros(len(tickers))
        for i in range(len(returns)):
            asset_signals = []
            for ticker in tickers:
                asset_df = pd.DataFrame({'Close': close_prices[ticker], 'Volume': volumes[ticker]})
                res_df = alpha_engine.generate_signal(asset_df)
                asset_signals.append(res_df['signal'].iloc[i] if not res_df.empty else 0.0)
            signals_arr = np.array(asset_signals)
            weights = np.sign(signals_arr) * (np.abs(signals_arr) / np.abs(signals_arr).sum()) if np.abs(signals_arr).sum() > 0 else np.zeros(len(tickers))
            final_w = weights * 4.0
            turnovers.append(np.sum(np.abs(final_w - current_weights)))
            current_weights = final_w
        avg_turnover = np.mean(turnovers)

        return {
            "Ann. Return": ann_ret,
            "Ann. Vol": ann_vol,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Max DD": max_dd,
            "Calmar": calmar,
            "Hit Rate": hit_rate,
            "Avg IC": avg_ic,
            "Beta to BTC": beta,
            "Ann. Alpha": ann_alpha,
            "Avg Turnover": avg_turnover
        }
    finally:
        # Clean up sys.path
        if root_path in sys.path:
            sys.path.remove(root_path)

if __name__ == '__main__':
    # Define paths
    # Root directory is current working directory
    root_dir = os.getcwd()
    # nosharashi_1 directory
    nosharashi_1_dir = os.path.join(root_dir, 'nosharashi_1')

    results_root = run_strategy_audit("nosharashi (Root)", root_dir)
    results_1 = run_strategy_audit("nosharashi_1", nosharashi_1_dir)

    df_comp = pd.DataFrame([results_root, results_1], index=['nosharashi', 'nosharashi_1'])

    print("\n" + "="*60)
    print("      ADAPTIVE WEIGHTED STRATEGY: QUANT COMPARISON")
    print("="*60)
    print(df_comp.to_string())
    print("="*60)

    # Determine winner based on Sharpe and Hit Rate
    if results_1['Sharpe'] > results_root['Sharpe']:
        winner = "nosharashi_1"
    else:
        winner = "nosharashi (Root)"

    print(f"\nWINNER based on Sharpe Ratio: {winner}")
