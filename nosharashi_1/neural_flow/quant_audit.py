import yfinance as yf
import pandas as pd
import numpy as np
from neural_flow.alpha.model import LiquidityAlpha
from neural_flow.execution.model import StealthExecution
from neural_flow.risk.model import VolatilitySwitch

def run_quant_audit():
    # 1. Setup Data
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD']
    print("Downloading data for Quant Audit...")
    data_raw = yf.download(tickers, period='1y', interval='1d')

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

    # 2. Run Strategy to get daily data
    alpha_engine = LiquidityAlpha()
    execution_engine = StealthExecution()
    risk_engine = VolatilitySwitch(base_leverage=4.0)

    portfolio_value = [1.0]
    current_weights = np.zeros(len(tickers))
    daily_strategy_rets = []
    all_predictions = []
    all_actuals = []

    for i in range(len(returns)):
        asset_signals = []
        asset_confidences = []

        day_preds = {}
        day_actuals = {}

        for ticker in tickers:
            asset_df = pd.DataFrame({'Close': close_prices[ticker], 'Volume': volumes[ticker]})
            res_df = alpha_engine.generate_signal(asset_df)

            if not res_df.empty and i < len(res_df):
                res = res_df.iloc[i]
                sig = res['signal']
                asset_signals.append(sig)
                asset_confidences.append(res['confidence'])
                day_preds[ticker] = sig
            else:
                asset_signals.append(0.0)
                asset_confidences.append(0.0)
                day_preds[ticker] = 0.0

            # The actual return for the NEXT day (if available)
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

        period_return = np.sum(final_weights * returns.iloc[i].values)
        costs = execution_engine.calculate_cost(np.sum(np.abs(final_weights - current_weights)))

        net_return = period_return - costs
        daily_strategy_rets.append(net_return)
        portfolio_value.append(portfolio_value[-1] * (1 + net_return))
        current_weights = final_weights

    strat_rets_series = pd.Series(daily_strategy_rets)

    print("\n" + "="*40)
    print("   QUANTITATIVE AUDIT RESULTS")
    print("="*40)

    # --- METRIC 1: RISK-ADJUSTED PERFORMANCE ---
    ann_ret = (np.prod(1 + strat_rets_series) - 1)
    ann_vol = strat_rets_series.std() * np.sqrt(252)
    sharpe = (ann_ret / ann_vol) if ann_vol != 0 else 0

    downside_rets = strat_rets_series[strat_rets_series < 0]
    downside_vol = downside_rets.std() * np.sqrt(252) if not downside_rets.empty else 0
    sortino = (ann_ret / downside_vol) if downside_vol != 0 else np.inf

    max_dd = (1 - pd.Series(portfolio_value).cummax() / pd.Series(portfolio_value)).max()
    calmar = (ann_ret / max_dd) if max_dd != 0 else np.inf

    print(f"\n[1] Risk-Adjusted Metrics:")
    print(f"Annualized Return: {ann_ret*100:.2f}%")
    print(f"Annualized Vol:    {ann_vol*100:.2f}%")
    print(f"Sharpe Ratio:     {sharpe:.4f}")
    print(f"Sortino Ratio:     {sortino:.4f}")
    print(f"Max Drawdown:     {max_dd*100:.2f}%")
    print(f"Calmar Ratio:     {calmar:.4f}")

    # --- METRIC 2: ALPHA & HIT RATE ---
    hits = 0
    total_preds = 0
    ic_list = []

    for day_p, day_a in zip(all_predictions, all_actuals):
        day_p_vals = []
        day_a_vals = []
        for t in tickers:
            if t in day_p and t in day_a:
                p = day_p[t]
                a = day_a[t]
                if p != 0:
                    total_preds += 1
                    if np.sign(p) == np.sign(a):
                        hits += 1
                    day_p_vals.append(p)
                    day_a_vals.append(a)

        if len(day_p_vals) > 1:
            corr = np.corrcoef(day_p_vals, day_a_vals)[0, 1]
            if not np.isnan(corr):
                ic_list.append(corr)

    hit_rate = (hits / total_preds * 100) if total_preds != 0 else 0
    avg_ic = np.mean(ic_list) if ic_list else 0

    print(f"\n[2] Alpha Quality:")
    print(f"Hit Rate:         {hit_rate:.2f}%")
    print(f"Avg Daily IC:     {avg_ic:.4f}")

    # --- METRIC 3: FACTOR EXPOSURE (BETA) ---
    # Strategy Returns vs BTC Returns
    btc_rets = returns.iloc[:, 0]
    # Align length
    common_len = min(len(strat_rets_series), len(btc_rets))
    s_rets = strat_rets_series.iloc[:common_len]
    b_rets = btc_rets.iloc[:common_len]

    covariance = np.cov(s_rets, b_rets)[0, 1]
    btc_variance = np.var(b_rets)
    beta = covariance / btc_variance if btc_variance != 0 else 0

    # Alpha = Strategy Ret - (Beta * BTC Ret)
    alpha_daily = s_rets.mean() - (beta * b_rets.mean())
    ann_alpha = alpha_daily * 252

    print(f"\n[3] Factor Decomposition:")
    print(f"Beta to BTC:       {beta:.4f}")
    print(f"Annualized Alpha: {ann_alpha*100:.2f}%")

    # --- METRIC 4: SCALABILITY (TURNOVER) ---
    turnovers = []
    current_weights = np.zeros(len(tickers))
    for i in range(len(returns)):
        # Recalculate weights just for turnover
        asset_signals = []
        for ticker in tickers:
            asset_df = pd.DataFrame({'Close': close_prices[ticker], 'Volume': volumes[ticker]})
            res_df = alpha_engine.generate_signal(asset_df)
            asset_signals.append(res_df['signal'].iloc[i] if not res_df.empty else 0.0)

        signals_arr = np.array(asset_signals)
        if np.abs(signals_arr).sum() > 0:
            weights = np.sign(signals_arr) * (np.abs(signals_arr) / np.abs(signals_arr).sum())
        else:
            weights = np.zeros(len(tickers))

        # Apply confidence caps (simplified)
        # (In real audit, we use the actual confidences from the loop)
        # For brevity in audit, we'll just use the weight changes
        final_w = weights * 4.0 # approximate lev
        turnovers.append(np.sum(np.abs(final_w - current_weights)))
        current_weights = final_w

    avg_daily_turnover = np.mean(turnovers)
    print(f"\n[4] Scalability:")
    print(f"Avg Daily Turnover: {avg_daily_turnover*100:.2f}%")

    print("="*40)

if __name__ == '__main__':
    run_quant_audit()
