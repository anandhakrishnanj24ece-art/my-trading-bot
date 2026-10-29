import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
from neural_flow.alpha.model import LiquidityAlpha
from neural_flow.execution.model import StealthExecution
from neural_flow.risk.model import VolatilitySwitch

# Re-implementing V3 classes for a consolidated audit
class AdaptiveWeightedV3:
    def __init__(self, lambda_base=0.005, tau_base=0.01, critical_threshold=3.0):
        self.lambda_base = lambda_base
        self.tau_base = tau_base
        self.critical_threshold = critical_threshold
        self.prev_weights = None

    def optimize(self, signals, confidences, market_vol=0.15):
        snr = np.mean(np.abs(signals)) / (np.std(signals) + 1e-6)
        lambda_adj = self.lambda_base / (1.0 + snr)
        abs_sigs = np.abs(signals)
        if abs_sigs.sum() == 0: return np.zeros(len(signals))
        weights = np.sign(signals) * (abs_sigs / abs_sigs.sum())
        weights = np.where(np.abs(weights) > lambda_adj, weights, 0)
        if np.abs(weights).sum() > 0: weights = weights / np.abs(weights).sum()
        else: weights = np.zeros(len(signals))
        final_weights = []
        for w, conf in zip(weights, confidences):
            cap = 0.05 + (np.clip(conf, 0, 3) / 3.0) * (0.50 - 0.05)
            final_weights.append(np.clip(w, -cap, cap))
        weights = np.array(final_weights)
        total_exp = np.sum(np.abs(weights))
        if total_exp > 1.0: weights = weights / total_exp
        if self.prev_weights is not None:
            new_weights = np.copy(self.prev_weights)
            vol_scale = 1.0 / (1.0 + market_vol * 10)
            tau_adj = self.tau_base * vol_scale
            for i in range(len(weights)):
                if abs(signals[i]) > self.critical_threshold:
                    new_weights[i] = weights[i]
                    continue
                diff = weights[i] - self.prev_weights[i]
                if diff < 0 and abs(diff) > (0.5 * tau_adj):
                    new_weights[i] = weights[i]
                elif diff > 0 and abs(diff) > (1.5 * tau_adj):
                    new_weights[i] = weights[i]
                else:
                    new_weights[i] = self.prev_weights[i]
            weights = new_weights
        self.prev_weights = weights
        return weights

class ProRiskManagerV3:
    def __init__(self, target_vol=0.15, base_leverage=4.0):
        self.target_vol = target_vol
        self.base_leverage = base_leverage

    def get_portfolio_leverage(self, weights, returns_df):
        if len(returns_df) < 20: return 1.0
        recent_rets = returns_df.tail(20)
        try:
            lw = LedoitWolf().fit(recent_rets)
            cov_matrix = lw.covariance_ * 252
        except:
            cov_matrix = recent_rets.cov() * 252
        port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        port_vol = np.sqrt(max(0, port_var))
        if port_vol == 0: return self.base_leverage
        return min(self.target_vol / port_vol, self.base_leverage)

def run_deep_validation():
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD']
    print("Downloading data for Deep Validation...")
    data_raw = yf.download(tickers, period='2y', interval='1d') # 2Y for more robustness

    if isinstance(data_raw.columns, pd.MultiIndex):
        close_prices = data_raw['Close'].ffill().dropna()
        volumes = data_raw['Volume'].ffill().dropna()
    else:
        close_prices = data_raw.ffill().dropna()
        volumes = data_raw.ffill().dropna()

    tickers_list = close_prices.columns.tolist()
    returns = close_prices.pct_change().dropna()
    common_index = returns.index.intersection(close_prices.index).intersection(volumes.index)
    returns = returns.loc[common_index]
    close_prices = close_prices.loc[common_index]
    volumes = volumes.loc[common_index]

    alpha_engine = LiquidityAlpha()
    execution_engine = StealthExecution()
    optimizer = AdaptiveWeightedV3()
    risk_manager = ProRiskManagerV3()

    portfolio_value = [1.0]
    current_weights = np.zeros(len(tickers_list))
    daily_rets = []
    daily_turnovers = []
    equity_curve = [1.0]

    print(f"Executing Deep Validation on {len(tickers_list)} assets...")
    for i in range(len(returns)):
        mkt_vol = returns.iloc[:i+1].std().mean() * np.sqrt(252) if i > 20 else 0.15
        asset_signals, asset_confidences = [], []
        for ticker in tickers_list:
            asset_df = pd.DataFrame({'Close': close_prices[ticker], 'Volume': volumes[ticker]})
            res_df = alpha_engine.generate_signal(asset_df)
            if not res_df.empty and i < len(res_df):
                res = res_df.iloc[i]
                asset_signals.append(res['signal'])
                asset_confidences.append(res['confidence'])
            else:
                asset_signals.append(0.0)
                asset_confidences.append(0.0)

        weights = optimizer.optimize(np.array(asset_signals), np.array(asset_confidences), market_vol=mkt_vol)
        lev = risk_manager.get_portfolio_leverage(weights, returns.iloc[:i+1])
        final_weights = weights * lev

        period_return = np.sum(final_weights * returns.iloc[i].values)
        turnover = np.sum(np.abs(final_weights - current_weights))
        costs = execution_engine.calculate_cost(turnover)

        net_return = period_return - costs
        daily_rets.append(net_return)
        portfolio_value.append(portfolio_value[-1] * (1 + net_return))
        current_weights = final_weights
        daily_turnovers.append(turnover)

    strat_rets = pd.Series(daily_rets)
    pv_series = pd.Series(portfolio_value)

    # --- QUANT METRICS ---
    ann_ret = (np.prod(1 + strat_rets) - 1)
    ann_vol = strat_rets.std() * np.sqrt(252)
    sharpe = (ann_ret / ann_vol) if ann_vol != 0 else 0

    downside_rets = strat_rets[strat_rets < 0]
    downside_vol = downside_rets.std() * np.sqrt(252) if not downside_rets.empty else 0
    sortino = (ann_ret / downside_vol) if downside_vol != 0 else np.inf

    mdd = (1 - pv_series.cummax() / pv_series).max()
    calmar = (ann_ret / mdd) if mdd != 0 else np.inf

    print("\n" + "="*50)
    print("   DEEP QUANT VALIDATION: ADAPTIVE V3")
    print("="*50)
    print(f"Total Return:       {ann_ret*100:.2f}%")
    print(f"Annualized Vol:     {ann_vol*100:.2f}%")
    print(f"Sharpe Ratio:       {sharpe:.4f}")
    print(f"Sortino Ratio:      {sortino:.4f}")
    print(f"Max Drawdown:       {mdd*100:.2f}%")
    print(f"Calmar Ratio:       {calmar:.4f}")
    print(f"Avg Daily Turnover: {np.mean(daily_turnovers)*100:.2f}%")
    print("="*50)

    # --- MONTE CARLO 100,000 ---
    from neural_flow.validation.simulation import circular_block_bootstrap
    print("\nRunning 100,000 Iteration Monte Carlo...")
    mc_rets = circular_block_bootstrap(strat_rets.values, n_sims=100000)
    final_vals = np.prod(1 + mc_rets, axis=1)

    print(f"MC Median:      {np.median(final_vals):.4f}")
    print(f"MC 5% VaR:      {np.percentile(final_vals, 5):.4f}")
    print(f"MC 1% VaR:      {np.percentile(final_vals, 1):.4f}")
    print(f"Prob. of Loss:  {(np.sum(final_vals < 1.0) / 100000)*100:.2f}%")
    print("="*50)

if __name__ == '__main__':
    run_deep_validation()
