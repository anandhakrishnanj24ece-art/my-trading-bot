import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
from neural_flow.alpha.model import LiquidityAlpha
from neural_flow.execution.model import StealthExecution
from neural_flow.risk.model import VolatilitySwitch

# Re-import the V2 classes to ensure they are used in the audit
class AdaptiveWeightedProV2:
    def __init__(self, lambda_base=0.005, tau_entry=0.02, tau_exit=0.005, critical_threshold=3.0):
        self.lambda_base = lambda_base
        self.tau_entry = tau_entry
        self.tau_exit = tau_exit
        self.critical_threshold = critical_threshold
        self.prev_weights = None

    def optimize(self, signals, confidences, market_momentum=1.0):
        lambda_adj = self.lambda_base * (1.0 / (1.0 + market_momentum))
        abs_sigs = np.abs(signals)
        if abs_sigs.sum() == 0: return np.zeros(len(signals))
        weights = np.sign(signals) * (abs_sigs / abs_sigs.sum())
        weights = np.where(np.abs(weights) > lambda_adj, weights, 0)
        if np.abs(weights).sum() > 0: weights = weights / np.abs(weights).sum()
        else: weights = np.zeros(len(signals))
        final_weights = []
        for w, conf in zip(weights, confidences):
            cap = 0.05 if conf < 1.0 else (0.20 if conf < 2.0 else 0.50)
            final_weights.append(np.clip(w, -cap, cap))
        weights = np.array(final_weights)
        total_exp = np.sum(np.abs(weights))
        if total_exp > 1.0: weights = weights / total_exp
        if self.prev_weights is not None:
            new_weights = np.copy(self.prev_weights)
            for i in range(len(weights)):
                if abs(signals[i]) > self.critical_threshold:
                    new_weights[i] = weights[i]
                    continue
                diff = weights[i] - self.prev_weights[i]
                if diff < 0 and abs(diff) > self.tau_exit:
                    new_weights[i] = weights[i]
                elif diff > 0 and abs(diff) > self.tau_entry:
                    new_weights[i] = weights[i]
                else:
                    new_weights[i] = self.prev_weights[i]
            weights = new_weights
        self.prev_weights = weights
        return weights

class ProRiskManagerV2:
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

def run_strategy(mode='pro_v2', tickers=[], close_prices=None, volumes=None):
    returns = close_prices.pct_change().dropna()
    common_index = returns.index.intersection(close_prices.index).intersection(volumes.index)
    returns = returns.loc[common_index]
    close_prices = close_prices.loc[common_index]
    volumes = volumes.loc[common_index]

    alpha_engine = LiquidityAlpha()
    execution_engine = StealthExecution()

    if mode == 'pro_v2':
        optimizer = AdaptiveWeightedProV2()
        risk_manager = ProRiskManagerV2()
    elif mode == 'pro_v1':
        # Simplified V1 (fixed lambda, symmetric hysteresis, la place vol)
        class AdaptiveV1:
            def __init__(self): self.prev_weights = None
            def optimize(self, signals, confs):
                abs_s = np.abs(signals)
                if abs_s.sum() == 0: return np.zeros(len(signals))
                w = np.sign(signals) * (abs_s / abs_s.sum())
                w = np.where(np.abs(w) > 0.005, w, 0)
                if np.abs(w).sum() > 0: w /= np.abs(w).sum()
                else: w = np.zeros(len(signals))
                fw = np.array([np.clip(val, - (0.05 if c < 1.0 else (0.20 if c < 2.0 else 0.50)), (0.05 if c < 1.0 else (0.20 if c < 2.0 else 0.50))) for val, c in zip(w, confs)])
                total = np.sum(np.abs(fw))
                if total > 1.0: fw /= total
                if self.prev_weights is not None:
                    fw = np.where(np.abs(fw - self.prev_weights) < 0.01, self.prev_weights, fw)
                self.prev_weights = fw
                return fw
        optimizer = AdaptiveV1()
        # Use standard VolSwitch for v1
        from neural_flow.risk.model import VolatilitySwitch
        risk_manager = VolatilitySwitch(base_leverage=4.0)
    else:
        # Standard Adaptive
        optimizer = None
        from neural_flow.risk.model import VolatilitySwitch
        risk_manager = VolatilitySwitch(base_leverage=4.0)

    portfolio_value = [1.0]
    current_weights = np.zeros(len(tickers))
    daily_rets = []

    for i in range(len(returns)):
        asset_signals, asset_confidences = [], []
        for ticker in tickers:
            asset_df = pd.DataFrame({'Close': close_prices[ticker], 'Volume': volumes[ticker]})
            res_df = alpha_engine.generate_signal(asset_df)
            if not res_df.empty and i < len(res_df):
                res = res_df.iloc[i]
                asset_signals.append(res['signal'])
                asset_confidences.append(res['confidence'])
            else:
                asset_signals.append(0.0)
                asset_confidences.append(0.0)

        if mode == 'pro_v2':
            mkt_mom = returns.iloc[i-5:i].mean().mean() * np.sqrt(252) if i > 5 else 0.0
            weights = optimizer.optimize(np.array(asset_signals), np.array(asset_confidences), market_momentum=mkt_mom)
            lev = risk_manager.get_portfolio_leverage(weights, returns.iloc[:i+1])
        elif mode == 'pro_v1':
            weights = optimizer.optimize(np.array(asset_signals), np.array(asset_confidences))
            # v1 use la place vol (approximate by using BTC as benchmark for simplicity in this audit)
            benchmark_rets = returns.iloc[:, 0]
            lev = risk_manager.get_leverage(benchmark_rets.iloc[:i+1])
        else:
            sigs = np.array(asset_signals)
            if np.abs(sigs).sum() > 0: weights = np.sign(sigs) * (np.abs(sigs) / np.abs(sigs).sum())
            else: weights = np.zeros(len(tickers))
            final_w_pre = np.array([np.clip(w, - (0.05 if c < 1.0 else (0.20 if c < 2.0 else 0.50)), (0.05 if c < 1.0 else (0.20 if c < 2.0 else 0.50))) for w, c in zip(weights, asset_confidences)])
            total_exp = np.sum(np.abs(final_w_pre))
            if total_exp > 1.0: final_w_pre /= total_exp
            weights = final_w_pre
            benchmark_rets = returns.iloc[:, 0]
            lev = risk_manager.get_leverage(benchmark_rets.iloc[:i+1])

        final_weights = weights * lev
        period_return = np.sum(final_weights * returns.iloc[i].values)
        turnover = np.sum(np.abs(final_weights - current_weights))
        costs = execution_engine.calculate_cost(turnover)

        net_return = period_return - costs
        daily_rets.append(net_return)
        portfolio_value.append(portfolio_value[-1] * (1 + net_return))
        current_weights = final_weights

    return pd.Series(daily_rets), portfolio_value

def run_full_validation():
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD']
    print("Downloading data for Full Validation...")
    data_raw = yf.download(tickers, period='1y', interval='1d')
    if isinstance(data_raw.columns, pd.MultiIndex):
        close_prices = data_raw['Close'].ffill().dropna()
        volumes = data_raw['Volume'].ffill().dropna()
    else:
        close_prices = data_raw.ffill().dropna()
        volumes = data_raw.ffill().dropna()

    tickers = close_prices.columns.tolist()

    # Run all 3 modes
    print("Calculating Standard Adaptive...")
    s_rets, s_pv = run_strategy('adaptive', tickers, close_prices, volumes)
    print("Calculating Adaptive Pro v1...")
    v1_rets, v1_pv = run_strategy('pro_v1', tickers, close_prices, volumes)
    print("Calculating Adaptive Pro v2...")
    v2_rets, v2_pv = run_strategy('pro_v2', tickers, close_prices, volumes)

    def get_metrics(rets, pv):
        ann_ret = (np.prod(1 + rets) - 1)
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = (ann_ret / ann_vol) if ann_vol != 0 else 0
        mdd = (1 - pd.Series(pv).cummax() / pd.Series(pv)).max()
        return ann_ret, sharpe, mdd

    s_m = get_metrics(s_rets, s_pv)
    v1_m = get_metrics(v1_rets, v1_pv)
    v2_m = get_metrics(v2_rets, v2_pv)

    print("\n" + "="*70)
    print(f"{'Metric':<20} | {'Standard':<15} | {'Pro v1':<15} | {'Pro v2':<15}")
    print("-" * 70)
    rows = [("Ann. Return", s_m[0], v1_m[0], v2_m[0]),
            ("Sharpe Ratio", s_m[1], v1_m[1], v2_m[1]),
            ("Max Drawdown", s_m[2], v1_m[2], v2_m[2])]
    for name, s, v1, v2 in rows:
        print(f"{name:<20} | {s:<15.4f} | {v1:<15.4f} | {v2:<15.4f}")
    print("="*70)

    # Monte Carlo for v2
    from neural_flow.validation.simulation import circular_block_bootstrap
    print("\nRunning 100,000 iteration Monte Carlo for Pro v2...")
    mc_rets = circular_block_bootstrap(v2_rets.values, n_sims=100000)
    final_vals = np.prod(1 + mc_rets, axis=1)
    print(f"MC Median: {np.median(final_vals):.4f}")
    print(f"MC 5% VaR: {np.percentile(final_vals, 5):.4f}")
    print(f"MC 1% VaR: {np.percentile(final_vals, 1):.4f}")

if __name__ == '__main__':
    run_full_validation()
