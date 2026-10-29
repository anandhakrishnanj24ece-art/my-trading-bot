import yfinance as yf
import pandas as pd
import numpy as np
from neural_flow.alpha.model import LiquidityAlpha
from neural_flow.execution.model import StealthExecution
from neural_flow.risk.model import VolatilitySwitch

# Import Pro classes from the run_adaptive_pro file logic
# Since they are in a script and not a module, we redefine them here for the audit
class AdaptiveWeightedPro:
    def __init__(self, lambda_reg=0.005, hysteresis_tau=0.01):
        self.lambda_reg = lambda_reg
        self.hysteresis_tau = hysteresis_tau
        self.prev_weights = None

    def optimize(self, signals, confidences):
        abs_sigs = np.abs(signals)
        if abs_sigs.sum() == 0: return np.zeros(len(signals))
        weights = np.sign(signals) * (abs_sigs / abs_sigs.sum())
        weights = np.where(np.abs(weights) > self.lambda_reg, weights, 0)
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
            diff = np.abs(weights - self.prev_weights)
            weights = np.where(diff < self.hysteresis_tau, self.prev_weights, weights)
        self.prev_weights = weights
        return weights

class ProRiskManager:
    def __init__(self, target_vol=0.15, base_leverage=4.0):
        self.target_vol = target_vol
        self.base_leverage = base_leverage

    def get_portfolio_leverage(self, weights, returns_df):
        if len(returns_df) < 20: return 1.0
        recent_rets = returns_df.tail(20)
        cov_matrix = recent_rets.cov() * 252
        port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        port_vol = np.sqrt(max(0, port_var))
        if port_vol == 0: return self.base_leverage
        return min(self.target_vol / port_vol, self.base_leverage)

def run_strategy(mode='adaptive', tickers=[], close_prices=None, volumes=None):
    returns = close_prices.pct_change().dropna()
    common_index = returns.index.intersection(close_prices.index).intersection(volumes.index)
    returns = returns.loc[common_index]
    close_prices = close_prices.loc[common_index]
    volumes = volumes.loc[common_index]

    alpha_engine = LiquidityAlpha()
    execution_engine = StealthExecution()

    portfolio_value = [1.0]
    current_weights = np.zeros(len(tickers))
    daily_rets = []
    turnovers = []

    if mode == 'pro':
        optimizer = AdaptiveWeightedPro()
        risk_manager = ProRiskManager()
    else:
        # Standard Adaptive logic
        optimizer = None
        risk_manager = VolatilitySwitch(base_leverage=4.0)

    for i in range(len(returns)):
        asset_signals = []
        asset_confidences = []
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

        sigs = np.array(asset_signals)
        confs = np.array(asset_confidences)

        if mode == 'pro':
            weights = optimizer.optimize(sigs, confs)
            lev = risk_manager.get_portfolio_leverage(weights, returns.iloc[:i+1])
        else:
            if np.abs(sigs).sum() > 0:
                weights = np.sign(sigs) * (np.abs(sigs) / np.abs(sigs).sum())
            else:
                weights = np.zeros(len(tickers))
            # Standard logic: apply confidence caps
            final_weights_pre = []
            for w, c in zip(weights, confs):
                cap = 0.05 if c < 1.0 else (0.20 if c < 2.0 else 0.50)
                final_weights_pre.append(np.clip(w, -cap, cap))
            weights = np.array(final_weights_pre)
            total_exp = np.sum(np.abs(weights))
            if total_exp > 1.0: weights /= total_exp

            # BTC benchmark leverage
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
        turnovers.append(turnover)

    return pd.Series(daily_rets), np.mean(turnovers), portfolio_value

def run_audit():
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD']
    print("Downloading data for Audit...")
    data_raw = yf.download(tickers, period='1y', interval='1d')
    if isinstance(data_raw.columns, pd.MultiIndex):
        close_prices = data_raw['Close'].ffill().dropna()
        volumes = data_raw['Volume'].ffill().dropna()
    else:
        close_prices = data_raw.ffill().dropna()
        volumes = data_raw.ffill().dropna()

    # Filter
    tickers = close_prices.columns.tolist()

    print("\nRunning Standard Adaptive...")
    s_rets, s_turn, s_pv = run_strategy('adaptive', tickers, close_prices, volumes)

    print("Running Adaptive Pro...")
    p_rets, p_turn, p_pv = run_strategy('pro', tickers, close_prices, volumes)

    def get_metrics(rets, pv):
        ann_ret = (np.prod(1 + rets) - 1)
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = (ann_ret / ann_vol) if ann_vol != 0 else 0

        downside_rets = rets[rets < 0]
        downside_vol = downside_rets.std() * np.sqrt(252) if not downside_rets.empty else 0
        sortino = (ann_ret / downside_vol) if downside_vol != 0 else np.inf

        mdd = (1 - pd.Series(pv).cummax() / pd.Series(pv)).max()
        calmar = (ann_ret / mdd) if mdd != 0 else np.inf
        return ann_ret, sharpe, sortino, mdd, calmar

    s_metrics = get_metrics(s_rets, s_pv)
    p_metrics = get_metrics(p_rets, p_pv)

    print("\n" + "="*60)
    print(f"{'Metric':<20} | {'Standard':<15} | {'Pro':<15} | {'Improv.'}")
    print("-" * 60)
    metrics_names = ["Ann. Return", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Calmar Ratio"]
    for name, s, p in zip(metrics_names, s_metrics, p_metrics):
        improv = ((p - s) / s * 100) if s != 0 else 0
        print(f"{name:<20} | {s:<15.4f} | {p:<15.4f} | {improv:+.2f}%")

    print("-" * 60)
    print(f"{'Avg Daily Turnover':<20} | {s_turn:<15.4f} | {p_turn:<15.4f} | {(p_turn-s_turn)/s_turn*100:+.2f}%")
    print("="*60)

    # Monte Carlo
    from neural_flow.validation.simulation import circular_block_bootstrap
    print("\nRunning 10,000 iteration Monte Carlo...")
    s_mc = circular_block_bootstrap(s_rets.values, n_sims=10000)
    p_mc = circular_block_bootstrap(p_rets.values, n_sims=10000)

    s_final = np.prod(1 + s_mc, axis=1)
    p_final = np.prod(1 + p_mc, axis=1)

    print(f"Standard MC 5% VaR: {np.percentile(s_final, 5):.4f}")
    print(f"Pro MC 5% VaR:      {np.percentile(p_final, 5):.4f}")

if __name__ == '__main__':
    run_audit()
