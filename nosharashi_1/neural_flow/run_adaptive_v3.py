import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
from neural_flow.alpha.model import LiquidityAlpha
from neural_flow.execution.model import StealthExecution
from neural_flow.risk.model import VolatilitySwitch

class AdaptiveWeightedV3:
    """
    Hyper-Adaptive v3 Optimizer.
    Implements Square-Root Lasso, Elastic Hysteresis, and Probabilistic Sizing.
    """
    def __init__(self, lambda_base=0.005, tau_base=0.01, critical_threshold=3.0):
        self.lambda_base = lambda_base
        self.tau_base = tau_base
        self.critical_threshold = critical_threshold
        self.prev_weights = None

    def optimize(self, signals, confidences, market_vol=0.15):
        # 1. Square-Root Lasso (Approximate)
        # Penalty is decoupled from noise sigma, making it more robust to vol spikes.
        # We scale the penalty inversely to signal strength (SNR)
        snr = np.mean(np.abs(signals)) / (np.std(signals) + 1e-6)
        lambda_adj = self.lambda_base / (1.0 + snr)

        abs_sigs = np.abs(signals)
        if abs_sigs.sum() == 0:
            return np.zeros(len(signals))

        weights = np.sign(signals) * (abs_sigs / abs_sigs.sum())
        weights = np.where(np.abs(weights) > lambda_adj, weights, 0)

        if np.abs(weights).sum() > 0:
            weights = weights / np.abs(weights).sum()
        else:
            weights = np.zeros(len(signals))

        # 2. Probabilistic Sizing (Weight scaling by confidence)
        final_weights = []
        for w, conf in zip(weights, confidences):
            # Linear scaling of exposure based on confidence probability
            # Conf 0.0 -> 0.05 cap, Conf 3.0+ -> 0.50 cap
            cap = 0.05 + (np.clip(conf, 0, 3) / 3.0) * (0.50 - 0.05)
            final_weights.append(np.clip(w, -cap, cap))
        weights = np.array(final_weights)

        total_exp = np.sum(np.abs(weights))
        if total_exp > 1.0:
            weights = weights / total_exp

        # 3. Elastic Band Hysteresis (Volatility-Adaptive Gating)
        if self.prev_weights is not None:
            new_weights = np.copy(self.prev_weights)
            # Shrink bands during high vol to react faster
            vol_scale = 1.0 / (1.0 + market_vol * 10)
            tau_adj = self.tau_base * vol_scale

            for i in range(len(weights)):
                if abs(signals[i]) > self.critical_threshold:
                    new_weights[i] = weights[i]
                    continue

                diff = weights[i] - self.prev_weights[i]
                # Fast exit (tau_exit = 0.5 * tau_adj), Slow entry (tau_entry = 1.5 * tau_adj)
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
    """
    Institutional Risk Manager v3.
    Implements Ledoit-Wolf and Portfolio Vol Targeting with Convexity scaling.
    """
    def __init__(self, target_vol=0.15, base_leverage=4.0):
        self.target_vol = target_vol
        self.base_leverage = base_leverage

    def get_portfolio_leverage(self, weights, returns_df):
        if len(returns_df) < 20:
            return 1.0

        recent_rets = returns_df.tail(20)
        try:
            lw = LedoitWolf().fit(recent_rets)
            cov_matrix = lw.covariance_ * 252
        except:
            cov_matrix = recent_rets.cov() * 252

        port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        port_vol = np.sqrt(max(0, port_var))

        if port_vol == 0:
            return self.base_leverage

        # Convexity Scaling: Allow slightly more leverage in low-vol/high-momentum regimes
        # (Simplified for this implementation as a smooth scaling factor)
        lev = self.target_vol / port_vol
        return min(lev, self.base_leverage)

def run_adaptive_v3():
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD']
    print("Downloading data for Adaptive v3...")
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

    alpha_engine = LiquidityAlpha()
    execution_engine = StealthExecution()
    optimizer = AdaptiveWeightedV3()
    risk_manager = ProRiskManagerV3()

    # Starting capital baseline: $10,000
    start_capital = 10000.0
    portfolio_value = [start_capital]
    current_weights = np.zeros(len(tickers))

    print(f"Executing Hyper-Adaptive v3 on {len(tickers)} assets...")
    for i in range(len(returns)):
        # Market Vol for Elastic Bands
        mkt_vol = returns.iloc[:i+1].std().mean() * np.sqrt(252) if i > 20 else 0.15

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

        weights = optimizer.optimize(np.array(asset_signals), np.array(asset_confidences), market_vol=mkt_vol)
        lev = risk_manager.get_portfolio_leverage(weights, returns.iloc[:i+1])
        final_weights = weights * lev

        period_return = np.sum(final_weights * returns.iloc[i].values)
        turnover = np.sum(np.abs(final_weights - current_weights))
        costs = execution_engine.calculate_cost(turnover)

        portfolio_value.append(portfolio_value[-1] * (1 + period_return - costs))
        current_weights = final_weights

    final_val = portfolio_value[-1]
    total_ret = (final_val - 1) * 100
    max_dd = (1 - pd.Series(portfolio_value).cummax() / pd.Series(portfolio_value)).max()

    # Save current balance for persistence
    with open('paper_balance.txt', 'w') as f:
        f.write(str(final_val))
    with open('paper_balance_v3.txt', 'w') as f:
        f.write(str(final_val))

    print(f"\n--- ADAPTIVE v3 RESULTS ---")
    print(f"Current Portfolio Value: ${final_val:.2f}")
    print(f"Total Return from Base: {((final_val - 10000.0)/10000.0*100):.2f}%")
    print(f"Max Drawdown: {max_dd:.4f}")

if __name__ == '__main__':
    run_adaptive_v3()
