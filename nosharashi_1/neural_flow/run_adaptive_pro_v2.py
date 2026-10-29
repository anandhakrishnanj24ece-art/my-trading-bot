import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
from neural_flow.alpha.model import LiquidityAlpha
from neural_flow.execution.model import StealthExecution
from neural_flow.risk.model import VolatilitySwitch

class AdaptiveWeightedProV2:
    """
    Hyper-Adaptive Weight Optimizer.
    Implements Dynamic L1-Regularization and Asymmetric Hysteresis.
    """
    def __init__(self, lambda_base=0.005, tau_entry=0.02, tau_exit=0.005, critical_threshold=3.0):
        self.lambda_base = lambda_base
        self.tau_entry = tau_entry
        self.tau_exit = tau_exit
        self.critical_threshold = critical_threshold
        self.prev_weights = None

    def optimize(self, signals, confidences, market_momentum=1.0):
        # 1. Dynamic Regularization (Momentum-Scaled L1)
        # Higher momentum -> Lower lambda -> Capture moon moves
        lambda_adj = self.lambda_base * (1.0 / (1.0 + market_momentum))

        abs_sigs = np.abs(signals)
        if abs_sigs.sum() == 0:
            return np.zeros(len(signals))

        weights = np.sign(signals) * (abs_sigs / abs_sigs.sum())

        # Soft Thresholding with Dynamic Lambda
        weights = np.where(np.abs(weights) > lambda_adj, weights, 0)

        if np.abs(weights).sum() > 0:
            weights = weights / np.abs(weights).sum()
        else:
            weights = np.zeros(len(signals))

        # 2. Confidence Caps
        final_weights = []
        for w, conf in zip(weights, confidences):
            cap = 0.05 if conf < 1.0 else (0.20 if conf < 2.0 else 0.50)
            final_weights.append(np.clip(w, -cap, cap))
        weights = np.array(final_weights)

        total_exp = np.sum(np.abs(weights))
        if total_exp > 1.0:
            weights = weights / total_exp

        # 3. Asymmetric Hysteresis
        if self.prev_weights is not None:
            new_weights = np.copy(self.prev_weights)
            for i in range(len(weights)):
                # Critical Override: Immediate update for extreme signals
                if abs(signals[i]) > self.critical_threshold:
                    new_weights[i] = weights[i]
                    continue

                diff = weights[i] - self.prev_weights[i]
                # Fast Exit: Lower threshold for reducing exposure
                if diff < 0 and abs(diff) > self.tau_exit:
                    new_weights[i] = weights[i]
                # Slow Entry: Higher threshold for increasing exposure
                elif diff > 0 and abs(diff) > self.tau_entry:
                    new_weights[i] = weights[i]
                else:
                    new_weights[i] = self.prev_weights[i]
            weights = new_weights

        self.prev_weights = weights
        return weights

class ProRiskManagerV2:
    """
    Institutional Risk Manager.
    Implements Ledoit-Wolf Shrinkage and Portfolio-wide Vol Targeting.
    """
    def __init__(self, target_vol=0.15, base_leverage=4.0):
        self.target_vol = target_vol
        self.base_leverage = base_leverage

    def get_portfolio_leverage(self, weights, returns_df):
        if len(returns_df) < 20:
            return 1.0

        # Use Ledoit-Wolf Shrinkage for robust covariance estimation
        recent_rets = returns_df.tail(20)
        try:
            lw = LedoitWolf().fit(recent_rets)
            cov_matrix = lw.covariance_ * 252
        except:
            # Fallback to standard cov if shrinkage fails
            cov_matrix = recent_rets.cov() * 252

        port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        port_vol = np.sqrt(max(0, port_var))

        if port_vol == 0:
            return self.base_leverage

        lev = self.target_vol / port_vol
        return min(lev, self.base_leverage)

def run_adaptive_pro_v2():
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD']
    print("Downloading data for Adaptive Pro v2...")
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

    # Components
    alpha_engine = LiquidityAlpha()
    execution_engine = StealthExecution()
    optimizer = AdaptiveWeightedProV2()
    risk_manager = ProRiskManagerV2()

    portfolio_value = [1.0]
    current_weights = np.zeros(len(tickers))

    print(f"Executing Hyper-Adaptive Pro v2 on {len(tickers)} assets...")
    for i in range(len(returns)):
        # Compute Market Momentum for Dynamic Lambda
        # Simple proxy: Avg return of top 3 assets over last 5 days
        if i > 5:
            mkt_mom = returns.iloc[i-5:i].mean().mean() * np.sqrt(252)
        else:
            mkt_mom = 0.0

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

        weights = optimizer.optimize(np.array(asset_signals), np.array(asset_confidences), market_momentum=mkt_mom)
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

    print(f"\n--- ADAPTIVE PRO v2 RESULTS ---")
    print(f"Final Portfolio Value: {final_val:.4f}")
    print(f"Total Return: {total_ret:.2f}%")
    print(f"Max Drawdown: {max_dd:.4f}")

if __name__ == '__main__':
    run_adaptive_pro_v2()
