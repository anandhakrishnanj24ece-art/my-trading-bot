import yfinance as yf
import pandas as pd
import numpy as np
from neural_flow.alpha.model import LiquidityAlpha
from neural_flow.execution.model import StealthExecution
from neural_flow.risk.model import VolatilitySwitch

class AdaptiveWeightedPro:
    """
    Institutional-grade weight optimizer.
    Implements L1-Regularization and Hysteresis Banding to reduce turnover.
    """
    def __init__(self, lambda_reg=0.01, hysteresis_tau=0.01):
        self.lambda_reg = lambda_reg
        self.hysteresis_tau = hysteresis_tau
        self.prev_weights = None

    def optimize(self, signals, confidences):
        """
        L1-Regularized weight optimization.
        Objective: Maximize (sum(w*sig) - lambda * sum|w|)
        Simplified implementation using soft-thresholding for L1.
        """
        # 1. Initial proportional weights
        abs_sigs = np.abs(signals)
        if abs_sigs.sum() == 0:
            return np.zeros(len(signals))

        weights = np.sign(signals) * (abs_sigs / abs_sigs.sum())

        # 2. L1-Regularization (Soft Thresholding)
        # We penalize weights that are too small relative to the lambda_reg
        # Effectively pruning low-conviction assets
        weights = np.where(np.abs(weights) > self.lambda_reg, weights, 0)

        # 3. Re-normalize after pruning
        if np.abs(weights).sum() > 0:
            weights = weights / np.abs(weights).sum()
        else:
            weights = np.zeros(len(signals))

        # 4. Apply Confidence Caps (Adaptive logic)
        final_weights = []
        for w, conf in zip(weights, confidences):
            cap = 0.05 if conf < 1.0 else (0.20 if conf < 2.0 else 0.50)
            final_weights.append(np.clip(w, -cap, cap))

        weights = np.array(final_weights)

        # Normalize again after caps
        total_exp = np.sum(np.abs(weights))
        if total_exp > 1.0:
            weights = weights / total_exp

        # 5. Hysteresis Banding
        if self.prev_weights is not None:
            # Only update weight if the change exceeds the threshold tau
            diff = np.abs(weights - self.prev_weights)
            weights = np.where(diff < self.hysteresis_tau, self.prev_weights, weights)

        self.prev_weights = weights
        return weights

class ProRiskManager:
    """
    Advanced Risk Manager.
    Implements Portfolio-wide Volatility Targeting and Dynamic Scaling.
    """
    def __init__(self, target_vol=0.15, base_leverage=4.0):
        self.target_vol = target_vol
        self.base_leverage = base_leverage

    def get_portfolio_leverage(self, weights, returns_df):
        """
        Calculates leverage based on Portfolio Volatility: sigma_p = sqrt(w^T * Sigma * w)
        """
        if len(returns_df) < 20:
            return 1.0

        # Use last 20 days of returns
        recent_rets = returns_df.tail(20)
        cov_matrix = recent_rets.cov() * 252 # Annualized Covariance

        # Portfolio Variance = w^T * Sigma * w
        port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        port_vol = np.sqrt(max(0, port_var))

        if port_vol == 0:
            return self.base_leverage

        # Scale factor: Target Vol / Realized Portfolio Vol
        lev = self.target_vol / port_vol
        return min(lev, self.base_leverage)

def run_adaptive_pro():
    # 1. Setup
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD']
    print("Downloading data for Adaptive Pro...")
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

    # Initialize Pro Components
    alpha_engine = LiquidityAlpha()
    execution_engine = StealthExecution()
    optimizer = AdaptiveWeightedPro(lambda_reg=0.005, hysteresis_tau=0.01)
    risk_manager = ProRiskManager(target_vol=0.15, base_leverage=4.0)

    portfolio_value = [1.0]
    current_weights = np.zeros(len(tickers))

    print(f"Executing Adaptive Pro Strategy on {len(tickers)} assets...")
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

        # Optimized Weights
        weights = optimizer.optimize(np.array(asset_signals), np.array(asset_confidences))

        # Portfolio-wide Vol Targeting
        lev = risk_manager.get_portfolio_leverage(weights, returns.iloc[:i+1])
        final_weights = weights * lev

        # PnL and Costs
        period_return = np.sum(final_weights * returns.iloc[i].values)
        turnover = np.sum(np.abs(final_weights - current_weights))
        costs = execution_engine.calculate_cost(turnover)

        portfolio_value.append(portfolio_value[-1] * (1 + period_return - costs))
        current_weights = final_weights

    final_val = portfolio_value[-1]
    total_ret = (final_val - 1) * 100
    max_dd = (1 - pd.Series(portfolio_value).cummax() / pd.Series(portfolio_value)).max()

    print(f"\n--- ADAPTIVE PRO STRATEGY RESULTS ---")
    print(f"Final Portfolio Value: {final_val:.4f}")
    print(f"Total Return: {total_ret:.2f}%")
    print(f"Max Drawdown: {max_dd:.4f}")

if __name__ == '__main__':
    run_adaptive_pro()
