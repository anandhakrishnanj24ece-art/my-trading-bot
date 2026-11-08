import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

class DCCGARCHRiskModel:
    """
    Dynamic Conditional Correlation (DCC) Risk Model.
    Estimates dynamic correlations to optimize portfolio weights.
    """
    def __init__(self, target_vol=0.15, base_leverage=4.0):
        self.target_vol = target_vol
        self.base_leverage = base_leverage

    def estimate_covariance(self, returns_df: pd.DataFrame) -> np.ndarray:
        """
        Estimates the dynamic covariance matrix.
        Implements a Ledoit-Wolf shrinkage as a robust proxy for DCC covariance.
        """
        if len(returns_df) < 20:
            return returns_df.cov().values * 252

        # Using Ledoit-Wolf for shrinkage to handle high-dimensional crypto assets
        lw = LedoitWolf().fit(returns_df)
        cov_matrix = lw.covariance_ * 252
        return cov_matrix

    def optimize_weights(self, signals: np.ndarray, returns_df: pd.DataFrame) -> np.ndarray:
        """
        Solves for minimum variance weights given the expected signals.
        Objective: Maximize (w^T * signals) - (gamma/2 * w^T * Sigma * w)
        """
        cov = self.estimate_covariance(returns_df)
        num_assets = len(signals)

        # Regularization to prevent extreme weights
        gamma = 1.0

        # Closed form solution for Mean-Variance optimization:
        # w = (1/gamma) * Sigma^-1 * signals
        try:
            inv_cov = np.linalg.inv(cov + np.eye(num_assets) * 1e-6)
            weights = (1.0 / gamma) * np.dot(inv_cov, signals)
        except np.linalg.LinAlgError:
            # Fallback to equal weight scaled by signal
            weights = signals / (np.sum(np.abs(signals)) + 1e-6)

        # Normalize weights so sum(|w|) = 1
        total_abs = np.sum(np.abs(weights))
        if total_abs > 0:
            weights = weights / total_abs

        return weights

    def get_portfolio_vol(self, weights: np.ndarray, returns_df: pd.DataFrame) -> float:
        """Calculates annualized portfolio volatility."""
        cov = self.estimate_covariance(returns_df)
        port_var = np.dot(weights.T, np.dot(cov, weights))
        return np.sqrt(max(0, port_var))
