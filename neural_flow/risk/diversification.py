import numpy as np
import pandas as pd

class CorrelationFilter:
    """
    Implements a Correlation Penalty to ensure portfolio diversification.
    Penalizes assets that are highly correlated with the rest of the portfolio.
    """
    def __init__(self, lambda_penalty=0.5, window=20):
        self.lambda_penalty = lambda_penalty
        self.window = window

    def apply_filter(self, weights, returns_df: pd.DataFrame):
        """
        Adjusts weights based on asset correlation.

        Args:
            weights: np.array of initial weights.
            returns_df: pd.DataFrame of asset returns.

        Returns:
            Adjusted weights np.array.
        """
        if len(returns_df) < self.window:
            return weights

        # 1. Calculate Rolling Correlation Matrix
        # We use the most recent window of returns
        recent_returns = returns_df.tail(self.window)
        corr_matrix = recent_returns.corr()

        # Fill NaNs with 0 (no correlation)
        corr_matrix = corr_matrix.fillna(0)

        # 2. Calculate Average Correlation for each asset
        # We exclude the diagonal (asset's correlation with itself is always 1)
        num_assets = len(weights)
        adjusted_weights = np.copy(weights)

        for i in range(num_assets):
            # Get correlations of asset i with all other assets
            corrs = corr_matrix.iloc[i, :].values

            # Sum correlations excluding the diagonal
            sum_corr = np.sum(corrs) - 1.0
            avg_corr = sum_corr / (num_assets - 1) if num_assets > 1 else 0

            # 3. Apply Penalty: W* = W * (1 - lambda * avg_corr)
            # Note: avg_corr can be negative (good for diversification)
            # We clip the penalty to ensure we don't invert the position
            penalty = 1.0 - (self.lambda_penalty * avg_corr)
            penalty = np.clip(penalty, 0.1, 2.0) # Don't zero out completely, don't boost too much

            adjusted_weights[i] *= penalty

        return adjusted_weights
