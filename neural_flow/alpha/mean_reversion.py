import numpy as np
import pandas as pd

class MeanReversionAlpha:
    """
    Mean Reversion Alpha: Identifies assets that have deviated significantly
    from the basket average (cross-sectional mean reversion).
    """
    def __init__(self, window=20, z_threshold=2.0):
        self.window = window
        self.z_threshold = z_threshold

    def generate_signals(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the z-score of each asset's return relative to the basket average.
        Returns a DataFrame with signals and confidence.
        """
        # 1. Calculate Basket Average Return (The Market Proxy)
        basket_avg = returns_df.mean(axis=1)

        # 2. Calculate Excess Returns (Relative Performance)
        excess_returns = returns_df.sub(basket_avg, axis=0)

        # 3. Calculate Z-Score of Excess Returns
        # We use rolling window for mean and std to account for regime changes
        rolling_mean = excess_returns.rolling(window=self.window).mean()
        rolling_std = excess_returns.rolling(window=self.window).std()

        z_scores = (excess_returns - rolling_mean) / (rolling_std + 1e-6)

        # 4. Generate Signal: Opposite of the Z-Score (Mean Reversion)
        # If Z-score is high (overperforming), signal is negative (Sell)
        # If Z-score is low (underperforming), signal is positive (Buy)
        signals = -z_scores

        # 5. Calculate Confidence
        # Confidence is based on the magnitude of the Z-score,
        # but capped to prevent extreme bets on outliers.
        confidence = np.abs(z_scores).clip(0, 3.0)

        # 6. Apply Z-Threshold Guard
        # Only generate signal if the absolute Z-score exceeds the threshold
        final_signals = np.where(np.abs(z_scores) > self.z_threshold, signals, 0.0)

        # Convert numpy arrays back to DataFrame
        signal_df = pd.DataFrame(final_signals, index=returns_df.index, columns=returns_df.columns)
        conf_df = pd.DataFrame(confidence, index=returns_df.index, columns=returns_df.columns)

        return signal_df, conf_df
