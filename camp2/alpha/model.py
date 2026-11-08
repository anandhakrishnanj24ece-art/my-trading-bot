import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import BayesianRidge

class ConditionalFactorModel:
    """
    Implementation of the Conditional Factor Model with Bayesian Shrinkage.
    mu = alpha + beta*F + gamma*R
    """
    def __init__(self, prior_sigma=1.0):
        self.model = BayesianRidge()
        self.prior_sigma = prior_sigma

    def fit(self, y: pd.Series, X: pd.DataFrame):
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(X), index=X.index)

class CrossSectionalAlpha:
    """
    Technique 1: Cross-Sectional Mean Reversion.
    """
    def __init__(self, window=20):
        self.window = window

    def generate_signal(self, returns: pd.DataFrame) -> pd.DataFrame:
        basket_avg = returns.mean(axis=1)
        relative_ret = returns.sub(basket_avg, axis=0)
        rolling_mean = relative_ret.rolling(self.window).mean()
        rolling_std = relative_ret.rolling(self.window).std()
        z_score = (relative_ret - rolling_mean) / (rolling_std + 1e-6)
        return -z_score

class MomentumAlpha:
    """
    Technique 2: Time-Series Momentum (Trend Following).
    Captures the 'momentum' factor using a volatility-adjusted return.
    """
    def __init__(self, window=20):
        self.window = window

    def generate_signal(self, returns: pd.DataFrame) -> pd.DataFrame:
        # Calculate risk-adjusted momentum: Return / Volatility
        rolling_ret = returns.rolling(self.window).sum()
        rolling_vol = returns.rolling(self.window).std()
        return rolling_ret / (rolling_vol + 1e-6)

class QualityAlpha:
    """
    Technique 3: Low Volatility Anomaly.
    Assets with lower volatility relative to the basket often outperform (Risk-Adjusted).
    """
    def __init__(self, window=20):
        self.window = window

    def generate_signal(self, returns: pd.DataFrame) -> pd.DataFrame:
        vol = returns.rolling(self.window).std()
        # Rank inverse volatility
        return 1.0 / (vol + 1e-6)

class VolatilityScaler:
    """
    Technique 4: Risk Parity Scaling.
    """
    def scale(self, signal: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        vol = returns.rolling(20).std()
        return signal.div(vol, axis=0)

class ConvexityAlpha:
    """
    Technique 5: Convexity/Skewness Alpha.
    Bets on assets with positive skew (fat right tail), which is common in crypto breakouts.
    """
    def __init__(self, window=60):
        self.window = window

    def generate_signal(self, returns: pd.DataFrame) -> pd.DataFrame:
        def calc_skew(x):
            return stats.skew(x)

        skew = returns.rolling(self.window).apply(calc_skew)
        return skew
