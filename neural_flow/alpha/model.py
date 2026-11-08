import numpy as np
import pandas as pd
from neural_flow.alpha.mean_reversion import MeanReversionAlpha

class AlphaAggregator:
    """
    Combines multiple alpha signals into a single composite signal.
    Balances between Trend/Liquidity and Mean Reversion.
    """
    def __init__(self, weights=None):
        # Default weights: Equal balance between Liquidity and Mean Reversion
        self.weights = weights if weights else {'liquidity': 0.5, 'mean_reversion': 0.5}

    def aggregate(self, liq_signals, liq_conf, mr_signals, mr_conf):
        """
        Aggregates signals and confidences.
        - liq_signals/conf: Results from LiquidityAlpha
        - mr_signals/conf: Results from MeanReversionAlpha
        """
        # Combined Signal = weighted sum of signals
        combined_signal = (liq_signals * self.weights['liquidity']) + (mr_signals * self.weights['mean_reversion'])

        # Combined Confidence = weighted average of confidences
        combined_conf = (liq_conf * self.weights['liquidity']) + (mr_conf * self.weights['mean_reversion'])

        return combined_signal, combined_conf

class LiquidityAlpha:

    """
    Neural-Flow Alpha: Focuses on Order Flow and Liquidity.
    Returns both the signal and a confidence score (strength).
    """
    def __init__(self, window=20):
        self.window = window

    def generate_signal(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data['Close']
        volume = data['Volume']

        # 1. Multi-Timeframe Trend Filters
        # Require a full window.  A partial SMA200 is not a 200-day trend and
        # must not generate a position during the warm-up period.
        sma200 = close.rolling(window=200, min_periods=200).mean()
        macro_trend = np.where(close > sma200, 1, -1)
        macro_trend = np.where(sma200.notna(), macro_trend, 0)

        # Medium Trend (SMA 20) - The "Momentum Guard"
        sma20 = close.rolling(window=20, min_periods=20).mean()
        medium_trend = np.where(close > sma20, 1, -1)
        medium_trend = np.where(sma20.notna(), medium_trend, 0)

        # Final Trend: Only trade when Macro and Medium agree.  Zero means
        # "warm-up / indecision" and is treated as cash, never a short.
        trend = np.where((macro_trend == medium_trend) & (macro_trend != 0),
                         macro_trend, 0)

        # 2. Advanced Alpha Signals
        # Toxicity Proxy (Volatility * Volume)
        volatility = close.pct_change().rolling(self.window).std()
        vol_vol_prod = volatility * volume
        z_tox = (vol_vol_prod - vol_vol_prod.rolling(self.window).mean()) / (vol_vol_prod.rolling(self.window).std() + 1e-6)

        # Imbalance Proxy (Price Change * Volume)
        obi_proxy = (close.pct_change() * volume).rolling(self.window).sum()
        z_obi = (obi_proxy - obi_proxy.rolling(self.window).mean()) / (obi_proxy.rolling(self.window).std() + 1e-6)

        # RSI-based Overbought/Oversold Filter (Prevents buying the top)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-6)
        rsi = 100 - (100 / (1 + rs))

        # RSI Guard: No Longs > 70, No Shorts < 30
        rsi_guard = np.where(rsi > 70, -1, np.where(rsi < 30, 1, 1))

        # Combine signals
        raw_signal = z_tox * z_obi

        # Final Signal: Must match Trend AND be approved by RSI Guard
        # We use np.where with boolean masks to avoid "truth value of a Series is ambiguous" errors
        long_condition = (np.sign(raw_signal) == 1) & (trend == 1) & (rsi < 70)
        short_condition = (np.sign(raw_signal) == -1) & (trend == -1) & (rsi > 30)

        signal = np.where(long_condition, raw_signal, np.where(short_condition, raw_signal, 0.0))

        # 3. Dynamic Confidence Score
        # Magnitude of Z-scores, penalized by extreme volatility and trend disagreement
        vol_ratio = volatility / (volatility.rolling(self.window * 5).mean() + 1e-6)
        vol_penalty = np.where(vol_ratio > 2.5, 0.3, np.where(vol_ratio > 1.5, 0.7, 1.0))

        # Confidence is higher when the trend is strong
        trend_strength = np.abs(close - sma200) / sma200
        conf_boost = np.clip(1 + trend_strength, 1.0, 1.5)

        confidence = ((np.abs(z_tox) + np.abs(z_obi)) / 2) * vol_penalty * conf_boost

        return pd.DataFrame({'signal': signal, 'confidence': confidence}, index=data.index)
