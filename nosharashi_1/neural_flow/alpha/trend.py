"""
Long-only time-series momentum with inverse-volatility sizing and a
portfolio volatility target.

This is the redesigned signal. It is deliberately simple and causal:

    signal_t = 1 if close_t > SMA_lookback(close)_t else 0
    raw_t    = signal_t / realised_vol_t
    weights  = raw normalised to gross 1, then scaled by
               min(target_vol / ex-ante portfolio vol, max_leverage)

Every input uses only data up to and including bar t, so weights_t can be
applied to the return from t to t+1 with no look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class TrendVolSignal:
    def __init__(self, lookback=50, vol_window=20, target_vol=0.15,
                 max_leverage=2.0, max_weight=0.75, ann=365):
        self.lookback = lookback
        self.vol_window = vol_window
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.max_weight = max_weight
        self.ann = ann

    def generate_weights(self, close: pd.DataFrame) -> pd.DataFrame:
        if isinstance(close, pd.Series):
            close = close.to_frame()

        ret = close.pct_change()
        vol = ret.rolling(self.vol_window).std() * np.sqrt(self.ann)
        sma = close.rolling(self.lookback, min_periods=self.lookback).mean()

        signal = (close > sma).astype(float)
        raw = signal / (vol + 1e-9)
        gross = raw.abs().sum(axis=1)
        weights = raw.div(gross.replace(0.0, np.nan), axis=0).fillna(0.0)

        # Portfolio-level volatility target (diagonal covariance approximation).
        port_vol = np.sqrt((weights.pow(2) * vol.pow(2)).sum(axis=1))
        scale = (self.target_vol / (port_vol + 1e-9)).clip(upper=self.max_leverage)
        weights = weights.mul(scale, axis=0).clip(upper=self.max_weight)
        return weights.fillna(0.0)

    def latest_weights(self, close: pd.DataFrame) -> pd.Series:
        return self.generate_weights(close).iloc[-1].fillna(0.0)
