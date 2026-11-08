"""
Core of the improved Adaptive v3 strategy.

The v3 concept is kept intact:
    raw signal -> Square-Root Lasso -> probabilistic sizing
               -> Elastic Hysteresis -> Ledoit-Wolf vol target

What changed to give it an edge:
  * the raw signal is a causal 50-day trend score, not LiquidityAlpha;
  * hysteresis actually bites (tau grows with volatility instead of shrinking);
  * 60-day Ledoit-Wolf covariance and 365-day crypto annualisation;
  * 2x leverage cap and 100% per-asset cap;
  * BTC 150-day regime filter: go to cash when BTC is below it.

Everything is causal: inputs use bars up to and including t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def trend_score(close, lookback=50, vol_window=20, cap=5.0):
    """Causal trend score: distance from the SMA in daily-volatility units.

    Returns (score, confidence), both DataFrames indexed like `close`.
    """
    if isinstance(close, pd.Series):
        close = close.to_frame()

    ret = close.pct_change()
    vol = ret.rolling(vol_window).std()
    sma = close.rolling(lookback, min_periods=lookback).mean()
    score = ((close / sma - 1.0) / (vol + 1e-9)).clip(-cap, cap)
    conf = score.abs().clip(0.0, 3.0)
    return score, conf


def btc_regime_series(close, lookback=150):
    """1.0 when BTC is above its lookback SMA, else 0.0.

    Falls back to the first available column if BTC-USD is missing from a
    download, so a single failed ticker cannot crash the live runner.
    """
    if isinstance(close, pd.Series):
        btc = close
    elif 'BTC-USD' in close.columns:
        btc = close['BTC-USD']
    else:
        btc = close.iloc[:, 0]
    # A full window is required: a partial SMA in the first `lookback` days is
    # not a 150-day regime signal and can produce misleading early exposure.
    sma = btc.rolling(lookback, min_periods=lookback).mean()
    return (btc > sma).astype(float)


class AdaptiveWeightedV3:
    """Square-Root Lasso + probabilistic sizing + Elastic Hysteresis."""

    def __init__(self, lambda_base=0.05, tau_base=0.02,
                 critical_threshold=3.0, prev_weights=None):
        self.lambda_base = lambda_base
        self.tau_base = tau_base
        self.critical_threshold = critical_threshold
        if prev_weights is None:
            self.prev_weights = None
        else:
            self.prev_weights = np.asarray(prev_weights, dtype=float)

    def optimize(self, signals, confidences, market_vol=0.15):
        signals = np.nan_to_num(np.asarray(signals, dtype=float))
        confidences = np.nan_to_num(np.asarray(confidences, dtype=float))

        # 1. Square-root lasso threshold (adaptive to signal/noise).
        snr = np.mean(np.abs(signals)) / (np.std(signals) + 1e-6)
        lambda_adj = self.lambda_base / (1.0 + snr)

        abs_sigs = np.abs(signals)
        if abs_sigs.sum() == 0:
            self.prev_weights = np.zeros(len(signals))
            return self.prev_weights

        weights = np.sign(signals) * (abs_sigs / abs_sigs.sum())
        weights = np.where(np.abs(weights) > lambda_adj, weights, 0.0)
        if np.abs(weights).sum() == 0:
            self.prev_weights = np.zeros(len(signals))
            return self.prev_weights
        weights = weights / np.abs(weights).sum()

        # 2. Probabilistic sizing: stronger confidence -> larger cap.
        capped = []
        for w, conf in zip(weights, confidences):
            cap = 0.05 + (np.clip(conf, 0.0, 3.0) / 3.0) * (0.50 - 0.05)
            capped.append(np.clip(w, -cap, cap))
        weights = np.array(capped)

        total_exp = np.abs(weights).sum()
        if total_exp > 1.0:
            weights = weights / total_exp

        # 3. Elastic hysteresis. tau grows with market volatility so the band
        #    is wider when turnover is most expensive.
        if self.prev_weights is not None and len(self.prev_weights) == len(weights):
            tau_adj = self.tau_base * (1.0 + market_vol)
            new_weights = np.copy(self.prev_weights)
            for i in range(len(weights)):
                if abs(signals[i]) > self.critical_threshold:
                    new_weights[i] = weights[i]
                    continue
                diff = weights[i] - self.prev_weights[i]
                if diff < 0 and abs(diff) > 0.5 * tau_adj:
                    new_weights[i] = weights[i]
                elif diff > 0 and abs(diff) > 1.5 * tau_adj:
                    new_weights[i] = weights[i]
                else:
                    new_weights[i] = self.prev_weights[i]
            weights = new_weights

        self.prev_weights = weights
        return weights


class ProRiskManagerV3:
    """Ledoit-Wolf covariance + portfolio volatility target."""

    def __init__(self, target_vol=0.15, base_leverage=2.0,
                 cov_window=60, ann=365):
        self.target_vol = target_vol
        self.base_leverage = base_leverage
        self.cov_window = cov_window
        self.ann = ann

    def get_portfolio_leverage(self, weights, returns_df):
        if len(returns_df) < 20:
            return 1.0

        recent = returns_df.tail(self.cov_window)
        try:
            lw = LedoitWolf().fit(recent.values)
            cov_matrix = lw.covariance_ * self.ann
        except Exception:
            cov_matrix = recent.cov().values * self.ann

        port_var = float(np.dot(weights.T, np.dot(cov_matrix, weights)))
        port_vol = np.sqrt(max(0.0, port_var))
        if port_vol <= 1e-9:
            return 1.0  # no positions -> no leverage

        return min(self.target_vol / port_vol, self.base_leverage)
