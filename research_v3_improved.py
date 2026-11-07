"""
Research: improve the existing v3 concept without throwing it away.

We keep the v3 stack:
    raw signal -> Square-Root Lasso -> probabilistic sizing
               -> Elastic Hysteresis -> Ledoit-Wolf vol target

and change what feeds it + a few risk details. Everything is causal:
weights formed at bar t earn the t -> t+1 return, costs are charged.

Run:  python research_v3_improved.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from research_signal2 import load_data, metrics, beta_alpha, COST, ANN
from neural_flow.alpha.model import LiquidityAlpha

warnings.filterwarnings('ignore')

V3_TICKERS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD',
              'ADA-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD']
TRAIN_START = pd.Timestamp('2020-11-07')
TRAIN_END = pd.Timestamp('2024-01-01')
TEST_START = pd.Timestamp('2024-01-01')
TEST_END = pd.Timestamp('2026-11-08')


# --------------------------------------------------------------------------- #
# v3 portfolio layer (kept, with optional fixes)
# --------------------------------------------------------------------------- #
class V3Optimizer:
    def __init__(self, lambda_base=0.005, tau_base=0.01, critical=3.0,
                 tau_mode='orig'):
        self.lambda_base = lambda_base
        self.tau_base = tau_base
        self.critical = critical
        self.tau_mode = tau_mode
        self.prev_weights = None

    def optimize(self, signals, confidences, market_vol=0.15):
        signals = np.nan_to_num(signals)
        confidences = np.nan_to_num(confidences)

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

        capped = []
        for w, c in zip(weights, confidences):
            cap = 0.05 + (np.clip(c, 0, 3) / 3.0) * (0.50 - 0.05)
            capped.append(np.clip(w, -cap, cap))
        weights = np.array(capped)
        total_exp = np.abs(weights).sum()
        if total_exp > 1.0:
            weights = weights / total_exp

        if self.prev_weights is not None and len(self.prev_weights) == len(weights):
            if self.tau_mode == 'orig':
                tau_adj = self.tau_base / (1.0 + market_vol * 10.0)
            elif self.tau_mode == 'vol':
                tau_adj = self.tau_base * (1.0 + market_vol)
            else:  # fixed
                tau_adj = self.tau_base

            new = np.copy(self.prev_weights)
            for i in range(len(weights)):
                if abs(signals[i]) > self.critical:
                    new[i] = weights[i]
                    continue
                diff = weights[i] - self.prev_weights[i]
                if diff < 0 and abs(diff) > 0.5 * tau_adj:
                    new[i] = weights[i]
                elif diff > 0 and abs(diff) > 1.5 * tau_adj:
                    new[i] = weights[i]
                else:
                    new[i] = self.prev_weights[i]
            weights = new

        self.prev_weights = weights
        return weights


class ProRiskManagerV3:
    def __init__(self, target_vol=0.15, base_leverage=4.0, cov_window=20, ann=252):
        self.target_vol = target_vol
        self.base_leverage = base_leverage
        self.cov_window = cov_window
        self.ann = ann

    def leverage(self, weights, returns):
        df = returns.tail(self.cov_window)
        if len(df) < 20:
            return 1.0
        try:
            cov = LedoitWolf().fit(df.values).covariance_ * self.ann
        except Exception:
            cov = df.cov().values * self.ann
        port_var = float(weights @ cov @ weights)
        port_vol = np.sqrt(max(0.0, port_var))
        if port_vol <= 1e-9:
            return 1.0  # no positions / no risk -> no leverage (was 4x bug)
        return min(self.target_vol / port_vol, self.base_leverage)


# --------------------------------------------------------------------------- #
# Signals
# --------------------------------------------------------------------------- #
def liq_signals(close, volume):
    engine = LiquidityAlpha()
    sig, conf = {}, {}
    for c in close.columns:
        r = engine.generate_signal(pd.DataFrame({'Close': close[c], 'Volume': volume[c]}))
        sig[c] = r['signal']
        conf[c] = r['confidence']
    return pd.DataFrame(sig), pd.DataFrame(conf)


def trend_signals(close, lookback=50, mode='dist'):
    ret = close.pct_change()
    vol = ret.rolling(20).std()
    sma = close.rolling(lookback, min_periods=lookback).mean()
    if mode == 'dist':
        score = (close / sma - 1.0) / (vol + 1e-9)
    elif mode == 'tstat':
        score = close.pct_change(lookback) / (vol * np.sqrt(lookback) + 1e-9)
    else:  # sign
        score = np.sign(close - sma)
    score = score.clip(-5, 5)
    return score, score.abs().clip(0, 3)


def combo_signals(close):
    parts = [trend_signals(close, lb)[0] for lb in (20, 50, 100)]
    score = sum(parts) / len(parts)
    return score, score.abs().clip(0, 3)


def mom_signals(close, lookback=60):
    ret = close.pct_change()
    vol = ret.rolling(20).std()
    score = (close.pct_change(lookback) / (vol * np.sqrt(lookback) + 1e-9)).clip(-5, 5)
    return score, score.abs().clip(0, 3)


def smooth(df, n):
    return df.rolling(n).mean() if n > 1 else df


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
DEFAULTS = dict(lambda_base=0.005, tau_base=0.01, critical=3.0, tau_mode='orig',
                target_vol=0.15, base_leverage=4.0, cov_window=20, risk_ann=252,
                long_only=False, regime='none', regime_off=0.0, max_asset=10.0,
                cost=COST)


def run_v3(close, fwd, sig_df, conf_df, start=None, end=None, **overrides):
    p = dict(DEFAULTS)
    p.update(overrides)
    tickers = close.columns.tolist()
    rets = close.pct_change()
    opt = V3Optimizer(p['lambda_base'], p['tau_base'], p['critical'], p['tau_mode'])
    risk = ProRiskManagerV3(p['target_vol'], p['base_leverage'],
                            p['cov_window'], p['risk_ann'])

    dates = close.index[:-1]
    if start is not None:
        dates = dates[dates >= start]
    if end is not None:
        dates = dates[dates < end]

    btc = close['BTC-USD']
    regime_sma = None
    if p['regime'].startswith('btc'):
        regime_len = int(p['regime'][3:])
        regime_sma = btc.rolling(regime_len, min_periods=1).mean()

    prev = np.zeros(len(tickers))
    rows = []
    for d in dates:
        sig = sig_df.loc[d].reindex(tickers).fillna(0.0).values
        conf = conf_df.loc[d].reindex(tickers).fillna(0.0).values
        if p['long_only']:
            sig = np.clip(sig, 0.0, None)

        mkt_vol = rets.loc[:d].tail(60).std().mean() * np.sqrt(ANN)
        w = opt.optimize(sig, conf, market_vol=mkt_vol)
        lev = risk.leverage(w, rets.loc[:d])
        final = np.clip(w * lev, -p['max_asset'], p['max_asset'])

        if regime_sma is not None:
            on = btc.loc[d] > regime_sma.loc[d]
            final = final * (1.0 if on else p['regime_off'])

        turnover = float(np.abs(final - prev).sum())
        gross = float(np.sum(final * fwd.loc[d].values))
        rows.append((gross - p['cost'] * turnover, gross, turnover, lev))
        prev = final

    return pd.DataFrame(rows, index=dates, columns=['net', 'gross', 'turn', 'lev'])


def slice_s(x, start, end):
    return x[(x.index >= start) & (x.index < end)]


def yearly(rets):
    return rets.groupby(rets.index.year).apply(lambda x: (1 + x).prod() - 1)


def report(name, df, fwd, start, end):
    r = slice_s(df['net'], start, end)
    m = metrics(r)
    b, a = beta_alpha(r, slice_s(fwd['BTC-USD'], start, end))
    turn = slice_s(df['turn'], start, end).mean()
    lev = slice_s(df['lev'], start, end).mean()
    print(f"{name:<38}{m['total']*100:>8.1f}{m['ann']*100:>8.1f}{m['vol']*100:>7.1f}"
          f"{m['sharpe']:>7.2f}{m['mdd']*100:>7.1f}{turn*100:>7.1f}{lev:>6.2f}{a*100:>8.1f}")
    return m


# --------------------------------------------------------------------------- #
def main():
    close, high, low, volume = load_data('6y')
    tickers = [t for t in V3_TICKERS if t in close.columns]
    close, volume = close[tickers], volume[tickers]
    fwd = close.shift(-1) / close - 1.0
    print(f"v3 universe: {len(tickers)} assets | {close.index[0].date()} -> {close.index[-1].date()}\n")

    liq_sig, liq_conf = liq_signals(close, volume)
    tr50_sig, tr50_conf = trend_signals(close, 50)
    tr50s_sig, tr50s_conf = smooth(tr50_sig, 3), smooth(tr50_conf, 3)
    combo_sig, combo_conf = combo_signals(close)
    mom_sig, mom_conf = mom_signals(close, 60)

    # ---- 1. reproduce the current v3 on the last 2 years ----
    print('=' * 112)
    print('BASELINE CHECK: current v3 (LiquidityAlpha + original layer), last ~2 years')
    print('=' * 112)
    base_2y = run_v3(close, fwd, liq_sig, liq_conf,
                     start=pd.Timestamp('2024-11-08'), end=TEST_END)
    report('current v3 (liq, orig layer)', base_2y, fwd,
           pd.Timestamp('2024-11-08'), TEST_END)

    # ---- 2. variants on the full sample ----
    variants = {
        'v3 baseline (liq, orig)': (liq_sig, liq_conf, {}),
        'v3 liq + fixed risk layer': (liq_sig, liq_conf,
                                      dict(tau_mode='vol', base_leverage=2.0,
                                           cov_window=60, risk_ann=365, max_asset=1.0)),
        'v3 trend50 + orig layer': (tr50_sig, tr50_conf, {}),
        'v3 trend50 + fixed risk': (tr50_sig, tr50_conf,
                                    dict(tau_mode='vol', base_leverage=2.0,
                                         cov_window=60, risk_ann=365, max_asset=1.0)),
        'v3 trend50 fixed, long-only': (tr50_sig, tr50_conf,
                                        dict(tau_mode='vol', base_leverage=2.0,
                                             cov_window=60, risk_ann=365, max_asset=1.0,
                                             long_only=True)),
        'v3 trend50 fixed + btc200': (tr50_sig, tr50_conf,
                                      dict(tau_mode='vol', base_leverage=2.0,
                                           cov_window=60, risk_ann=365, max_asset=1.0,
                                           regime='btc200', regime_off=0.0)),
        'v3 trend50 fixed + btc200 half': (tr50_sig, tr50_conf,
                                           dict(tau_mode='vol', base_leverage=2.0,
                                                cov_window=60, risk_ann=365, max_asset=1.0,
                                                regime='btc200', regime_off=0.5)),
        'v3 combo(20/50/100) + fixed': (combo_sig, combo_conf,
                                        dict(tau_mode='vol', base_leverage=2.0,
                                             cov_window=60, risk_ann=365, max_asset=1.0)),
        'v3 mom60 + fixed': (mom_sig, mom_conf,
                             dict(tau_mode='vol', base_leverage=2.0,
                                  cov_window=60, risk_ann=365, max_asset=1.0)),
        'v3 trend50 smooth3 + fixed': (tr50s_sig, tr50s_conf,
                                       dict(tau_mode='vol', base_leverage=2.0,
                                            cov_window=60, risk_ann=365, max_asset=1.0)),
        'v3 trend50 fixed, tau=0.03': (tr50_sig, tr50_conf,
                                       dict(tau_mode='vol', base_leverage=2.0, tau_base=0.03,
                                            cov_window=60, risk_ann=365, max_asset=1.0)),
        'v3 trend50 fixed, 3x': (tr50_sig, tr50_conf,
                                 dict(tau_mode='vol', base_leverage=3.0,
                                      cov_window=60, risk_ann=365, max_asset=1.5)),
    }

    results = {}
    for name, (sig, conf, ov) in variants.items():
        results[name] = run_v3(close, fwd, sig, conf, **ov)

    hdr = (f"{'variant':<38}{'total%':>8}{'CAGR%':>8}{'vol%':>7}{'Sharpe':>7}"
           f"{'MDD%':>7}{'turn%':>7}{'lev':>6}{'alpha%':>8}")
    for label, s, e in [('TRAIN 2020-11..2023-12', TRAIN_START, TRAIN_END),
                        ('TEST  2024-01..2026-11', TEST_START, TEST_END),
                        ('FULL  2020-11..2026-11', TRAIN_START, TEST_END)]:
        print('\n' + '=' * 112)
        print(label)
        print('=' * 112)
        print(hdr)
        print('-' * 112)
        for name, df in results.items():
            report(name, df, fwd, s, e)
        print('-' * 112)
        m = metrics(slice_s(fwd['BTC-USD'], s, e))
        print(f"{'BTC buy & hold':<38}{m['total']*100:>8.1f}{m['ann']*100:>8.1f}{m['vol']*100:>7.1f}"
              f"{m['sharpe']:>7.2f}{m['mdd']*100:>7.1f}{'-':>7}{'-':>6}{'-':>8}")

    # ---- 3. yearly for the leading candidates ----
    print('\n' + '=' * 112)
    print('YEAR-BY-YEAR (net)')
    print('=' * 112)
    chosen = ['v3 trend50 + fixed risk', 'v3 trend50 fixed, long-only',
              'v3 trend50 fixed + btc200', 'v3 combo(20/50/100) + fixed',
              'v3 baseline (liq, orig)']
    df = pd.DataFrame({n: yearly(slice_s(results[n]['net'], TRAIN_START, TEST_END))
                       for n in chosen}) * 100
    df['BTC B&H'] = yearly(slice_s(fwd['BTC-USD'], TRAIN_START, TEST_END)) * 100
    print(df.round(1).to_string())

    # ---- 4. gross vs net for the leader ----
    leader = 'v3 trend50 + fixed risk'
    print('\nLeader gross vs net (full):')
    for col in ('gross', 'net'):
        m = metrics(slice_s(results[leader][col], TRAIN_START, TEST_END))
        print(f"  {col:<6} CAGR={m['ann']*100:6.1f}%  Sharpe={m['sharpe']:.2f}  "
              f"MDD={m['mdd']*100:5.1f}%  vol={m['vol']*100:5.1f}%")


if __name__ == '__main__':
    main()
