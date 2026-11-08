"""
Signal research v3: robustness across regimes.

Tests the promising directions on BOTH the train period (2020-11 -> 2023-12)
and the untouched test period (2024-01 -> 2026-11), plus year-by-year.

Candidates:
  * Cross-sectional low-vol / low-range / high-liquidity factors.
  * Time-series momentum (trend) across the universe, long-only and long-short.
  * BTC / ETH trend with vol targeting.
  * A blend of trend + low-vol factor.

Run:  python research_signal3.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from research_signal2 import (load_data, build_features, cs_z, ic_stats,
                              run_portfolio, run_weighted, metrics, beta_alpha,
                              ANN, COST, TEST_START)

warnings.filterwarnings('ignore')

TRAIN_START = pd.Timestamp('2020-11-07')
TRAIN_END = TEST_START
TEST_END = pd.Timestamp('2026-11-08')


def run_weights(weights, fwd, start, end, rebalance=1, cost=COST):
    dates = weights.index.intersection(fwd.dropna(how='all').index)
    dates = dates[(dates >= start) & (dates < end)]
    w = weights.loc[dates].fillna(0.0)
    if rebalance > 1:
        keep = w.iloc[::rebalance]
        w = keep.reindex(dates).ffill().fillna(0.0)
    fr = fwd.loc[dates]
    turn = w.diff().abs().sum(axis=1)
    turn.iloc[0] = w.iloc[0].abs().sum()
    gross = (w * fr).sum(axis=1)
    net = gross - turn * cost
    return gross, net, float(turn.mean())


def apply_vol_target(rets, target=0.15, span=20, cap=2.0):
    rv = rets.rolling(span).std() * np.sqrt(ANN)
    scale = (target / (rv + 1e-9)).clip(0, cap).shift(1).fillna(1.0)
    return rets * scale


def tsmom_weights(close, lookback=50, long_short=True):
    ret = close.pct_change()
    vol = ret.rolling(20).std() * np.sqrt(ANN)
    sig = np.sign(close - close.rolling(lookback).mean())
    if not long_short:
        sig = sig.clip(lower=0)
    raw = sig / (vol + 1e-9)
    gross = raw.abs().sum(axis=1)
    return raw.div(gross.replace(0, np.nan), axis=0).fillna(0.0)


def slice_s(x, start, end):
    return x[(x.index >= start) & (x.index < end)]


def report(name, gross, net, bench, start, end, extra=''):
    g, n = slice_s(gross, start, end), slice_s(net, start, end)
    m = metrics(n)
    b, a = beta_alpha(n, slice_s(bench, start, end))
    print(f"{name:<30}{m['total']*100:>8.1f}{m['ann']*100:>8.1f}{m['vol']*100:>7.1f}"
          f"{m['sharpe']:>8.2f}{m['mdd']*100:>7.1f}{m['hit']*100:>7.1f}"
          f"{b:>6.2f}{a*100:>8.1f}  {extra}")
    return n


def yearly(rets):
    return rets.groupby(rets.index.year).apply(lambda x: (1 + x).prod() - 1)


def main():
    print('Loading data...')
    close, high, low, volume = load_data('6y')
    fwd = close.shift(-1) / close - 1.0
    bench = fwd.mean(axis=1)
    feats = build_features(close, high, low, volume)

    # composites selected on train only (same as v2)
    rows = []
    for name, f in feats.items():
        it, tt, _ = ic_stats(f, fwd, TRAIN_START, TRAIN_END)
        io, to, _ = ic_stats(f, fwd, TEST_START, TEST_END)
        rows.append((name, it, tt, io, to))
    ic = pd.DataFrame(rows, columns=['feature', 'train_IC', 'train_t', 'test_IC', 'test_t'])
    sel = ic[(ic['train_t'].abs() >= 3.0) & ic['train_IC'].notna()]
    comp = sum(np.sign(r.train_IC) * cs_z(feats[r.feature]) for r in sel.itertuples()
               if r.feature != 'rev_5') / max(1, len(sel))
    three = (-cs_z(feats['vol_20']) - cs_z(feats['range_20']) + cs_z(feats['dollar_vol_20'])) / 3

    # trend weights
    ts50 = tsmom_weights(close, 50, True)
    ts100 = tsmom_weights(close, 100, True)
    ts200 = tsmom_weights(close, 200, True)
    ts50_lo = tsmom_weights(close, 50, False)
    ts200_lo = tsmom_weights(close, 200, False)

    btc = close['BTC-USD']
    btc_vol = btc.pct_change().rolling(20).std() * np.sqrt(ANN)
    btc_lev = (0.15 / (btc_vol + 1e-9)).clip(0, 2)
    btc50 = ((btc > btc.rolling(50).mean()).astype(float) * btc_lev).to_frame('BTC-USD')
    btc200 = ((btc > btc.rolling(200).mean()).astype(float) * btc_lev).to_frame('BTC-USD')

    # build return streams once
    streams = {}
    for name, sig, reb in [
        ('low-vol LS', -feats['vol_20'], 7),
        ('low-range LS', -feats['range_20'], 7),
        ('high-liquidity LS', feats['dollar_vol_20'], 7),
        ('3-factor weighted', three, 7),
        ('composite weighted', comp, 7),
        ('trend50 LS', ts50, 7),
        ('trend100 LS', ts100, 7),
        ('trend200 LS', ts200, 7),
        ('trend50 long-only', ts50_lo, 7),
        ('trend200 long-only', ts200_lo, 7),
    ]:
        if 'weighted' in name or name.startswith('low-') or name.startswith('high-'):
            # factor-weighted for the first group; trend uses run_weights
            if name.startswith(('low-', 'high-', '3-factor', 'composite')):
                g, n, t = run_weighted(sig, fwd, TRAIN_START, TEST_END, reb)
            else:
                g, n, t = run_weights(sig, fwd, TRAIN_START, TEST_END, reb)
        else:
            g, n, t = run_weights(sig, fwd, TRAIN_START, TEST_END, reb)
        streams[name] = (g, n, t)

    for name, w in [('BTC trend50', btc50), ('BTC trend200', btc200)]:
        g, n, t = run_weights(w, fwd, TRAIN_START, TEST_END, 1)
        streams[name] = (g, n, t)

    # vol-target the trend streams
    for name in ['trend50 LS', 'trend100 LS', 'trend200 LS', 'trend50 long-only',
                 'trend200 long-only', 'BTC trend50', 'BTC trend200']:
        g, n, t = streams[name]
        streams[name] = (g, apply_vol_target(n), t)

    hdr = (f"{'strategy':<30}{'total%':>8}{'CAGR%':>8}{'vol%':>7}{'Sharpe':>8}"
           f"{'MDD%':>7}{'hit%':>7}{'beta':>6}{'alpha%':>8}")

    for label, s, e in [('TRAIN 2020-11 -> 2023-12', TRAIN_START, TRAIN_END),
                        ('TEST 2024-01 -> 2026-11', TEST_START, TEST_END)]:
        print('\n' + '=' * 100)
        print(label + '  (net of 6.4 bps turnover)')
        print('=' * 100)
        print(hdr)
        print('-' * 100)
        for name in ['low-vol LS', 'low-range LS', 'high-liquidity LS', '3-factor weighted',
                     'composite weighted', 'trend50 LS', 'trend100 LS', 'trend200 LS',
                     'trend50 long-only', 'trend200 long-only', 'BTC trend50', 'BTC trend200']:
            g, n, t = streams[name]
            report(name, g, n, bench, s, e, extra=f't={t*100:.0f}%')
        m = metrics(slice_s(bench, s, e))
        print(f"{'equal-weight universe':<30}{m['total']*100:>8.1f}{m['ann']*100:>8.1f}"
              f"{m['vol']*100:>7.1f}{m['sharpe']:>8.2f}{m['mdd']*100:>7.1f}{m['hit']*100:>7.1f}"
              f"{'-':>6}{'-':>8}")

    # ---- year by year for the strongest candidates ----
    print('\n' + '=' * 100)
    print('YEAR-BY-YEAR NET RETURNS')
    print('=' * 100)
    chosen = ['trend50 LS', 'trend200 LS', 'BTC trend50', 'low-vol LS',
              '3-factor weighted', 'composite weighted']
    yn = {name: slice_s(streams[name][1], TRAIN_START, TEST_END) for name in chosen}
    yn['equal-weight'] = slice_s(bench, TRAIN_START, TEST_END)
    df = pd.DataFrame({k: yearly(v) for k, v in yn.items()}) * 100
    print(df.round(1).to_string())

    # ---- blend: trend50 LS + low-vol LS, equal risk ----
    print('\nBlend (50% trend50 LS + 50% low-vol LS):')
    a = yn['trend50 LS']
    b = yn['low-vol LS']
    blend = 0.5 * a + 0.5 * b
    for label, s, e in [('train', TRAIN_START, TRAIN_END), ('test', TEST_START, TEST_END)]:
        m = metrics(slice_s(blend, s, e))
        print(f"  {label:<6} total={m['total']*100:6.1f}%  CAGR={m['ann']*100:5.1f}%  "
              f"vol={m['vol']*100:5.1f}%  Sharpe={m['sharpe']:.2f}  MDD={m['mdd']*100:5.1f}%")


if __name__ == '__main__':
    main()
