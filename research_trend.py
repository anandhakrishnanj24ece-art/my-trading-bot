"""
Trend research: stress-test BTC / ETH / multi-asset time-series momentum.

Metrics on train (2020-11 -> 2023-12) and test (2024-01 -> 2026-11),
plus yearly returns. No look-ahead: weights at t earn t -> t+1.

Run:  python research_trend.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from research_signal2 import load_data, metrics, beta_alpha, ANN, COST
from research_signal3 import run_weights, slice_s, yearly

warnings.filterwarnings('ignore')

TRAIN_START = pd.Timestamp('2020-11-07')
TRAIN_END = pd.Timestamp('2024-01-01')
TEST_START = pd.Timestamp('2024-01-01')
TEST_END = pd.Timestamp('2026-11-08')


def single_trend(price, lookback, long_short=False, target=0.15, max_lev=2.0):
    ret = price.pct_change()
    vol = ret.rolling(20).std() * np.sqrt(ANN)
    if long_short:
        sig = np.sign(price - price.rolling(lookback).mean())
    else:
        sig = (price > price.rolling(lookback).mean()).astype(float)
    lev = (target / (vol + 1e-9)).clip(0, max_lev)
    return (sig * lev).fillna(0.0).to_frame(price.name)


def golden_cross(price, fast=50, slow=200, target=0.15, max_lev=2.0):
    ret = price.pct_change()
    vol = ret.rolling(20).std() * np.sqrt(ANN)
    sig = (price.rolling(fast).mean() > price.rolling(slow).mean()).astype(float)
    lev = (target / (vol + 1e-9)).clip(0, max_lev)
    return (sig * lev).fillna(0.0).to_frame(price.name)


def multi_trend(close, assets, lookback, target=0.15, max_lev=2.0):
    ret = close[assets].pct_change()
    vol = ret.rolling(20).std() * np.sqrt(ANN)
    sig = (close[assets] > close[assets].rolling(lookback).mean()).astype(float)
    raw = sig / (vol + 1e-9)
    gross = raw.abs().sum(axis=1)
    w = raw.div(gross.replace(0, np.nan), axis=0).fillna(0.0)
    # ex-ante portfolio vol (diagonal approximation), causal
    port_vol = np.sqrt((w ** 2 * vol ** 2).sum(axis=1))
    scale = (target / (port_vol + 1e-9)).clip(0, max_lev)
    return w.mul(scale, axis=0)


def main():
    close, high, low, volume = load_data('6y')
    fwd = close.shift(-1) / close - 1.0
    bench = fwd.mean(axis=1)

    variants = {}
    for lb in (20, 50, 100, 150, 200, 250):
        variants[f'BTC trend{lb} long'] = single_trend(close['BTC-USD'], lb)
    for lb in (50, 100, 200):
        variants[f'BTC trend{lb} LS'] = single_trend(close['BTC-USD'], lb, long_short=True)
    variants['BTC 50/200 cross'] = golden_cross(close['BTC-USD'], 50, 200)
    variants['BTC 20/100 cross'] = golden_cross(close['BTC-USD'], 20, 100)
    for lb in (50, 100, 200):
        variants[f'ETH trend{lb} long'] = single_trend(close['ETH-USD'], lb)
    variants['BTC+ETH trend50'] = multi_trend(close, ['BTC-USD', 'ETH-USD'], 50)
    variants['BTC+ETH trend200'] = multi_trend(close, ['BTC-USD', 'ETH-USD'], 200)
    variants['all-asset trend50'] = multi_trend(close, list(close.columns), 50)
    variants['all-asset trend200'] = multi_trend(close, list(close.columns), 200)

    # evaluate
    rows = []
    for name, w in variants.items():
        for reb, rname in [(1, '1d'), (7, '7d')]:
            g, n, t = run_weights(w, fwd, TRAIN_START, TEST_END, reb, COST)
            tr, te = slice_s(n, TRAIN_START, TRAIN_END), slice_s(n, TEST_START, TEST_END)
            mt, ms = metrics(tr), metrics(te)
            yr = yearly(n)
            rows.append(dict(name=f'{name} [{rname}]', train_cagr=mt['ann'], train_sh=mt['sharpe'],
                             train_mdd=mt['mdd'], test_cagr=ms['ann'], test_sh=ms['sharpe'],
                             test_mdd=ms['mdd'], turn=t,
                             y2021=yr.get(2021, np.nan), y2022=yr.get(2022, np.nan),
                             y2023=yr.get(2023, np.nan), y2024=yr.get(2024, np.nan),
                             y2025=yr.get(2025, np.nan), y2026=yr.get(2026, np.nan)))
    df = pd.DataFrame(rows)

    print('\n' + '=' * 118)
    print('TREND VARIANTS  (net of costs; train = 2020-11..2023-12, test = 2024-01..2026-11)')
    print('=' * 118)
    hdr = (f"{'variant':<30}{'trCAGR':>8}{'trSh':>6}{'trMDD':>7}"
           f"{'teCAGR':>8}{'teSh':>6}{'teMDD':>7}{'turn':>6}"
           f"{'2021':>7}{'2022':>7}{'2023':>7}{'2024':>7}{'2025':>7}{'2026':>7}")
    print(hdr)
    print('-' * 118)
    for _, r in df.iterrows():
        print(f"{r['name']:<30}{r['train_cagr']*100:>8.1f}{r['train_sh']:>6.2f}{r['train_mdd']*100:>7.1f}"
              f"{r['test_cagr']*100:>8.1f}{r['test_sh']:>6.2f}{r['test_mdd']*100:>7.1f}"
              f"{r['turn']*100:>6.0f}"
              f"{r['y2021']*100:>7.1f}{r['y2022']*100:>7.1f}{r['y2023']*100:>7.1f}"
              f"{r['y2024']*100:>7.1f}{r['y2025']*100:>7.1f}{r['y2026']*100:>7.1f}")
    print('=' * 118)

    # benchmarks, yearly
    for name, series in [('BTC buy & hold', fwd['BTC-USD']), ('equal-weight', bench)]:
        s = slice_s(series, TRAIN_START, TEST_END)
        yr = yearly(s)
        mt = metrics(slice_s(series, TRAIN_START, TRAIN_END))
        ms = metrics(slice_s(series, TEST_START, TEST_END))
        print(f"{name:<30}{mt['ann']*100:>8.1f}{mt['sharpe']:>6.2f}{mt['mdd']*100:>7.1f}"
              f"{ms['ann']*100:>8.1f}{ms['sharpe']:>6.2f}{ms['mdd']*100:>7.1f}{0:>6.0f}"
              f"{yr.get(2021,0)*100:>7.1f}{yr.get(2022,0)*100:>7.1f}{yr.get(2023,0)*100:>7.1f}"
              f"{yr.get(2024,0)*100:>7.1f}{yr.get(2025,0)*100:>7.1f}{yr.get(2026,0)*100:>7.1f}")


if __name__ == '__main__':
    main()
