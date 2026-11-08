"""
Parameter sensitivity + rolling robustness for the trend signal.

Run:  python research_trend_sweep.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from research_signal2 import load_data, metrics, beta_alpha, ANN, COST
from research_signal3 import run_weights, slice_s, yearly
from research_trend import multi_trend, single_trend

warnings.filterwarnings('ignore')

TRAIN_START = pd.Timestamp('2020-11-07')
TRAIN_END = pd.Timestamp('2024-01-01')
TEST_START = pd.Timestamp('2024-01-01')
TEST_END = pd.Timestamp('2026-11-08')


def multi_trend_confirm(close, assets, fast=50, slow=100, target=0.15, max_lev=2.0):
    ret = close[assets].pct_change()
    vol = ret.rolling(20).std() * np.sqrt(ANN)
    above = close[assets] > close[assets].rolling(fast).mean()
    stack = close[assets].rolling(fast).mean() > close[assets].rolling(slow).mean()
    sig = (above & stack).astype(float)
    raw = sig / (vol + 1e-9)
    gross = raw.abs().sum(axis=1)
    w = raw.div(gross.replace(0, np.nan), axis=0).fillna(0.0)
    port_vol = np.sqrt((w ** 2 * vol ** 2).sum(axis=1))
    scale = (target / (port_vol + 1e-9)).clip(0, max_lev)
    return w.mul(scale, axis=0)


def evaluate(w, fwd, close):
    g, n, t = run_weights(w, fwd, TRAIN_START, TEST_END, 1, COST)
    return (n, metrics(slice_s(n, TRAIN_START, TRAIN_END)),
            metrics(slice_s(n, TEST_START, TEST_END)), metrics(n), t)


def main():
    close, high, low, volume = load_data('6y')
    fwd = close.shift(-1) / close - 1.0

    # ---- grid: BTC+ETH, lookback x vol target ----
    grid = []
    for lb in (20, 30, 40, 50, 60, 70, 80, 100, 120, 150):
        for vt in (0.10, 0.15, 0.20, 0.25):
            w = multi_trend(close, ['BTC-USD', 'ETH-USD'], lb, target=vt)
            n, tr, te, full, t = evaluate(w, fwd, close)
            grid.append(dict(lb=lb, vt=vt, tr=tr, te=te, full=full, turn=t, rets=n))
    gdf = pd.DataFrame(grid)
    gdf['test_sh'] = gdf['te'].map(lambda x: x['sharpe'])
    gdf['full_sh'] = gdf['full'].map(lambda x: x['sharpe'])

    print('BTC+ETH trend: TEST Sharpe grid (rows=lookback, cols=vol target)')
    print('       ' + ''.join(f'{v:>9.0%}' for v in (0.10, 0.15, 0.20, 0.25)))
    for lb in (20, 30, 40, 50, 60, 70, 80, 100, 120, 150):
        vals = [gdf[(gdf.lb == lb) & (gdf.vt == vt)]['te'].iloc[0]['sharpe']
                for vt in (0.10, 0.15, 0.20, 0.25)]
        print(f'  {lb:<5d}' + ''.join(f'{v:>9.2f}' for v in vals))

    print('\nTop 10 configs by TEST Sharpe:')
    for _, r in gdf.sort_values('test_sh', ascending=False).head(10).iterrows():
        print(f"  lb={r['lb']:<4d} vt={r['vt']:.0%}  train Sh={r['tr']['sharpe']:.2f}  "
              f"test Sh={r['te']['sharpe']:.2f}  test CAGR={r['te']['ann']*100:.1f}%  "
              f"test MDD={r['te']['mdd']*100:.1f}%  turn={r['turn']*100:.0f}%")

    print('\nTop 10 configs by FULL-period Sharpe:')
    for _, r in gdf.sort_values('full_sh', ascending=False).head(10).iterrows():
        print(f"  lb={r['lb']:<4d} vt={r['vt']:.0%}  full Sh={r['full']['sharpe']:.2f}  "
              f"train Sh={r['tr']['sharpe']:.2f}  test Sh={r['te']['sharpe']:.2f}")

    # ---- chosen config + confirm filter + single assets ----
    print('\n' + '=' * 96)
    print('CANDIDATE COMPARISON (daily rebalance, net)')
    print('=' * 96)
    print(f"{'variant':<34}{'trCAGR':>8}{'trSh':>6}{'trMDD':>7}"
          f"{'teCAGR':>8}{'teSh':>6}{'teMDD':>7}{'fullSh':>7}{'turn':>6}")
    cands = {
        'BTC+ETH trend50 vt15': multi_trend(close, ['BTC-USD', 'ETH-USD'], 50, 0.15),
        'BTC+ETH trend50 vt10': multi_trend(close, ['BTC-USD', 'ETH-USD'], 50, 0.10),
        'BTC+ETH trend50 vt20': multi_trend(close, ['BTC-USD', 'ETH-USD'], 50, 0.20),
        'BTC+ETH trend50/100 confirm': multi_trend_confirm(close, ['BTC-USD', 'ETH-USD'], 50, 100, 0.15),
        'BTC+ETH trend50/200 confirm': multi_trend_confirm(close, ['BTC-USD', 'ETH-USD'], 50, 200, 0.15),
        'BTC+ETH trend40/100 confirm': multi_trend_confirm(close, ['BTC-USD', 'ETH-USD'], 40, 100, 0.15),
        'BTC trend50 vt15': multi_trend(close, ['BTC-USD'], 50, 0.15),
        'ETH trend50 vt15': multi_trend(close, ['ETH-USD'], 50, 0.15),
        'BTC+ETH+SOL trend50': multi_trend(close, [c for c in ['BTC-USD', 'ETH-USD', 'SOL-USD'] if c in close.columns], 50, 0.15),
    }
    for name, w in cands.items():
        n, tr, te, full, t = evaluate(w, fwd, close)
        print(f"{name:<34}{tr['ann']*100:>8.1f}{tr['sharpe']:>6.2f}{tr['mdd']*100:>7.1f}"
              f"{te['ann']*100:>8.1f}{te['sharpe']:>6.2f}{te['mdd']*100:>7.1f}"
              f"{full['sharpe']:>7.2f}{t*100:>6.0f}")

    # ---- chosen: rolling 1y Sharpe and t-stat ----
    w = multi_trend(close, ['BTC-USD', 'ETH-USD'], 50, 0.15)
    n, tr, te, full, t = evaluate(w, fwd, close)
    daily_sh = n.mean() / (n.std() + 1e-12)
    tstat = daily_sh * np.sqrt(len(n))
    roll = n.rolling(252)
    roll_sh = (roll.apply(lambda x: (1 + x).prod() ** (365 / 252) - 1)
               / (roll.std() * np.sqrt(ANN)))
    print('\nChosen: BTC+ETH trend50, vol target 15%, daily rebalance')
    print(f"  full: total={full['total']*100:.1f}%  CAGR={full['ann']*100:.1f}%  "
          f"vol={full['vol']*100:.1f}%  Sharpe={full['sharpe']:.2f}  MDD={full['mdd']*100:.1f}%")
    print(f"  t-stat of daily mean = {tstat:.2f} over {len(n)} days")
    print(f"  rolling 1y Sharpe: min={roll_sh.min():.2f}  median={roll_sh.median():.2f}  "
          f"max={roll_sh.max():.2f}  %positive={100*(roll_sh>0).mean():.0f}%")
    print('\n  Yearly returns:')
    for y, v in yearly(n).items():
        print(f'    {y}: {v*100:+.1f}%')
    print('\n  Beta / alpha vs BTC buy&hold:')
    b, a = beta_alpha(n, fwd['BTC-USD'])
    print(f'    beta={b:.2f}  alpha={a*100:.1f}%/yr')


if __name__ == '__main__':
    main()
