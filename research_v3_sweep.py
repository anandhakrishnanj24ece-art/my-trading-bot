"""
Robustness sweep for the improved v3 (trend signal + fixed risk layer +
BTC 200d regime filter).

Run:  python research_v3_sweep.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from research_signal2 import load_data, metrics, beta_alpha, COST, ANN
from research_v3_improved import (V3_TICKERS, TRAIN_START, TRAIN_END,
                                  TEST_START, TEST_END, run_v3,
                                  trend_signals, mom_signals, combo_signals,
                                  slice_s, yearly)

warnings.filterwarnings('ignore')

FIXED = dict(tau_mode='vol', base_leverage=2.0, cov_window=60,
             risk_ann=365, max_asset=1.0)


def evaluate(close, fwd, sig, conf, **ov):
    df = run_v3(close, fwd, sig, conf, **ov)
    tr = metrics(slice_s(df['net'], TRAIN_START, TRAIN_END))
    te = metrics(slice_s(df['net'], TEST_START, TEST_END))
    fu = metrics(slice_s(df['net'], TRAIN_START, TEST_END))
    turn = slice_s(df['turn'], TRAIN_START, TEST_END).mean()
    return dict(tr=tr, te=te, fu=fu, turn=turn, df=df)


def line(name, r):
    return (f"{name:<40}{r['tr']['sharpe']:>7.2f}{r['te']['sharpe']:>7.2f}"
            f"{r['fu']['sharpe']:>7.2f}{r['te']['ann']*100:>8.1f}"
            f"{r['te']['mdd']*100:>8.1f}{r['turn']*100:>7.1f}")


def main():
    close, high, low, volume = load_data('6y')
    tickers = [t for t in V3_TICKERS if t in close.columns]
    close, volume = close[tickers], volume[tickers]
    fwd = close.shift(-1) / close - 1.0

    cache = {}

    def sig_for(kind, lb=50):
        key = (kind, lb)
        if key not in cache:
            if kind == 'trend':
                cache[key] = trend_signals(close, lb)
            elif kind == 'mom':
                cache[key] = mom_signals(close, lb)
            else:
                cache[key] = combo_signals(close)
        return cache[key]

    hdr = (f"{'variant':<40}{'trSh':>7}{'teSh':>7}{'fuSh':>7}"
           f"{'teCAGR':>8}{'teMDD':>8}{'turn':>7}")
    print('=' * 84)
    print('ROBUSTNESS SWEEP (all with fixed risk layer + BTC200 regime, off=0)')
    print('=' * 84)
    print(hdr)
    print('-' * 84)

    out = []

    # --- base choice ---
    s, c = sig_for('trend', 50)
    base = evaluate(close, fwd, s, c, regime='btc200', regime_off=0.0, **FIXED)
    out.append(('trend50 + btc200 [base]', base))
    print(line(out[-1][0], base))

    # --- trend lookback sensitivity ---
    for lb in (20, 30, 40, 60, 80, 100, 120):
        s, c = sig_for('trend', lb)
        r = evaluate(close, fwd, s, c, regime='btc200', regime_off=0.0, **FIXED)
        out.append((f'trend{lb} + btc200', r))
        print(line(out[-1][0], r))

    # --- regime SMA sensitivity ---
    for sma in (100, 150, 250):
        s, c = sig_for('trend', 50)
        r = evaluate(close, fwd, s, c, regime=f'btc{sma}', regime_off=0.0, **FIXED)
        out.append((f'trend50 + btc{sma}', r))
        print(line(out[-1][0], r))

    # --- regime off level ---
    s, c = sig_for('trend', 50)
    for off in (0.3, 0.5):
        r = evaluate(close, fwd, s, c, regime='btc200', regime_off=off, **FIXED)
        out.append((f'trend50 + btc200 off={off}', r))
        print(line(out[-1][0], r))

    # --- vol target ---
    for vt in (0.10, 0.20, 0.25):
        s, c = sig_for('trend', 50)
        r = evaluate(close, fwd, s, c, regime='btc200', regime_off=0.0,
                     **dict(FIXED, target_vol=vt))
        out.append((f'trend50 + btc200 vt={vt:.0%}', r))
        print(line(out[-1][0], r))

    # --- lasso strength ---
    for lam in (0.0, 0.02, 0.05):
        s, c = sig_for('trend', 50)
        r = evaluate(close, fwd, s, c, regime='btc200', regime_off=0.0,
                     **dict(FIXED, lambda_base=lam))
        out.append((f'trend50 + btc200 lambda={lam}', r))
        print(line(out[-1][0], r))

    # --- hysteresis strength ---
    for tau in (0.005, 0.02, 0.03):
        s, c = sig_for('trend', 50)
        r = evaluate(close, fwd, s, c, regime='btc200', regime_off=0.0,
                     **dict(FIXED, tau_base=tau))
        out.append((f'trend50 + btc200 tau={tau}', r))
        print(line(out[-1][0], r))

    # --- alternative signals with the regime filter ---
    for kind, lb, label in [('mom', 60, 'mom60'), ('combo', 0, 'combo20/50/100')]:
        s, c = sig_for(kind, lb)
        r = evaluate(close, fwd, s, c, regime='btc200', regime_off=0.0, **FIXED)
        out.append((f'{label} + btc200', r))
        print(line(out[-1][0], r))

    # --- double costs on the base choice ---
    s, c = sig_for('trend', 50)
    r = evaluate(close, fwd, s, c, regime='btc200', regime_off=0.0,
                 **dict(FIXED, cost=COST * 2))
    out.append(('trend50 + btc200, 2x costs', r))
    print(line(out[-1][0], r))

    # --- yearly for the base ---
    print('\n' + '=' * 84)
    print('BASE YEAR-BY-YEAR (net)')
    print('=' * 84)
    y = yearly(slice_s(base['df']['net'], TRAIN_START, TEST_END)) * 100
    btc = yearly(slice_s(fwd['BTC-USD'], TRAIN_START, TEST_END)) * 100
    print(pd.DataFrame({'v3 improved': y, 'BTC B&H': btc}).round(1).to_string())

    # --- rolling 1y Sharpe for the base ---
    net = slice_s(base['df']['net'], TRAIN_START, TEST_END)
    roll = net.rolling(252)
    roll_sh = (roll.apply(lambda x: (1 + x).prod() ** (365 / 252) - 1)
               / (roll.std() * np.sqrt(ANN)))
    print(f"\nRolling 1y Sharpe: min={roll_sh.min():.2f}  median={roll_sh.median():.2f}  "
          f"max={roll_sh.max():.2f}  positive={100*(roll_sh > 0).mean():.0f}%")
    print(f"Full-period daily t-stat: {net.mean()/(net.std()+1e-12)*np.sqrt(len(net)):.2f}")

    # --- rank by full Sharpe ---
    print('\nTop 8 by FULL-period Sharpe:')
    for name, r in sorted(out, key=lambda x: x[1]['fu']['sharpe'], reverse=True)[:8]:
        print(f"  {name:<38} tr={r['tr']['sharpe']:.2f}  te={r['te']['sharpe']:.2f}  "
              f"full={r['fu']['sharpe']:.2f}  teCAGR={r['te']['ann']*100:.1f}%")


if __name__ == '__main__':
    main()
