"""
Lambda / robustness sweep for the 15-asset, trend50, BTC150, band=0.02 v3.

Run:  python research_v3_lambda.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from research_signal2 import load_data, metrics, beta_alpha, COST, ANN
from research_v3_improved import (TRAIN_START, TRAIN_END, TEST_START, TEST_END,
                                  slice_s, yearly)
from research_v3_tune import (U9, U15, evaluate, sig_trend, sig_xs_blend,
                              reg_btc, reg_btc_hyst, reg_none)

warnings.filterwarnings('ignore')

BASE = dict(no_trade_band=0.02)


def main():
    close_all, high, low, volume = load_data('6y')
    c15 = close_all[[t for t in U15 if t in close_all.columns]]
    c9 = close_all[[t for t in U9 if t in close_all.columns]]
    f15 = c15.shift(-1) / c15 - 1.0
    f9 = c9.shift(-1) / c9 - 1.0
    s15, q15 = sig_trend(c15, 50)
    s9, q9 = sig_trend(c9, 50)
    xs15, xq15 = sig_xs_blend(c15, 50, 0.3)
    r15 = reg_btc(c15, 150)
    r9 = reg_btc(c9, 150)
    rh15 = reg_btc_hyst(c15, 150, 0.01)

    hdr = (f"{'config':<40}{'trSh':>7}{'teSh':>7}{'fuSh':>7}"
           f"{'teCAGR':>8}{'teMDD':>8}{'turn':>7}")
    print('=' * 91)
    print('LAMBDA SWEEP (15a, trend50, btc150, band=0.02)')
    print('=' * 91)
    print(hdr)
    print('-' * 91)

    results = []
    for lam in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12):
        r = evaluate(c15, f15, s15, q15, r15, lambda_base=lam, **BASE)
        results.append((f'lambda={lam}', r))
        print(f"{'lambda=' + str(lam):<40}{r['tr']['sharpe']:>7.2f}"
              f"{r['te']['sharpe']:>7.2f}{r['fu']['sharpe']:>7.2f}"
              f"{r['te']['ann']*100:>8.1f}{r['te']['mdd']*100:>8.1f}"
              f"{r['turn']*100:>7.1f}")

    print('-' * 91)

    # Interactions at lambda=0.05.
    print('\nINTERACTIONS AT LAMBDA=0.05')
    print(hdr)
    print('-' * 91)
    tests = [
        ('9 assets', c9, f9, s9, q9, r9, {}),
        ('band 0.01', c15, f15, s15, q15, r15, dict(no_trade_band=0.01)),
        ('band 0.03', c15, f15, s15, q15, r15, dict(no_trade_band=0.03)),
        ('band 0.04', c15, f15, s15, q15, r15, dict(no_trade_band=0.04)),
        ('cov 40', c15, f15, s15, q15, r15, dict(cov_window=40)),
        ('vt 20%', c15, f15, s15, q15, r15, dict(target_vol=0.20)),
        ('vt 25%', c15, f15, s15, q15, r15, dict(target_vol=0.25)),
        ('off=-0.5', c15, f15, s15, q15, r15, dict(regime_off=-0.5)),
        ('xs blend', c15, f15, xs15, xq15, r15, {}),
        ('hyst regime', c15, f15, s15, q15, rh15, {}),
        ('no regime', c15, f15, s15, q15, reg_none(c15), {}),
    ]
    for name, cc, ff, ss, qq, rr, ov in tests:
        r = evaluate(cc, ff, ss, qq, rr, lambda_base=0.05, **dict(BASE, **ov))
        results.append((name, r))
        print(f"{name:<40}{r['tr']['sharpe']:>7.2f}{r['te']['sharpe']:>7.2f}"
              f"{r['fu']['sharpe']:>7.2f}{r['te']['ann']*100:>8.1f}"
              f"{r['te']['mdd']*100:>8.1f}{r['turn']*100:>7.1f}")

    # Detailed validation for the chosen config.
    r = evaluate(c15, f15, s15, q15, r15, lambda_base=0.05, **BASE)
    df = r['df']
    print('\n' + '=' * 91)
    print('DETAILED: 15a trend50 btc150 band=0.02 lambda=0.05')
    print('=' * 91)
    print(f"{'window':<24}{'total%':>9}{'CAGR%':>8}{'vol%':>8}{'Sharpe':>8}"
          f"{'MDD%':>8}{'hit%':>8}{'alpha%':>9}")
    print('-' * 91)
    for wl, s, e in [('train', TRAIN_START, TRAIN_END),
                     ('test', TEST_START, TEST_END),
                     ('full', TRAIN_START, TEST_END)]:
        rr = slice_s(df['net'], s, e)
        m = metrics(rr)
        _, a = beta_alpha(rr, slice_s(f15['BTC-USD'], s, e))
        print(f"{wl:<24}{m['total']*100:>9.1f}{m['ann']*100:>8.1f}"
              f"{m['vol']*100:>8.1f}{m['sharpe']:>8.2f}{m['mdd']*100:>8.1f}"
              f"{m['hit']*100:>8.1f}{a*100:>9.1f}")

    net = slice_s(df['net'], TRAIN_START, TEST_END)
    tstat = net.mean() / (net.std() + 1e-12) * np.sqrt(len(net))
    roll = net.rolling(252)
    roll_sh = (roll.apply(lambda x: (1 + x).prod() ** (365 / 252) - 1)
               / (roll.std() * np.sqrt(ANN)))
    print(f"daily t-stat {tstat:.2f} | avg leverage "
          f"{slice_s(df['lev'], TRAIN_START, TEST_END).mean():.2f}x")
    print(f"rolling 1y Sharpe min={roll_sh.min():.2f} median={roll_sh.median():.2f} "
          f"max={roll_sh.max():.2f} positive={100*(roll_sh > 0).mean():.0f}%")

    y = yearly(slice_s(net, TRAIN_START, TEST_END)) * 100
    btc = yearly(slice_s(f15['BTC-USD'], TRAIN_START, TEST_END)) * 100
    print(pd.DataFrame({'v3': y, 'BTC': btc}).round(1).to_string())

    # Cost stress.
    r2 = evaluate(c15, f15, s15, q15, r15, lambda_base=0.05,
                  **dict(BASE, cost=COST * 2))
    print('\n2x costs:')
    for wl, s, e in [('test', TEST_START, TEST_END),
                     ('full', TRAIN_START, TEST_END)]:
        m = metrics(slice_s(r2['df']['net'], s, e))
        print(f"  {wl}: CAGR={m['ann']*100:.1f}% Sharpe={m['sharpe']:.2f} "
              f"MDD={m['mdd']*100:.1f}%")


if __name__ == '__main__':
    main()
