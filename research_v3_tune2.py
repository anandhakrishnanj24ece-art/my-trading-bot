"""
Second-round tuning: combinations around the 15-asset / trend50 / BTC150 core.

Run:  python research_v3_tune2.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from research_signal2 import load_data, metrics, beta_alpha, COST, ANN
from research_v3_improved import (TRAIN_START, TRAIN_END, TEST_START, TEST_END,
                                  slice_s, yearly)
from research_v3_tune import (U15, run_config, evaluate, sig_trend, sig_xs_blend,
                              reg_btc, reg_btc_hyst, reg_combo)

warnings.filterwarnings('ignore')


def main():
    close_all, high, low, volume = load_data('6y')
    close = close_all[[t for t in U15 if t in close_all.columns]]
    fwd = close.shift(-1) / close - 1.0

    sig, conf = sig_trend(close, 50)
    xs, xsc = sig_xs_blend(close, 50, 0.3)
    reg = reg_btc(close, 150)
    reg_h = reg_btc_hyst(close, 150, 0.01)
    reg_c = reg_combo(close, 150, 50)

    configs = [
        ('band 0.01', dict(no_trade_band=0.01)),
        ('band 0.02', dict(no_trade_band=0.02)),
        ('band 0.03', dict(no_trade_band=0.03)),
        ('band 0.04', dict(no_trade_band=0.04)),
        ('band 0.05', dict(no_trade_band=0.05)),
        ('band .02 + cov40', dict(no_trade_band=0.02, cov_window=40)),
        ('band .02 + cov90', dict(no_trade_band=0.02, cov_window=90)),
        ('band .02 + vt20', dict(no_trade_band=0.02, target_vol=0.20)),
        ('band .02 + vt25', dict(no_trade_band=0.02, target_vol=0.25)),
        ('band .02 + cov40 + vt20', dict(no_trade_band=0.02, cov_window=40,
                                         target_vol=0.20)),
        ('band .03 + cov40', dict(no_trade_band=0.03, cov_window=40)),
        ('band .02 + xs blend', dict(no_trade_band=0.02)),
        ('band .02 + hyst regime', dict(no_trade_band=0.02)),
        ('band .02 + off=-0.5', dict(no_trade_band=0.02, regime_off=-0.5)),
        ('band .02 + inverse-vol', dict(no_trade_band=0.02, inverse_vol=True)),
        ('band .02 + max_asset=.5', dict(no_trade_band=0.02, max_asset=0.5)),
        ('band .02 + tau=.03', dict(no_trade_band=0.02, tau_base=0.03)),
        ('band .02 + lambda=.05', dict(no_trade_band=0.02, lambda_base=0.05)),
        ('band .02 + lev=3', dict(no_trade_band=0.02, base_leverage=3.0)),
        ('band .02 + no regime', dict(no_trade_band=0.02)),
        ('band .02 + rebal2d', dict(no_trade_band=0.02, rebalance_days=2)),
    ]

    hdr = (f"{'config':<38}{'trSh':>7}{'teSh':>7}{'fuSh':>7}"
           f"{'teCAGR':>8}{'teMDD':>8}{'turn':>7}")
    print('=' * 89)
    print('V3 TUNING ROUND 2 (15 assets, trend50)')
    print('=' * 89)
    print(hdr)
    print('-' * 89)

    results = []
    for name, ov in configs:
        s, c = (xs, xsc) if 'xs blend' in name else (sig, conf)
        r_ = reg_h if 'hyst' in name else (reg_c if 'combo' in name else reg)
        if 'no regime' in name:
            from research_v3_tune import reg_none
            r_ = reg_none(close)
        r = evaluate(close, fwd, s, c, r_, **ov)
        results.append((name, r))
        print(f"{name:<38}{r['tr']['sharpe']:>7.2f}{r['te']['sharpe']:>7.2f}"
              f"{r['fu']['sharpe']:>7.2f}{r['te']['ann']*100:>8.1f}"
              f"{r['te']['mdd']*100:>8.1f}{r['turn']*100:>7.1f}")

    print('-' * 89)

    # Best by test Sharpe and by full Sharpe.
    print('\nTop by TEST Sharpe:')
    for name, r in sorted(results, key=lambda x: x[1]['te']['sharpe'],
                          reverse=True)[:8]:
        print(f"  {name:<36} tr={r['tr']['sharpe']:.2f} te={r['te']['sharpe']:.2f}"
              f" full={r['fu']['sharpe']:.2f} CAGR={r['te']['ann']*100:.1f}%"
              f" MDD={r['te']['mdd']*100:.1f}%")

    print('\nTop by FULL Sharpe:')
    for name, r in sorted(results, key=lambda x: x[1]['fu']['sharpe'],
                          reverse=True)[:8]:
        print(f"  {name:<36} tr={r['tr']['sharpe']:.2f} te={r['te']['sharpe']:.2f}"
              f" full={r['fu']['sharpe']:.2f} CAGR={r['te']['ann']*100:.1f}%"
              f" MDD={r['te']['mdd']*100:.1f}%")

    # Detailed validation for band 0.02 and band 0.02 + VT20.
    for label, ov in [('BAND 0.02', dict(no_trade_band=0.02)),
                      ('BAND 0.02 + VT20', dict(no_trade_band=0.02,
                                                 target_vol=0.20))]:
        r = evaluate(close, fwd, sig, conf, reg, **ov)
        df = r['df']
        print('\n' + '=' * 89)
        print(label)
        print('=' * 89)
        print(f"{'window':<24}{'total%':>9}{'CAGR%':>8}{'vol%':>8}{'Sharpe':>8}"
              f"{'MDD%':>8}{'hit%':>8}{'alpha%':>9}")
        print('-' * 89)
        for wl, s, e in [('train', TRAIN_START, TRAIN_END),
                         ('test', TEST_START, TEST_END),
                         ('full', TRAIN_START, TEST_END)]:
            rr = slice_s(df['net'], s, e)
            m = metrics(rr)
            _, a = beta_alpha(rr, slice_s(fwd['BTC-USD'], s, e))
            print(f"{wl:<24}{m['total']*100:>9.1f}{m['ann']*100:>8.1f}"
                  f"{m['vol']*100:>8.1f}{m['sharpe']:>8.2f}{m['mdd']*100:>8.1f}"
                  f"{m['hit']*100:>8.1f}{a*100:>9.1f}")
        net = slice_s(df['net'], TRAIN_START, TEST_END)
        tstat = net.mean() / (net.std() + 1e-12) * np.sqrt(len(net))
        print(f"daily t-stat {tstat:.2f} | avg leverage "
              f"{slice_s(df['lev'], TRAIN_START, TEST_END).mean():.2f}x")
        y = yearly(slice_s(net, TRAIN_START, TEST_END)) * 100
        btc = yearly(slice_s(fwd['BTC-USD'], TRAIN_START, TEST_END)) * 100
        print(pd.DataFrame({'v3': y, 'BTC': btc}).round(1).to_string())

    # Cost stress on band 0.02.
    r2 = evaluate(close, fwd, sig, conf, reg, no_trade_band=0.02, cost=COST * 2)
    print('\nBAND 0.02, 2x costs:')
    for wl, s, e in [('test', TEST_START, TEST_END),
                     ('full', TRAIN_START, TEST_END)]:
        m = metrics(slice_s(r2['df']['net'], s, e))
        print(f"  {wl}: CAGR={m['ann']*100:.1f}% Sharpe={m['sharpe']:.2f} "
              f"MDD={m['mdd']*100:.1f}%")


if __name__ == '__main__':
    main()
