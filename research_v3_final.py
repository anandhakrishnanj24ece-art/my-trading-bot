"""Final validation of the chosen improved-v3 configuration.

trend50 signal + BTC150 regime + fixed risk layer + tau=0.02, lambda=0.02.

Run:  python research_v3_final.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from research_signal2 import load_data, metrics, beta_alpha, COST, ANN
from research_v3_improved import (V3_TICKERS, TRAIN_START, TRAIN_END,
                                  TEST_START, TEST_END, run_v3, trend_signals,
                                  slice_s, yearly)

warnings.filterwarnings('ignore')

CFG = dict(tau_mode='vol', tau_base=0.02, lambda_base=0.02,
           base_leverage=2.0, cov_window=60, risk_ann=365, max_asset=1.0,
           target_vol=0.15)


def show(name, df, fwd, cost_note=''):
    print(f"\n{name} {cost_note}")
    print(f"{'window':<22}{'total%':>9}{'CAGR%':>8}{'vol%':>8}{'Sharpe':>8}"
          f"{'MDD%':>8}{'hit%':>8}{'alpha%':>9}{'turn%':>8}")
    print('-' * 88)
    for label, s, e in [('train 2020-11..2023-12', TRAIN_START, TRAIN_END),
                        ('test  2024-01..2026-11', TEST_START, TEST_END),
                        ('full  2020-11..2026-11', TRAIN_START, TEST_END)]:
        r = slice_s(df['net'], s, e)
        m = metrics(r)
        b, a = beta_alpha(r, slice_s(fwd['BTC-USD'], s, e))
        turn = slice_s(df['turn'], s, e).mean()
        print(f"{label:<22}{m['total']*100:>9.1f}{m['ann']*100:>8.1f}{m['vol']*100:>8.1f}"
              f"{m['sharpe']:>8.2f}{m['mdd']*100:>8.1f}{m['hit']*100:>8.1f}"
              f"{a*100:>9.1f}{turn*100:>8.1f}")


def main():
    close, high, low, volume = load_data('6y')
    tickers = [t for t in V3_TICKERS if t in close.columns]
    close = close[tickers]
    fwd = close.shift(-1) / close - 1.0

    sig, conf = trend_signals(close, 50)
    df = run_v3(close, fwd, sig, conf, regime='btc150', regime_off=0.0, **CFG)
    show('CHOSEN: trend50 + BTC150', df, fwd)

    # gross vs net
    print('\nGross vs net (full):')
    for col in ('gross', 'net'):
        m = metrics(slice_s(df[col], TRAIN_START, TEST_END))
        print(f"  {col:<6} CAGR={m['ann']*100:6.1f}%  Sharpe={m['sharpe']:.2f}  "
              f"vol={m['vol']*100:5.1f}%  MDD={m['mdd']*100:5.1f}%")

    # yearly
    print('\nYear-by-year (net):')
    y = yearly(slice_s(df['net'], TRAIN_START, TEST_END)) * 100
    btc = yearly(slice_s(fwd['BTC-USD'], TRAIN_START, TEST_END)) * 100
    print(pd.DataFrame({'v3 improved': y, 'BTC B&H': btc}).round(1).to_string())

    # significance / rolling
    net = slice_s(df['net'], TRAIN_START, TEST_END)
    tstat = net.mean() / (net.std() + 1e-12) * np.sqrt(len(net))
    roll = net.rolling(252)
    roll_sh = (roll.apply(lambda x: (1 + x).prod() ** (365 / 252) - 1)
               / (roll.std() * np.sqrt(ANN)))
    print(f"\nDaily t-stat: {tstat:.2f} | avg leverage: "
          f"{slice_s(df['lev'], TRAIN_START, TEST_END).mean():.2f}x")
    print(f"Rolling 1y Sharpe: min={roll_sh.min():.2f} median={roll_sh.median():.2f} "
          f"max={roll_sh.max():.2f} positive={100*(roll_sh > 0).mean():.0f}%")

    # cost sensitivity
    df2 = run_v3(close, fwd, sig, conf, regime='btc150', regime_off=0.0,
                 **dict(CFG, cost=COST * 2))
    show('\nSame config, 2x costs', df2, fwd)

    # vol target 20%
    df3 = run_v3(close, fwd, sig, conf, regime='btc150', regime_off=0.0,
                 **dict(CFG, target_vol=0.20))
    show('\nSame config, 20% vol target', df3, fwd)


if __name__ == '__main__':
    main()
