"""Focused test: is the BTC 200-day regime filter special, or does any long
moving-average filter work?

Run:  python research_v3_regime.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from research_signal2 import load_data, metrics
from research_v3_improved import (V3_TICKERS, TRAIN_START, TRAIN_END,
                                  TEST_START, TEST_END, trend_signals,
                                  slice_s)
from research_v3_sweep import evaluate, FIXED

warnings.filterwarnings('ignore')


def main():
    close, high, low, volume = load_data('6y')
    tickers = [t for t in V3_TICKERS if t in close.columns]
    close = close[tickers]
    fwd = close.shift(-1) / close - 1.0

    print(f"{'variant':<34}{'trSh':>7}{'teSh':>7}{'fuSh':>7}{'teCAGR':>8}{'teMDD':>8}{'turn':>7}")
    print('-' * 78)

    for lb in (50,):
        s, c = trend_signals(close, lb)
        for sma in (50, 75, 100, 125, 150, 175, 200, 225, 250, 300):
            r = evaluate(close, fwd, s, c, regime=f'btc{sma}', regime_off=0.0, **FIXED)
            print(f"trend{lb} + btc{sma:<3}                      "
                  f"{r['tr']['sharpe']:>7.2f}{r['te']['sharpe']:>7.2f}"
                  f"{r['fu']['sharpe']:>7.2f}{r['te']['ann']*100:>8.1f}"
                  f"{r['te']['mdd']*100:>8.1f}{r['turn']*100:>7.1f}")

    print('-' * 78)
    for lb in (30, 40, 60, 80):
        s, c = trend_signals(close, lb)
        for sma in (150, 200, 250):
            r = evaluate(close, fwd, s, c, regime=f'btc{sma}', regime_off=0.0, **FIXED)
            print(f"trend{lb:<3} + btc{sma:<3}                      "
                  f"{r['tr']['sharpe']:>7.2f}{r['te']['sharpe']:>7.2f}"
                  f"{r['fu']['sharpe']:>7.2f}{r['te']['ann']*100:>8.1f}"
                  f"{r['te']['mdd']*100:>8.1f}{r['turn']*100:>7.1f}")


if __name__ == '__main__':
    main()
