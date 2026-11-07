"""
Audit 1: data quality for the 15-asset v3 universe.

Checks:
  * history length / start dates
  * zero-return frequency (stale prices)
  * extreme daily moves
  * missing values
  * rough liquidity (average dollar volume)
  * BTC correlation / beta of the live book

Run:  python audit_v3_data.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_trend import load_data
from backtest_v3_improved import TICKERS, run_v3_live_logic


def main():
    close, high, low, volume = load_data('6y')
    tickers = [t for t in TICKERS if t in close.columns]
    close, volume = close[tickers], volume[tickers]

    print('=' * 110)
    print('DATA QUALITY')
    print('=' * 110)
    print(f"{'asset':<10}{'first':>12}{'last':>12}{'rows':>6}{'nan':>5}"
          f"{'zero%':>7}{'min day%':>10}{'max day%':>10}{'med $vol(m)':>13}")
    print('-' * 110)
    for t in tickers:
        s = close[t]
        r = s.pct_change()
        dv = (s * volume[t]).dropna()
        print(f"{t:<10}{str(s.first_valid_index().date()):>12}"
              f"{str(s.last_valid_index().date()):>12}{len(s):>6}"
              f"{int(s.isna().sum()):>5}{100*(r == 0).mean():>7.2f}"
              f"{100*r.min():>10.2f}{100*r.max():>10.2f}"
              f"{dv.median()/1e6:>13.2f}")

    print('\nDate range:', close.index[0].date(), '->', close.index[-1].date(),
          '| rows:', len(close))
    print('Duplicate dates:', int(close.index.duplicated().sum()))
    print('Any price <= 0:', bool((close <= 0).any().any()))
    print('Any NaN in close:', int(close.isna().sum().sum()))

    # Liquidity by year (median daily dollar volume, $m).
    print('\nMedian daily dollar volume by year ($m):')
    dv = close * volume
    by_year = dv.groupby(dv.index.year).median() / 1e6
    print(by_year.round(1).to_string())

    # Performance / exposure summary.
    fwd = close.shift(-1) / close - 1.0
    df, targets = run_v3_live_logic(close, fwd)
    net = df['net']

    print('\n' + '=' * 110)
    print('BOOK EXPOSURE / RISK')
    print('=' * 110)
    long_exp, short_exp, gross, net_exp = [], [], [], []
    for d, w, lev in targets:
        long_exp.append(float(w[w > 0].sum()))
        short_exp.append(float(w[w < 0].sum()))
        gross.append(float(np.abs(w).sum()))
        net_exp.append(float(w.sum()))
    exp = pd.DataFrame({'long': long_exp, 'short': short_exp,
                        'gross': gross, 'net': net_exp},
                       index=[x[0] for x in targets])
    print(exp.describe().round(3).to_string())

    # Standard Sharpe and tail stats.
    r = net.dropna()
    std_sharpe = r.mean() / r.std() * np.sqrt(365)
    cagr = (1 + r).prod() ** (365 / len(r)) - 1
    print(f"\nCAGR/vol Sharpe (reported): {cagr / (r.std()*np.sqrt(365)):.2f}")
    print(f"Standard Sharpe (mean/std): {std_sharpe:.2f}")
    print(f"Skew: {r.skew():.2f} | Excess kurtosis: {r.kurtosis():.2f}")
    print(f"Worst day: {100*r.min():.2f}% | Best day: {100*r.max():.2f}%")
    for q in (0.01, 0.05):
        print(f"Daily CVaR {q:.0%}: {100*r.quantile(q):.2f}%")
    for n in (7, 30, 90):
        rr = (1 + r).rolling(n).apply(np.prod, raw=True) - 1
        print(f"Worst {n}d: {100*rr.min():.2f}%")

    # Beta / correlation to BTC.
    btc = fwd['BTC-USD'].reindex(r.index)
    dfb = pd.concat([r, btc], axis=1).dropna()
    beta = np.cov(dfb.iloc[:, 0], dfb.iloc[:, 1])[0, 1] / np.cov(dfb.iloc[:, 0], dfb.iloc[:, 1])[1, 1]
    corr = dfb.iloc[:, 0].corr(dfb.iloc[:, 1])
    print(f"\nBeta to BTC: {beta:.2f} | Correlation: {corr:.2f}")


if __name__ == '__main__':
    main()
