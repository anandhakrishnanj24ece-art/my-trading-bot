"""
Honest backtest for the redesigned TrendVol signal.

Rules that make this trustworthy:
  * Weights at bar t are computed only from data up to and including t.
  * Those weights earn the return from t to t+1 (no same-bar look-ahead).
  * Costs are charged on turnover at the same 6.4 bps used live.
  * Reported train (2020-11 -> 2023-12) and test (2024-01 -> 2026-11) separately.

Run:  python backtest_trend.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from neural_flow.alpha.trend import TrendVolSignal

TRAIN_START = pd.Timestamp('2020-11-07')
TRAIN_END = pd.Timestamp('2024-01-01')
TEST_START = pd.Timestamp('2024-01-01')
TEST_END = pd.Timestamp('2026-11-08')

ASSETS = ['BTC-USD', 'ETH-USD']
UNIVERSE = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD', 'DOT-USD',
    'LINK-USD', 'LTC-USD', 'DOGE-USD', 'AVAX-USD', 'ATOM-USD', 'NEAR-USD',
    'ALGO-USD', 'FIL-USD',
]

COST = 0.0005 + 0.0002 * 0.7  # StealthExecution fee + 70% of slippage
ANN = 365
CACHE = Path('trend_data.pkl')


def load_data(period='6y', use_cache=True):
    if use_cache:
        for path in (CACHE, Path('research_data.pkl')):
            if path.exists():
                return pd.read_pickle(path)

    last_err = None
    for _ in range(5):
        try:
            raw = yf.download(UNIVERSE, period=period, interval='1d', progress=False,
                              threads=True, auto_adjust=True)
            if raw is None or len(raw) == 0 or 'Close' not in raw:
                raise RuntimeError('empty download')
            close, high, low, volume = (raw[k].ffill() for k in ('Close', 'High', 'Low', 'Volume'))
            if 'BTC-USD' not in close.columns or close['BTC-USD'].notna().sum() < 500:
                raise RuntimeError('BTC history missing')
            first = close.apply(lambda s: s.first_valid_index())
            valid = [c for c in close.columns
                     if first.get(c) is not None and first[c] <= pd.Timestamp('2021-06-01')
                     and close[c].loc['2021-06-01':].notna().mean() > 0.90]
            close, high, low, volume = (x[valid] for x in (close, high, low, volume))
            start = close.apply(lambda s: s.first_valid_index()).max()
            result = (close.loc[start:].ffill(), high.loc[start:].ffill(),
                      low.loc[start:].ffill(), volume.loc[start:].ffill())
            pd.to_pickle(result, CACHE)
            return result
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(3)
    raise RuntimeError(f'data download failed after retries: {last_err}')


def metrics(rets):
    if len(rets) == 0:
        return dict(total=0, ann=0, vol=0, sharpe=0, mdd=0, hit=0)
    total = float((1 + rets).prod() - 1)
    ann = (1 + total) ** (ANN / len(rets)) - 1
    vol = float(rets.std() * np.sqrt(ANN))
    eq = (1 + rets).cumprod()
    return dict(total=total, ann=ann, vol=vol,
                sharpe=(ann / vol if vol else 0.0),
                mdd=float((1 - eq / eq.cummax()).max()),
                hit=float((rets > 0).mean()))


def beta_alpha(rets, bench):
    df = pd.concat([rets, bench], axis=1).dropna()
    if len(df) < 30:
        return np.nan, np.nan
    cov = np.cov(df.iloc[:, 0], df.iloc[:, 1])
    beta = cov[0, 1] / cov[1, 1]
    alpha = (df.iloc[:, 0].mean() - beta * df.iloc[:, 1].mean()) * ANN
    return beta, alpha


def run(close, fwd, assets=ASSETS, lookback=50, vol_window=20,
        target_vol=0.15, max_leverage=2.0, cost=COST):
    assets = [a for a in assets if a in close.columns]
    weights = TrendVolSignal(
        lookback=lookback, vol_window=vol_window, target_vol=target_vol,
        max_leverage=max_leverage,
    ).generate_weights(close[assets])

    dates = weights.index.intersection(fwd.dropna(how='all').index)
    w, fr = weights.loc[dates], fwd.loc[dates]
    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()
    gross = (w * fr).sum(axis=1)
    net = gross - turnover * cost
    return w, gross, net, turnover


def slice_s(x, start, end):
    return x[(x.index >= start) & (x.index < end)]


def yearly(rets):
    return rets.groupby(rets.index.year).apply(lambda x: (1 + x).prod() - 1)


def row(name, rets, bench, start, end):
    r = slice_s(rets, start, end)
    m = metrics(r)
    b, a = beta_alpha(r, slice_s(bench, start, end))
    print(f"{name:<22}{m['total']*100:>9.1f}{m['ann']*100:>8.1f}{m['vol']*100:>8.1f}"
          f"{m['sharpe']:>8.2f}{m['mdd']*100:>8.1f}{m['hit']*100:>8.1f}"
          f"{b:>7.2f}{a*100:>9.1f}")


def main():
    close, high, low, volume = load_data('6y')
    fwd = close.shift(-1) / close - 1.0
    assets = [a for a in ASSETS if a in close.columns]
    print(f"Assets: {', '.join(assets)} | bars: {len(close)} | "
          f"{close.index[0].date()} -> {close.index[-1].date()}")

    w, gross, net, turnover = run(close, fwd, assets)
    btc = fwd['BTC-USD']
    ew = fwd.mean(axis=1)

    hdr = (f"{'window':<22}{'total%':>9}{'CAGR%':>8}{'vol%':>8}{'Sharpe':>8}"
           f"{'MDD%':>8}{'hit%':>8}{'beta':>7}{'alpha%':>9}")
    for label, s, e in [('train 2020-11..2023-12', TRAIN_START, TRAIN_END),
                        ('test  2024-01..2026-11', TEST_START, TEST_END),
                        ('full  2020-11..2026-11', TRAIN_START, TEST_END)]:
        print('\n' + '=' * 88)
        print(label)
        print('=' * 88)
        print(hdr)
        print('-' * 88)
        row('TrendVol (net)', net, btc, s, e)
        row('BTC buy & hold', btc, btc, s, e)
        row('equal-weight', ew, btc, s, e)

    print('\n' + '=' * 88)
    print('YEAR-BY-YEAR (net)')
    print('=' * 88)
    df = pd.DataFrame({
        'TrendVol': yearly(slice_s(net, TRAIN_START, TEST_END)),
        'BTC B&H': yearly(slice_s(btc, TRAIN_START, TEST_END)),
        'EW universe': yearly(slice_s(ew, TRAIN_START, TEST_END)),
    }) * 100
    print(df.round(1).to_string())

    full = metrics(slice_s(net, TRAIN_START, TEST_END))
    tstat = (net.mean() / (net.std() + 1e-12)) * np.sqrt(len(net))
    print(f"\nFull-period turnover: {turnover.mean()*100:.1f}% per bar "
          f"(~{turnover.mean()*365*COST*100:.1f}%/yr cost drag)")
    print(f"Daily-mean t-stat:   {tstat:.2f}")
    print(f"Latest target weights ({close.index[-1].date()}): "
          + ', '.join(f"{a}={w[a].iloc[-1]*100:.1f}%" for a in assets))


if __name__ == '__main__':
    main()
