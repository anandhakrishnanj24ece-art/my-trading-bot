"""
Honest backtest of the improved Adaptive v3 strategy.

Uses the exact same core as the live runner (neural_flow/alpha/v3_core.py):
    15-asset trend50 signal -> Square-Root Lasso (lambda=0.05)
    -> probabilistic sizing -> Elastic Hysteresis -> 2% no-trade band
    -> 60d Ledoit-Wolf 15% vol target -> BTC 150d regime filter (cash below it)

Rules that make this trustworthy:
  * weights at bar t use data up to and including t;
  * those weights earn the t -> t+1 return (no same-bar look-ahead);
  * the no-trade band and turnover use the actual, price-drifted book, not the
    previous target;
  * costs are charged on turnover at 6.4 bps per unit, plus 3 bps/day on the
    short book;
  * the first year of the 7-year cache is signal warm-up only, matching the
    1-year history the live runner downloads;
  * train (2020-11 -> 2023-12) and test (2024-01 -> 2026-11) are separate;
  * both the CAGR/vol Sharpe and the standard mean/std Sharpe are reported.

Run:  python backtest_v3_improved.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest_trend import (metrics, beta_alpha, slice_s, yearly, COST, ANN)
from neural_flow.alpha.v3_core import (AdaptiveWeightedV3, ProRiskManagerV3,
                                       trend_score, btc_regime_series)

TICKERS = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD',
           'DOT-USD', 'LINK-USD', 'LTC-USD', 'DOGE-USD', 'AVAX-USD',
           'ATOM-USD', 'NEAR-USD', 'ALGO-USD', 'FIL-USD']
TRAIN_START = pd.Timestamp('2020-11-07')
TRAIN_END = pd.Timestamp('2024-01-01')
TEST_START = pd.Timestamp('2024-01-01')
TEST_END = pd.Timestamp('2026-11-08')

LOOKBACK = 50
REGIME_LOOKBACK = 150
TARGET_VOL = 0.15
BASE_LEVERAGE = 2.0
COV_WINDOW = 60
LAMBDA_BASE = 0.05
TAU_BASE = 0.02
MAX_ASSET_WEIGHT = 1.0
NO_TRADE_BAND = 0.02
SHORT_COST_BPS = 3.0          # same daily short financing as the live runner
WARMUP_CACHE = Path('trend_data_7y_clean.pkl')
RAW_CACHE = Path('trend_data_7y.pkl')


def load_7y():
    """7 years of data: the first year is signal warm-up, not P&L.

    Prefer the cleaned cache (pre-launch FIL/AVAX placeholders removed).
    """
    path = WARMUP_CACHE if WARMUP_CACHE.exists() else RAW_CACHE
    if not path.exists():
        raise FileNotFoundError(f'{path} missing; run get_7y_data.py')
    return pd.read_pickle(path)


def run_v3_live_logic(close, fwd, cost=COST, short_cost_bps=0.0,
                      funding_bps=0.0, band=NO_TRADE_BAND, start=None):
    """Loop that mirrors run_adaptive_v3.py bar by bar.

    `current` is the *actual* portfolio at the start of each bar, after price
    drift.  The no-trade band and the turnover are measured against `current`,
    not against the previous target.  When the band says "hold", the real
    position has already drifted, so comparing with the old target would
    understate both the trade and its cost.
    """
    tickers = close.columns.tolist()
    score, conf = trend_score(close, LOOKBACK)
    regime = btc_regime_series(close, REGIME_LOOKBACK)
    # Late-listed assets have leading NaNs.  Their signal is zero there, so
    # zero returns keep the 60-day covariance finite without peeking ahead.
    rets = close.pct_change().fillna(0.0)
    dates = close.index if start is None else close.index[close.index >= start]

    optimizer = AdaptiveWeightedV3(lambda_base=LAMBDA_BASE, tau_base=TAU_BASE,
                                   critical_threshold=3.0)
    risk = ProRiskManagerV3(target_vol=TARGET_VOL,
                            base_leverage=BASE_LEVERAGE,
                            cov_window=COV_WINDOW, ann=365)

    current = np.zeros(len(tickers))
    rows, targets = [], []
    for d in dates[:-1]:
        sig = score.loc[d].reindex(tickers).fillna(0.0).values
        cf = conf.loc[d].reindex(tickers).fillna(0.0).values

        mkt_vol = rets.loc[:d].tail(COV_WINDOW).std().mean() * np.sqrt(ANN)
        raw = optimizer.optimize(sig, cf, market_vol=mkt_vol)
        lev = risk.get_portfolio_leverage(raw, rets.loc[:d])
        desired = np.clip(raw * lev, -MAX_ASSET_WEIGHT, MAX_ASSET_WEIGHT)

        # No-trade band, compared with the actual (drifted) book.
        if band > 0:
            target = np.where(np.abs(desired - current) >= band,
                              desired, current)
        else:
            target = desired

        # A regime exit must always go fully to cash, regardless of the band.
        if regime.loc[d] == 0:
            target = np.zeros(len(tickers))

        turnover = float(np.abs(target - current).sum())
        r = fwd.loc[d].values
        gross = float(np.sum(target * r))
        short_gross = float(np.abs(np.minimum(target, 0.0)).sum())
        gross_exposure = float(np.abs(target).sum())
        short_cost = short_cost_bps / 1e4 * short_gross
        funding_cost = funding_bps / 1e4 * gross_exposure
        rows.append((gross - cost * turnover - short_cost - funding_cost,
                     gross, turnover, lev, short_gross, funding_cost))
        targets.append((d, target.copy(), lev))

        # Drift the book to the start of the next bar.
        port_ret = gross
        denom = 1.0 + port_ret
        current = (target * (1.0 + r) / denom if denom > 1e-9
                   else np.zeros_like(target))

    df = pd.DataFrame(rows, index=dates[:-1],
                      columns=['net', 'gross', 'turn', 'lev', 'short', 'fund'])
    return df, targets


def show(name, df, fwd):
    print(f"\n{name}")
    print(f"{'window':<24}{'total%':>9}{'CAGR%':>8}{'vol%':>8}{'Shrp':>7}"
          f"{'StdShrp':>8}{'MDD%':>8}{'hit%':>8}{'alpha%':>9}{'turn%':>8}")
    print('-' * 98)
    for label, s, e in [('train 2020-11..2023-12', TRAIN_START, TRAIN_END),
                        ('test  2024-01..2026-11', TEST_START, TEST_END),
                        ('full  2020-11..2026-11', TRAIN_START, TEST_END)]:
        r = slice_s(df['net'], s, e)
        m = metrics(r)
        std_sharpe = r.mean() / (r.std() + 1e-12) * np.sqrt(ANN)
        _, a = beta_alpha(r, slice_s(fwd['BTC-USD'], s, e))
        turn = slice_s(df['turn'], s, e).mean()
        print(f"{label:<24}{m['total']*100:>9.1f}{m['ann']*100:>8.1f}"
              f"{m['vol']*100:>8.1f}{m['sharpe']:>7.2f}{std_sharpe:>8.2f}"
              f"{m['mdd']*100:>8.1f}{m['hit']*100:>8.1f}{a*100:>9.1f}{turn*100:>8.1f}")


def main():
    close, high, low, volume = load_7y()
    tickers = [t for t in TICKERS if t in close.columns]
    close = close[tickers]
    fwd = (close.shift(-1) / close - 1.0).fillna(0.0)
    print(f"universe: {len(tickers)} assets | {close.index[0].date()} -> {close.index[-1].date()}")
    print(f"warm-up: {close.index[0].date()} -> {TRAIN_START.date()} | "
          f"P&L: {TRAIN_START.date()} -> {close.index[-1].date()}")

    df, targets = run_v3_live_logic(close, fwd, short_cost_bps=SHORT_COST_BPS,
                                    start=TRAIN_START)
    show('IMPROVED v3 (15 assets, trend50, BTC150, lambda=0.05, band 2%, '
         'short 3 bp/day)', df, fwd)

    print('\nGross vs net (full):')
    for col in ('gross', 'net'):
        m = metrics(slice_s(df[col], TRAIN_START, TEST_END))
        print(f"  {col:<6} CAGR={m['ann']*100:6.1f}%  Sharpe={m['sharpe']:.2f}  "
              f"vol={m['vol']*100:5.1f}%  MDD={m['mdd']*100:5.1f}%")

    print('\nYear-by-year (net):')
    y = yearly(slice_s(df['net'], TRAIN_START, TEST_END)) * 100
    btc = yearly(slice_s(fwd['BTC-USD'], TRAIN_START, TEST_END)) * 100
    print(pd.DataFrame({'v3 improved': y, 'BTC B&H': btc}).round(1).to_string())

    net = slice_s(df['net'], TRAIN_START, TEST_END)
    tstat = net.mean() / (net.std() + 1e-12) * np.sqrt(len(net))
    print(f"\nDaily t-stat: {tstat:.2f} | avg leverage: "
          f"{slice_s(df['lev'], TRAIN_START, TEST_END).mean():.2f}x")

    # Cost stress, keeping the same short financing.
    df2, _ = run_v3_live_logic(close, fwd, cost=COST * 2,
                               short_cost_bps=SHORT_COST_BPS,
                               start=TRAIN_START)
    show('\nSame config, 2x costs + 3 bp/day short', df2, fwd)

    # Latest book (last completed bar in the backtest).
    d, w, lev = targets[-1]
    print(f"\nLatest backtest book ({d.date()}): leverage {lev:.2f}x")
    for t, x in sorted(zip(tickers, w), key=lambda kv: -abs(kv[1])):
        if abs(x) > 1e-4:
            print(f"  {t:<10} {x*100:>6.1f}%")


if __name__ == '__main__':
    main()
