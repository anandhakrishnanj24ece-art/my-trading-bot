"""
Corrected TrendVol backtest.

The previous version of this file applied the drawdown overlay using an equity
curve that already contained the return from t -> t+1. That is look-ahead: the
overlay "knew" the next bar's loss and cut exposure before it happened. It
inflated CAGR from ~28% to ~58% and Sharpe from ~1.35 to ~2.55.

This version uses a fully causal recursive loop:
    * at bar t, equity/peak are known only through t-1;
    * the overlay multiplier is applied to target weights at t;
    * those weights earn the return t -> t+1;
    * equity and peak are updated after the return is realised.

It also reports the survivorship-bias test (15 survivors vs 44 assets including
LUNC/USTC/FTT/CEL), cost sensitivity, and a block-bootstrap t-stat.

Run: python backtest_trend_broad.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from neural_flow.alpha.trend import TrendVolSignal
from neural_flow.risk.model import drawdown_multiplier

TRAIN_START = pd.Timestamp('2020-11-07')
TRAIN_END = pd.Timestamp('2024-01-01')
TEST_START = pd.Timestamp('2024-01-01')
TEST_END = pd.Timestamp('2026-11-08')
ANN = 365
COST = 0.0005 + 0.0002 * 0.7  # 6.4 bps per unit turnover

U2 = ['BTC-USD', 'ETH-USD']
U15 = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD', 'DOT-USD',
       'LINK-USD', 'LTC-USD', 'DOGE-USD', 'AVAX-USD', 'ATOM-USD', 'NEAR-USD',
       'ALGO-USD', 'FIL-USD']


def first_part(path):
    d = pd.read_pickle(path)
    return d[0] if isinstance(d, (tuple, list)) else d


def metrics(rets):
    rets = rets.dropna()
    if len(rets) < 2:
        return dict(total=0.0, ann=0.0, vol=0.0, sharpe=0.0, mdd=0.0, hit=0.0)
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


def run_causal(close, fwd, assets, cost=COST, overlay=True,
               lookback=50, vol_window=20, target_vol=0.15, max_leverage=2.0,
               dd_caution=0.05, dd_halt=0.15, dd_floor=0.25):
    """Fully causal TrendVol backtest."""
    assets = [a for a in assets if a in close.columns]
    weights = TrendVolSignal(
        lookback=lookback, vol_window=vol_window, target_vol=target_vol,
        max_leverage=max_leverage,
    ).generate_weights(close[assets]).fillna(0.0)

    dates = weights.index.intersection(fwd.dropna(how='all').index)
    w0 = weights.loc[dates]
    # Missing forward returns are zero-contribution, not NaN-propagating.
    fr = fwd.loc[dates].fillna(0.0)

    net, turn, actual = [], [], []
    eq, peak = 1.0, 1.0
    prev = pd.Series(0.0, index=w0.columns)
    for t in range(len(w0)):
        if overlay:
            m = drawdown_multiplier(peak, eq, caution=dd_caution,
                                    halt=dd_halt, floor=dd_floor)
        else:
            m = 1.0
        wt = w0.iloc[t] * m
        tr = float((wt - prev).abs().sum())
        gross = float((wt * fr.iloc[t]).sum())
        r = gross - tr * cost
        net.append(r)
        turn.append(tr)
        actual.append(wt)
        eq *= (1.0 + r)
        if eq > peak:
            peak = eq
        prev = wt

    net = pd.Series(net, index=w0.index)
    turn = pd.Series(turn, index=w0.index)
    actual = pd.DataFrame(actual, index=w0.index, columns=w0.columns)
    return actual, net, turn


def run_buggy_reference(close, fwd, assets, cost=COST):
    """The old, look-ahead version. Kept only to quantify the error."""
    weights = TrendVolSignal().generate_weights(close[assets]).fillna(0.0)
    dates = weights.index.intersection(fwd.dropna(how='all').index)
    w = weights.loc[dates]
    fr = fwd.loc[dates].fillna(0.0)
    eq = (1 + (w * fr).sum(axis=1)).cumprod()
    dd = (eq.cummax() - eq) / eq.cummax()
    mult = dd.apply(lambda d: drawdown_multiplier(1.0, 1.0 - d))
    w = w.mul(mult, axis=0)
    tr = w.diff().abs().sum(axis=1)
    tr.iloc[0] = w.iloc[0].abs().sum()
    net = (w * fr).sum(axis=1) - tr * cost
    return w, net, tr


def slice_s(x, start, end):
    return x[(x.index >= start) & (x.index < end)]


def row(name, rets, bench, start, end):
    r = slice_s(rets, start, end)
    m = metrics(r)
    b, a = beta_alpha(r, slice_s(bench, start, end))
    print(f"{name:<34}{m['total']*100:>9.1f}{m['ann']*100:>8.1f}{m['vol']*100:>8.1f}"
          f"{m['sharpe']:>8.2f}{m['mdd']*100:>8.1f}{m['hit']*100:>8.1f}"
          f"{b:>7.2f}{a*100:>9.1f}")


def block_bootstrap_tstat(rets, block=20, n_boot=2000, seed=7):
    r = rets.dropna().values
    n = len(r)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = []
        while len(idx) < n:
            start = rng.integers(0, n)
            idx.extend([(start + k) % n for k in range(block)])
        means[b] = r[np.array(idx[:n])].mean()
    se = means.std(ddof=1)
    return r.mean(), se, r.mean() / se if se else np.nan, np.percentile(means, [2.5, 97.5])


def main():
    c15 = first_part('research_data.pkl')
    c44 = first_part('broad_data_delisted.pkl')
    fwd15 = c15.shift(-1) / c15 - 1.0
    fwd44 = c44.shift(-1) / c44 - 1.0
    btc15 = fwd15['BTC-USD']

    print(f"15 survivors: {c15.shape[1]} assets, {len(c15)} bars, "
          f"{c15.index[0].date()} -> {c15.index[-1].date()}")
    print(f"44 incl. collapsed: {c44.shape[1]} assets, {len(c44)} bars, "
          f"{c44.index[0].date()} -> {c44.index[-1].date()}")

    hdr = (f"{'window / variant':<34}{'total%':>9}{'CAGR%':>8}{'vol%':>8}{'Sharpe':>8}"
           f"{'MDD%':>8}{'hit%':>8}{'beta':>7}{'alpha%':>9}")

    for label, s, e in [('train 2020-11..2023-12', TRAIN_START, TRAIN_END),
                        ('test  2024-01..2026-11', TEST_START, TEST_END),
                        ('full  2020-11..2026-11', TRAIN_START, TEST_END)]:
        print('\n' + '=' * 104)
        print(label)
        print('=' * 104)
        print(hdr)
        print('-' * 104)
        _, n_nov, _ = run_causal(c15, fwd15, U15, overlay=False)
        _, n_c15, _ = run_causal(c15, fwd15, U15, overlay=True)
        _, n_c44, _ = run_causal(c44, fwd44, list(c44.columns), overlay=True)
        _, n_bug, _ = run_buggy_reference(c15, fwd15, U15)
        _, n_b2, _ = run_causal(c15, fwd15, U2, overlay=False)
        _, n_c2, _ = run_causal(c15, fwd15, U2, overlay=True)
        row('BTC+ETH, no overlay', n_b2, btc15, s, e)
        row('BTC+ETH, causal overlay', n_c2, btc15, s, e)
        row('15 survivors, no overlay', n_nov, btc15, s, e)
        row('15 survivors, causal overlay', n_c15, btc15, s, e)
        row('44 incl. collapsed, causal', n_c44, btc15, s, e)
        row('15 survivors, OLD buggy overlay', n_bug, btc15, s, e)
        row('BTC buy & hold', btc15, btc15, s, e)

    # Summary + significance
    _, n_c15, tr15 = run_causal(c15, fwd15, U15, overlay=True)
    full = metrics(slice_s(n_c15, TRAIN_START, TEST_END))
    print('\n' + '=' * 104)
    print('CORRECTED FULL-PERIOD SUMMARY (15 survivors, causal overlay)')
    print('=' * 104)
    print(f"  Total return:   {full['total']*100:.1f}%")
    print(f"  CAGR:           {full['ann']*100:.1f}%")
    print(f"  Sharpe:         {full['sharpe']:.2f}")
    print(f"  Max DD:         {full['mdd']*100:.1f}%")
    print(f"  Hit rate:       {full['hit']*100:.1f}%")
    print(f"  Avg turnover:   {tr15.mean()*100:.1f}% per bar")
    print(f"  Cost drag/yr:   {tr15.mean()*365*COST*100:.1f}%")

    print('\n  Block-bootstrap t-stat (autocorrelation-robust):')
    for block in [1, 10, 20, 30]:
        mean_t, se, t, ci = block_bootstrap_tstat(n_c15, block=block)
        print(f"    block={block:>2}  daily mean={mean_t*1e4:6.2f}bp  "
              f"t={t:5.2f}  95% CI=[{ci[0]*1e4:6.2f}, {ci[1]*1e4:6.2f}] bp")

    print('\n  Cost sensitivity (causal, 15 survivors):')
    for bps in [6.4, 10, 15, 30, 50]:
        _, n, _ = run_causal(c15, fwd15, U15, cost=bps / 10000.0, overlay=True)
        m = metrics(slice_s(n, TRAIN_START, TEST_END))
        print(f"    {bps:>4.1f} bps: CAGR {m['ann']*100:5.1f}%  "
              f"Sharpe {m['sharpe']:4.2f}  MDD {m['mdd']*100:5.1f}%")


if __name__ == '__main__':
    main()
