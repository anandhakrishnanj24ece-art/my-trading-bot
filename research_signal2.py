"""
Causal signal research v2 for daily crypto.

Fixes vs v1:
  * Correct BTC / equal-weight benchmarks.
  * Adds factor-weighted (all-asset) portfolios, which align better with IC
    than top/bottom-tercile portfolios.
  * Adds beta / alpha decomposition vs the equal-weight market.
  * Adds a focused train-selected composite and a BTC time-series trend.
  * No look-ahead: features at t earn t -> t+1 returns.

Run:  python research_signal2.py
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from neural_flow.alpha.model import LiquidityAlpha

warnings.filterwarnings('ignore')

CACHE = Path('research_data.pkl')

TICKERS = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD', 'DOT-USD',
    'LINK-USD', 'LTC-USD', 'DOGE-USD', 'AVAX-USD', 'ATOM-USD', 'NEAR-USD',
    'ALGO-USD', 'FIL-USD',
]
TEST_START = pd.Timestamp('2024-01-01')
COST = 0.0005 + 0.0002 * 0.7
ANN = 365


# --------------------------------------------------------------------------- #
def load_data(period='6y', use_cache=True):
    if use_cache and CACHE.exists():
        return pd.read_pickle(CACHE)

    last_err = None
    for attempt in range(5):
        try:
            raw = yf.download(TICKERS, period=period, interval='1d', progress=False,
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
        except Exception as e:
            last_err = e
            time.sleep(3)
    raise RuntimeError(f'data download failed after retries: {last_err}')


def _rsi(close, n=14):
    d = close.diff()
    gain = d.where(d > 0, 0).rolling(n).mean()
    loss = (-d.where(d < 0, 0)).rolling(n).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-12))


def build_features(close, high, low, volume):
    ret = close.pct_change()
    logv = np.log(volume.replace(0, np.nan) + 1.0)
    f = {
        'mom_5': close.pct_change(5),
        'mom_20': close.pct_change(20),
        'mom_60': close.pct_change(60),
        'mom_120': close.pct_change(120),
        'rev_1': -ret,
        'vol_20': ret.rolling(20).std(),
        'vol_60': ret.rolling(60).std(),
        'ma_gap_20': close / close.rolling(20).mean() - 1,
        'ma_gap_50': close / close.rolling(50).mean() - 1,
        'dist_high_60': close / close.rolling(60).max() - 1,
        'rsi_14': _rsi(close, 14),
        'vol_z_20': (logv - logv.rolling(20).mean()) / (logv.rolling(20).std() + 1e-12),
        'dollar_vol_20': np.log((close * volume).rolling(20).mean() + 1.0),
        'range_20': ((high - low) / close).rolling(20).mean(),
        'skew_20': ret.rolling(20).skew(),
    }
    return f


def cs_z(df):
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def ic_stats(feat, fwd, start, end, min_assets=6):
    dates = feat.index.intersection(fwd.index)
    dates = dates[(dates >= start) & (dates < end)]
    vals = []
    for d in dates:
        x, y = feat.loc[d], fwd.loc[d]
        m = x.notna() & y.notna()
        if m.sum() >= min_assets:
            c = spearmanr(x[m], y[m]).correlation
            if np.isfinite(c):
                vals.append(c)
    v = pd.Series(vals)
    if len(v) < 20:
        return np.nan, np.nan, len(v)
    return v.mean(), v.mean() / (v.std() + 1e-12) * np.sqrt(len(v)), len(v)


# --------------------------------------------------------------------------- #
def _dates(signal, fwd, start, end):
    d = signal.index.intersection(fwd.dropna(how='all').index)
    return d[(d >= start) & (d < end)]


def run_portfolio(signal, fwd, start, end, rebalance=1, top_frac=1 / 3,
                  long_short=True, cost=COST):
    dates = _dates(signal, fwd, start, end)
    sig, fr = signal.loc[dates], fwd.loc[dates]
    cols = fr.columns
    cur = pd.Series(0.0, index=cols)
    gross, net, turns = [], [], []
    reb = set(dates[::rebalance])
    for d in dates:
        turn = 0.0
        if d in reb:
            s = sig.loc[d].dropna()
            if len(s) >= 6:
                n = max(1, int(round(len(s) * top_frac)))
                order = s.sort_values()
                new = pd.Series(0.0, index=cols)
                if long_short:
                    new[order.index[-n:]] = 0.5 / n
                    new[order.index[:n]] = -0.5 / n
                else:
                    new[order.index[-n:]] = 1.0 / n
                turn = float((new - cur).abs().sum())
                cur = new
        r = float((cur * fr.loc[d]).sum())
        gross.append(r)
        net.append(r - turn * cost)
        turns.append(turn)
    return (pd.Series(gross, index=dates), pd.Series(net, index=dates),
            float(np.mean(turns)) if turns else 0.0)


def run_weighted(signal, fwd, start, end, rebalance=7, cost=COST):
    """Factor portfolio: weights proportional to cross-sectional signal,
    demeaned (market-neutral), gross exposure 1.0."""
    dates = _dates(signal, fwd, start, end)
    sig, fr = signal.loc[dates], fwd.loc[dates]
    cols = fr.columns
    cur = pd.Series(0.0, index=cols)
    gross, net, turns = [], [], []
    reb = set(dates[::rebalance])
    for d in dates:
        turn = 0.0
        if d in reb:
            s = sig.loc[d].dropna()
            if len(s) >= 6:
                w = s - s.mean()
                g = w.abs().sum()
                new = pd.Series(0.0, index=cols)
                if g > 0:
                    new[w.index] = w / g
                turn = float((new - cur).abs().sum())
                cur = new
        r = float((cur * fr.loc[d]).sum())
        gross.append(r)
        net.append(r - turn * cost)
        turns.append(turn)
    return (pd.Series(gross, index=dates), pd.Series(net, index=dates),
            float(np.mean(turns)) if turns else 0.0)


def run_weights(weights, fwd, start, end, cost=COST):
    dates = _dates(weights, fwd, start, end)
    w = weights.loc[dates].fillna(0.0)
    fr = fwd.loc[dates]
    turn = w.diff().abs().sum(axis=1)
    turn.iloc[0] = w.iloc[0].abs().sum()
    gross = (w * fr).sum(axis=1)
    net = gross - turn * cost
    return gross, net, float(turn.mean())


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


# --------------------------------------------------------------------------- #
def walk_forward_ridge(feats, fwd, train_days=730, refit=30, alpha=10.0):
    names = list(feats)
    z = {n: cs_z(feats[n]) for n in names}
    dates, cols = fwd.index, fwd.columns
    X = np.stack([z[n].reindex(index=dates, columns=cols).values for n in names], axis=-1)
    Y = fwd.sub(fwd.mean(axis=1), axis=0).values
    pred = pd.DataFrame(np.nan, index=dates, columns=cols)
    for t in range(train_days, len(dates), refit):
        Xtr = X[t - train_days:t].reshape(-1, len(names))
        Ytr = Y[t - train_days:t].reshape(-1)
        m = np.isfinite(Xtr).all(axis=1) & np.isfinite(Ytr)
        if m.sum() < 200:
            continue
        model = Ridge(alpha=alpha).fit(Xtr[m], Ytr[m])
        for j in range(t, min(t + refit, len(dates))):
            Xp = X[j]
            mm = np.isfinite(Xp).all(axis=1)
            if mm.sum() >= 6:
                p = np.full(len(cols), np.nan)
                p[mm] = model.predict(Xp[mm])
                pred.iloc[j] = p
    return pred


def composite(sel_df, feats, drop=()):
    parts = []
    for r in sel_df.itertuples():
        if r.feature in drop:
            continue
        parts.append(np.sign(r.train_IC) * cs_z(feats[r.feature]))
    if not parts:
        return pd.DataFrame(0.0, index=next(iter(feats.values())).index,
                            columns=next(iter(feats.values())).columns)
    return sum(parts) / len(parts)


# --------------------------------------------------------------------------- #
def main():
    print('Loading daily crypto data (6y)...')
    close, high, low, volume = load_data('6y')
    fwd = close.shift(-1) / close - 1.0
    end = close.index[-1] + pd.Timedelta(days=1)
    print(f'  {len(close.columns)} assets | {close.index[0].date()} -> {close.index[-1].date()}'
          f' | {len(close)} bars')

    feats = build_features(close, high, low, volume)

    # ---- feature IC (train = before 2024, test = 2024+) ----
    rows = []
    for name, f in feats.items():
        it, tt, _ = ic_stats(f, fwd, close.index[0], TEST_START)
        io, to, _ = ic_stats(f, fwd, TEST_START, end)
        rows.append((name, it, tt, io, to))
    ic = pd.DataFrame(rows, columns=['feature', 'train_IC', 'train_t', 'test_IC', 'test_t'])
    print('\nFeature IC (cross-sectional Spearman):')
    print(f"{'feature':<16}{'train IC':>10}{'train t':>9}{'test IC':>10}{'test t':>9}")
    for _, r in ic.reindex(ic['test_IC'].abs().sort_values(ascending=False).index).iterrows():
        print(f"{r['feature']:<16}{r['train_IC']:>10.4f}{r['train_t']:>9.2f}"
              f"{r['test_IC']:>10.4f}{r['test_t']:>9.2f}")

    sel15 = ic[(ic['train_t'].abs() >= 1.5) & ic['train_IC'].notna()]
    sel30 = ic[(ic['train_t'].abs() >= 3.0) & ic['train_IC'].notna()]
    comp15 = composite(sel15, feats, drop=('rev_5', 'mom_20', 'ma_gap_50'))
    comp30 = composite(sel30, feats, drop=('rev_5',))
    print(f"\nComposite A (|train t|>=1.5, dedup): "
          + ', '.join(f"{r.feature}{'+' if r.train_IC > 0 else '-'}" for r in sel15.itertuples()
                      if r.feature not in ('rev_5', 'mom_20', 'ma_gap_50')))
    print(f"Composite B (|train t|>=3.0, dedup): "
          + ', '.join(f"{r.feature}{'+' if r.train_IC > 0 else '-'}" for r in sel30.itertuples()
                      if r.feature != 'rev_5'))

    # ---- existing alpha ----
    la = LiquidityAlpha()
    cur_alpha = pd.DataFrame({
        c: la.generate_signal(pd.DataFrame({'Close': close[c], 'Volume': volume[c]}))['signal']
        for c in close.columns
    })

    # ---- walk-forward ridge ----
    print('\nRunning walk-forward ridge...')
    ridge = walk_forward_ridge(feats, fwd)

    # ---- BTC time-series trend ----
    btc = close['BTC-USD']
    btc_vol = btc.pct_change().rolling(20).std() * np.sqrt(ANN)
    btc_lev = (0.15 / (btc_vol + 1e-9)).clip(0, 2)
    btc_trend = ((btc > btc.rolling(50).mean()).astype(float) * btc_lev).to_frame('BTC-USD')
    btc_trend200 = ((btc > btc.rolling(200).mean()).astype(float) * btc_lev).to_frame('BTC-USD')

    # ---- benchmarks ----
    bench = fwd.mean(axis=1)
    btc_bh = fwd['BTC-USD']

    print('\n' + '=' * 102)
    print('OUT-OF-SAMPLE (test 2024-01-01 -> end, net of 6.4 bps turnover)')
    print('=' * 102)
    print(f"{'strategy':<30}{'total%':>8}{'CAGR%':>8}{'vol%':>7}{'Sharpe':>8}"
          f"{'MDD%':>7}{'hit%':>7}{'beta':>6}{'alpha%':>8}")
    print('-' * 102)

    def row(name, gross, net, bench_series=bench):
        m = metrics(net)
        b, a = beta_alpha(net, bench_series)
        print(f"{name:<30}{m['total']*100:>8.1f}{m['ann']*100:>8.1f}{m['vol']*100:>7.1f}"
              f"{m['sharpe']:>8.2f}{m['mdd']*100:>7.1f}{m['hit']*100:>7.1f}"
              f"{b:>6.2f}{a*100:>8.1f}")

    def slice_s(x):
        s = x[(x.index >= TEST_START) & (x.index < end)]
        return s

    # benchmarks
    m = metrics(slice_s(btc_bh))
    print(f"{'BTC buy & hold':<30}{m['total']*100:>8.1f}{m['ann']*100:>8.1f}{m['vol']*100:>7.1f}"
          f"{m['sharpe']:>8.2f}{m['mdd']*100:>7.1f}{m['hit']*100:>7.1f}{'-':>6}{'-':>8}")
    m = metrics(slice_s(bench))
    print(f"{'equal-weight universe':<30}{m['total']*100:>8.1f}{m['ann']*100:>8.1f}{m['vol']*100:>7.1f}"
          f"{m['sharpe']:>8.2f}{m['mdd']*100:>7.1f}{m['hit']*100:>7.1f}{'-':>6}{'-':>8}")
    print('-' * 102)

    tests = [
        ('current alpha  LS 7d', lambda: run_portfolio(cur_alpha, fwd, TEST_START, end, 7, long_short=True)),
        ('current alpha  weighted 7d', lambda: run_weighted(cur_alpha, fwd, TEST_START, end, 7)),
        ('composite A  weighted 1d', lambda: run_weighted(comp15, fwd, TEST_START, end, 1)),
        ('composite A  weighted 7d', lambda: run_weighted(comp15, fwd, TEST_START, end, 7)),
        ('composite A  LS 7d', lambda: run_portfolio(comp15, fwd, TEST_START, end, 7, long_short=True)),
        ('composite B  weighted 7d', lambda: run_weighted(comp30, fwd, TEST_START, end, 7)),
        ('composite B  LS 7d', lambda: run_portfolio(comp30, fwd, TEST_START, end, 7, long_short=True)),
        ('ridge  weighted 7d', lambda: run_weighted(ridge, fwd, TEST_START, end, 7)),
        ('BTC trend 50d  vol-target', lambda: run_weights(btc_trend, fwd, TEST_START, end)),
        ('BTC trend 200d vol-target', lambda: run_weights(btc_trend200, fwd, TEST_START, end)),
    ]
    for name, fn in tests:
        g, n, turn = fn()
        row(f"{name} [t={turn*100:.0f}%]", g, n)

    # ---- test IC of the composites ----
    print('\nTest-period IC:')
    for name, sig in [('composite A', comp15), ('composite B', comp30),
                      ('ridge', ridge), ('current alpha', cur_alpha)]:
        mi, ti, nn = ic_stats(sig, fwd, TEST_START, end)
        print(f'  {name:<16} IC={mi:+.4f}  t={ti:+.2f}  (n={nn})')

    print('\nLS = long/short bottom/top third, gross 1.0. Weighted = market-neutral,'
          ' weights proportional to signal, gross 1.0. [t=..] is avg turnover per bar.')


if __name__ == '__main__':
    main()
