"""
Causal signal research for daily crypto.

Goals:
  * No look-ahead: features at bar t use data up to t; they earn t -> t+1 return.
  * Train/test split + walk-forward, so reported numbers are out-of-sample.
  * Net of realistic costs (6.4 bps per unit turnover).
  * Compare classic crypto signals and the existing LiquidityAlpha.

Run:  python research_signal.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from neural_flow.alpha.model import LiquidityAlpha

TICKERS = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD', 'DOT-USD',
    'LINK-USD', 'LTC-USD', 'DOGE-USD', 'AVAX-USD', 'ATOM-USD', 'NEAR-USD',
    'ALGO-USD', 'FIL-USD',
]
TEST_START = pd.Timestamp('2024-01-01')
COST = 0.0005 + 0.0002 * 0.7  # StealthExecution fee + 70% of slippage
ANN = 365


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data(period='6y'):
    raw = yf.download(TICKERS, period=period, interval='1d', progress=False,
                      threads=True, auto_adjust=True)
    close = raw['Close'].ffill()
    high = raw['High'].ffill()
    low = raw['Low'].ffill()
    volume = raw['Volume'].ffill()

    # Keep assets with a long enough history (data by mid-2021, >90% coverage).
    first = close.apply(lambda s: s.first_valid_index())
    valid = [c for c in close.columns
             if first.get(c) is not None and first[c] <= pd.Timestamp('2021-06-01')
             and close[c].loc['2021-06-01':].notna().mean() > 0.90]
    close, high, low, volume = (x[valid] for x in (close, high, low, volume))

    start = close.apply(lambda s: s.first_valid_index()).max()
    close = close.loc[start:].ffill()
    high = high.loc[start:].ffill()
    low = low.loc[start:].ffill()
    volume = volume.loc[start:].ffill()
    return close, high, low, volume


# --------------------------------------------------------------------------- #
# Features (all causal)
# --------------------------------------------------------------------------- #
def _rsi(close, n=14):
    d = close.diff()
    gain = d.where(d > 0, 0).rolling(n).mean()
    loss = (-d.where(d < 0, 0)).rolling(n).mean()
    rs = gain / (loss + 1e-12)
    return 100 - 100 / (1 + rs)


def build_features(close, high, low, volume):
    ret = close.pct_change()
    logv = np.log(volume.replace(0, np.nan) + 1.0)
    f = {}
    f['mom_5'] = close.pct_change(5)
    f['mom_20'] = close.pct_change(20)
    f['mom_60'] = close.pct_change(60)
    f['mom_120'] = close.pct_change(120)
    f['rev_1'] = -ret
    f['rev_5'] = -close.pct_change(5)
    f['vol_20'] = ret.rolling(20).std()
    f['vol_60'] = ret.rolling(60).std()
    f['ma_gap_20'] = close / close.rolling(20).mean() - 1
    f['ma_gap_50'] = close / close.rolling(50).mean() - 1
    f['dist_high_60'] = close / close.rolling(60).max() - 1
    f['rsi_14'] = _rsi(close, 14)
    f['vol_z_20'] = (logv - logv.rolling(20).mean()) / (logv.rolling(20).std() + 1e-12)
    f['dollar_vol_20'] = np.log((close * volume).rolling(20).mean() + 1.0)
    f['range_20'] = ((high - low) / close).rolling(20).mean()
    f['skew_20'] = ret.rolling(20).skew()
    return f


# --------------------------------------------------------------------------- #
# Cross-sectional helpers
# --------------------------------------------------------------------------- #
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
    vals = pd.Series(vals)
    if len(vals) < 20:
        return np.nan, np.nan, len(vals)
    return vals.mean(), vals.mean() / (vals.std() + 1e-12) * np.sqrt(len(vals)), len(vals)


# --------------------------------------------------------------------------- #
# Portfolio construction
# --------------------------------------------------------------------------- #
def run_portfolio(signal, fwd, start, end, rebalance=1, top_frac=1 / 3,
                  long_short=True, cost=COST):
    dates = signal.index.intersection(fwd.dropna(how='all').index)
    dates = dates[(dates >= start) & (dates < end)]
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


def metrics(rets):
    if len(rets) == 0:
        return dict(total=0, ann=0, vol=0, sharpe=0, mdd=0, hit=0)
    total = float((1 + rets).prod() - 1)
    ann = (1 + total) ** (ANN / len(rets)) - 1
    vol = float(rets.std() * np.sqrt(ANN))
    sharpe = ann / vol if vol else 0.0
    eq = (1 + rets).cumprod()
    mdd = float((1 - eq / eq.cummax()).max())
    return dict(total=total, ann=ann, vol=vol, sharpe=sharpe, mdd=mdd,
                hit=float((rets > 0).mean()))


def fmt(name, m, turn=None):
    tail = f"  turn={turn*100:5.1f}%/bar" if turn is not None else ""
    return (f"{name:<26}{m['total']*100:>9.1f}{m['ann']*100:>9.1f}"
            f"{m['vol']*100:>8.1f}{m['sharpe']:>8.2f}{m['mdd']*100:>8.1f}"
            f"{m['hit']*100:>7.1f}{tail}")


# --------------------------------------------------------------------------- #
# Walk-forward ridge
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


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print('Loading daily crypto data (6y)...')
    close, high, low, volume = load_data('6y')
    fwd = close.shift(-1) / close - 1.0
    print(f'  assets: {len(close.columns)} | {close.index[0].date()} -> {close.index[-1].date()}'
          f' | {len(close)} bars')
    print(f'  tickers: {", ".join(close.columns)}')

    feats = build_features(close, high, low, volume)
    train_end = TEST_START
    test_start = TEST_START

    # ---- 1. Feature IC, train vs test ----
    rows = []
    for name, f in feats.items():
        it, tt, nt = ic_stats(f, fwd, close.index[0], train_end)
        io, to, no = ic_stats(f, fwd, test_start, close.index[-1] + pd.Timedelta(days=1))
        rows.append((name, it, tt, io, to))
    ic = pd.DataFrame(rows, columns=['feature', 'train_IC', 'train_t', 'test_IC', 'test_t'])
    ic['abs_test'] = ic['test_IC'].abs()
    print('\nFeature IC (Spearman, cross-sectional):')
    print(f"{'feature':<16}{'train IC':>10}{'train t':>9}{'test IC':>10}{'test t':>9}")
    for _, r in ic.sort_values('abs_test', ascending=False).iterrows():
        print(f"{r['feature']:<16}{r['train_IC']:>10.4f}{r['train_t']:>9.2f}"
              f"{r['test_IC']:>10.4f}{r['test_t']:>9.2f}")

    # ---- 2. Composite: pick features/signs on TRAIN only ----
    sel = ic[(ic['train_t'].abs() >= 1.5) & ic['train_IC'].notna()]
    print(f"\nTrain-selected features ({len(sel)}): "
          + ', '.join(f"{r.feature}({'+' if r.train_IC > 0 else '-'})" for r in sel.itertuples()))
    if len(sel):
        comp = sum(np.sign(r.train_IC) * cs_z(feats[r.feature]) for r in sel.itertuples())
        comp = comp / len(sel)
    else:
        comp = pd.DataFrame(0.0, index=close.index, columns=close.columns)

    # ---- 3. Current LiquidityAlpha ----
    la = LiquidityAlpha()
    cur_alpha = pd.DataFrame({
        c: la.generate_signal(pd.DataFrame({'Close': close[c], 'Volume': volume[c]}))['signal']
        for c in close.columns
    })

    # ---- 4. Walk-forward ridge ----
    print('\nRunning walk-forward ridge (refit monthly, 2y window)...')
    ridge = walk_forward_ridge(feats, fwd)

    # ---- 5. Portfolios on TEST period ----
    print('\n' + '=' * 92)
    print('OUT-OF-SAMPLE PORTFOLIOS (test from 2024-01-01, net of 6.4 bps turnover)')
    print('=' * 92)
    print(f"{'strategy':<26}{'total%':>9}{'CAGR%':>9}{'vol%':>8}{'Sharpe':>8}{'MDD%':>8}{'hit%':>7}")

    lines = []
    strategies = [
        ('BTC buy & hold', fwd[['BTC-USD']], 1, False),
        ('equal-weight universe', fwd.mean(axis=1).to_frame('ew'), 1, False),
        ('current LiquidityAlpha', cur_alpha, 7, True),
        ('XSMOM 20d (LS)', feats['mom_20'], 7, True),
        ('XSMOM 60d (LS)', feats['mom_60'], 7, True),
        ('XSMOM 120d (LS)', feats['mom_120'], 7, True),
        ('short-term reversal (LS)', feats['rev_1'], 1, True),
        ('low-vol 20d (LS)', -feats['vol_20'], 7, True),
        ('trend/MA gap 50 (LS)', feats['ma_gap_50'], 7, True),
        ('composite (train-selected)', comp, 7, True),
        ('ridge walk-forward (LS)', ridge, 7, True),
        ('XSMOM 20d long-only', feats['mom_20'], 7, False),
        ('low-vol long-only', -feats['vol_20'], 7, False),
        ('BTC trend filter long-only', feats['ma_gap_50'], 7, False),
    ]
    for name, sig, reb, ls in strategies:
        g, n, turn = run_portfolio(sig, fwd, test_start, close.index[-1] + pd.Timedelta(days=1),
                                   rebalance=reb, long_short=ls)
        m = metrics(n)
        mg = metrics(g)
        line = fmt(name, m, turn)
        print(line + f"   gross Sharpe={mg['sharpe']:.2f}")
        lines.append((name, m, mg, turn))

    # ---- 6. Composite/ridge IC on test ----
    print('\nTest-period IC:')
    for name, sig in [('composite', comp), ('ridge', ridge), ('current alpha', cur_alpha)]:
        mi, ti, n = ic_stats(sig, fwd, test_start, close.index[-1] + pd.Timedelta(days=1))
        print(f"  {name:<16} IC={mi:+.4f}  t={ti:+.2f}  (n={n})")

    print('\nNotes: LS = long/short top/bottom third, gross 1.0. '
          'Rebalance = 1d or 7d. Crypto annualization uses 365 days.')


if __name__ == '__main__':
    main()
