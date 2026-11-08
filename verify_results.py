"""
Independent audit of the TrendVol backtest.

Checks, in order:
  1. Data integrity: fake/backfilled history, flat-price tails, NaN.
  2. Signal causality: truncation test (no look-ahead in weights).
  3. Drawdown-overlay look-ahead: buggy vs lagged vs fully recursive.
  4. Survivorship bias: fixed 15 survivors vs 40-asset vs 44-asset incl. delisted.
  5. Cost sensitivity: 6.4 / 15 / 30 / 50 bps per unit turnover.
  6. Autocorrelation-robust significance: block bootstrap.

Run: python verify_results.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from neural_flow.alpha.trend import TrendVolSignal
from neural_flow.risk.model import drawdown_multiplier

ANN = 365
COST = 0.0005 + 0.0002 * 0.7  # 6.4 bps

U15 = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD', 'DOT-USD',
       'LINK-USD', 'LTC-USD', 'DOGE-USD', 'AVAX-USD', 'ATOM-USD', 'NEAR-USD',
       'ALGO-USD', 'FIL-USD']
DELISTED = ['LUNC-USD', 'USTC-USD', 'FTT-USD', 'CEL-USD']


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


# ----------------------------------------------------------------------------
# 1. DATA INTEGRITY
# ----------------------------------------------------------------------------
def data_audit():
    print("=" * 96)
    print("1. DATA INTEGRITY")
    print("=" * 96)
    for name in ['research_data.pkl', 'trend_data_7y.pkl', 'trend_data_7y_clean.pkl',
                 'broad_data.pkl', 'broad_data_delisted.pkl']:
        c = first_part(name)
        rets = c.pct_change()
        flat = (rets.abs() < 1e-12).mean()
        # Longest run of zero returns per column
        longest = {}
        for col in c.columns:
            z = (rets[col].abs() < 1e-12).astype(int)
            if len(z) == 0:
                continue
            grp = (z != z.shift()).cumsum()
            longest[col] = int(z.groupby(grp).sum().max())
        worst = sorted(longest.items(), key=lambda kv: kv[1], reverse=True)[:5]
        print(f"\n{name}: {c.shape[0]} bars x {c.shape[1]} cols | {c.index[0].date()} -> {c.index[-1].date()}")
        print(f"  mean flat-return fraction: {flat.mean():.3f}")
        print(f"  longest flat run (days): {worst}")
        print(f"  FIL first valid: {c['FIL-USD'].first_valid_index() if 'FIL-USD' in c else 'n/a'}")
        print(f"  total NaN: {int(c.isna().sum().sum())}")


# ----------------------------------------------------------------------------
# 2. SIGNAL CAUSALITY
# ----------------------------------------------------------------------------
def causality_test():
    print("\n" + "=" * 96)
    print("2. SIGNAL CAUSALITY (truncation test)")
    print("=" * 96)
    c = first_part('research_data.pkl')[U15].copy()
    sig = TrendVolSignal()
    full = sig.generate_weights(c)
    tests = [pd.Timestamp(x) for x in ['2022-06-15', '2024-03-10', '2025-12-01']]
    ok = True
    for t in tests:
        trunc = sig.generate_weights(c.loc[:t]).iloc[-1]
        ref = full.loc[t]
        diff = float((trunc - ref).abs().max())
        status = 'OK' if diff < 1e-9 else 'LOOK-AHEAD!'
        ok &= diff < 1e-9
        print(f"  {t.date()}: max |w_trunc - w_full| = {diff:.2e}  {status}")
    print(f"  RESULT: {'causal' if ok else 'NON-CAUSAL'}")


# ----------------------------------------------------------------------------
# 3. OVERLAY LOOK-AHEAD
# ----------------------------------------------------------------------------
def _signal_weights(close, assets):
    assets = [a for a in assets if a in close.columns]
    return TrendVolSignal().generate_weights(close[assets])


def overlay_variants(close, fwd, assets, cost=COST):
    w0 = _signal_weights(close, assets)
    dates = w0.index.intersection(fwd.dropna(how='all').index)
    w0 = w0.loc[dates]
    fr = fwd.loc[dates]

    # --- buggy: multiplier at t uses equity that includes return t->t+1 ---
    pre = (w0 * fr).sum(axis=1)
    eq = (1 + pre).cumprod()
    dd = (eq.cummax() - eq) / eq.cummax()
    mult_buggy = dd.apply(lambda d: drawdown_multiplier(1.0, 1.0 - d))
    wb = w0.mul(mult_buggy, axis=0)
    tb = wb.diff().abs().sum(axis=1); tb.iloc[0] = wb.iloc[0].abs().sum()
    net_buggy = (wb * fr).sum(axis=1) - tb * cost

    # --- lagged fix: multiplier at t uses dd through t-1 ---
    mult_lag = mult_buggy.shift(1).fillna(1.0)
    wl = w0.mul(mult_lag, axis=0)
    tl = wl.diff().abs().sum(axis=1); tl.iloc[0] = wl.iloc[0].abs().sum()
    net_lag = (wl * fr).sum(axis=1) - tl * cost

    # --- fully recursive, causal loop (overlay changes its own equity) ---
    net_rec = []
    eqv, peak, prev_w = 1.0, 1.0, pd.Series(0.0, index=w0.columns)
    for t in range(len(w0)):
        d = (peak - eqv) / peak
        m = drawdown_multiplier(1.0, 1.0 - d)
        wt = w0.iloc[t] * m
        turn = float((wt - prev_w).abs().sum())
        gross = float((wt * fr.iloc[t]).sum())
        r = gross - turn * cost
        net_rec.append(r)
        eqv *= (1 + r)
        peak = max(peak, eqv)
        prev_w = wt
    net_rec = pd.Series(net_rec, index=w0.index)
    return net_buggy, net_lag, net_rec


def overlay_audit():
    print("\n" + "=" * 96)
    print("3. DRAWDOWN-OVERLAY LOOK-AHEAD")
    print("=" * 96)
    c = first_part('research_data.pkl')
    fwd = c.shift(-1) / c - 1.0
    for assets, tag in [(U15, '15 survivors')]:
        nb, nl, nr = overlay_variants(c, fwd, assets)
        print(f"\n{tag}:")
        print(f"  {'variant':<26}{'CAGR%':>8}{'Sharpe':>8}{'MDD%':>8}{'total%':>9}")
        for name, s in [('buggy (look-ahead)', nb), ('lagged fix', nl), ('recursive causal', nr)]:
            m = metrics(s)
            print(f"  {name:<26}{m['ann']*100:>8.1f}{m['sharpe']:>8.2f}{m['mdd']*100:>8.1f}{m['total']*100:>9.1f}")


# ----------------------------------------------------------------------------
# 4. SURVIVORSHIP BIAS
# ----------------------------------------------------------------------------
def survivorship_audit():
    print("\n" + "=" * 96)
    print("4. SURVIVORSHIP BIAS")
    print("=" * 96)
    sets = []

    c15 = first_part('research_data.pkl')
    sets.append(('15 fixed survivors (used in report)', c15, U15))

    c40 = first_part('broad_data.pkl')
    sets.append(('40 assets, survivors only', c40, [a for a in c40.columns]))

    c44 = first_part('broad_data_delisted.pkl')
    sets.append(('44 assets incl. LUNC/USTC/FTT/CEL', c44, [a for a in c44.columns]))

    print(f"\n{'universe':<40}{'CAGR%':>8}{'Sharpe':>8}{'MDD%':>8}{'vol%':>8}")
    for name, c, assets in sets:
        fwd = c.shift(-1) / c - 1.0
        _, _, nr = overlay_variants(c, fwd, assets)
        m = metrics(nr)
        print(f"{name:<40}{m['ann']*100:>8.1f}{m['sharpe']:>8.2f}{m['mdd']*100:>8.1f}{m['vol']*100:>8.1f}")

    # Explicit contribution of the collapsed coins
    print("\nDelisted-asset behaviour (are they ffill'd flat after collapse?):")
    for a in DELISTED:
        if a not in c44:
            continue
        s = c44[a]
        last_valid = s.last_valid_index()
        tail = s.loc[last_valid - pd.Timedelta(days=10):]
        print(f"  {a:<10} last_valid={last_valid.date()}  "
              f"last_return_10d={(tail.iloc[-1]/tail.iloc[0]-1)*100:6.1f}%  "
              f"flat_days_after={int((s.loc[last_valid:].pct_change().abs()<1e-12).sum())}")


# ----------------------------------------------------------------------------
# 5. COST SENSITIVITY
# ----------------------------------------------------------------------------
def cost_audit():
    print("\n" + "=" * 96)
    print("5. COST SENSITIVITY (recursive causal overlay)")
    print("=" * 96)
    c = first_part('research_data.pkl')
    fwd = c.shift(-1) / c - 1.0
    print(f"\n{'cost/unit turnover':<24}{'CAGR%':>8}{'Sharpe':>8}{'MDD%':>8}")
    for bps in [6.4, 10, 15, 30, 50]:
        _, _, nr = overlay_variants(c, fwd, U15, cost=bps / 10000.0)
        m = metrics(nr)
        print(f"{str(bps)+' bps':<24}{m['ann']*100:>8.1f}{m['sharpe']:>8.2f}{m['mdd']*100:>8.1f}")


# ----------------------------------------------------------------------------
# 6. BLOCK BOOTSTRAP
# ----------------------------------------------------------------------------
def block_bootstrap_tstat(rets, block=20, n_boot=2000, seed=7):
    r = rets.dropna().values
    n = len(r)
    mean_t = r.mean()
    # stationary block bootstrap of the mean
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = []
        while len(idx) < n:
            start = rng.integers(0, n)
            idx.extend([(start + k) % n for k in range(block)])
        means[b] = r[np.array(idx[:n])].mean()
    se = means.std(ddof=1)
    return mean_t, se, mean_t / se if se else np.nan, np.percentile(means, [2.5, 97.5])


def bootstrap_audit():
    print("\n" + "=" * 96)
    print("6. AUTOCORRELATION-ROBUST SIGNIFICANCE (stationary block bootstrap)")
    print("=" * 96)
    c = first_part('research_data.pkl')
    fwd = c.shift(-1) / c - 1.0
    _, _, nr = overlay_variants(c, fwd, U15)
    for block in [1, 10, 20, 30]:
        mean_t, se, t, ci = block_bootstrap_tstat(nr, block=block)
        print(f"  block={block:>2}  daily mean={mean_t*1e4:6.2f}bp  "
              f"t={t:5.2f}  95% CI of daily mean=[{ci[0]*1e4:6.2f}, {ci[1]*1e4:6.2f}] bp")


if __name__ == '__main__':
    data_audit()
    causality_test()
    overlay_audit()
    survivorship_audit()
    cost_audit()
    bootstrap_audit()
