"""
Stricter edge search on the broad, survivorship-inclusive crypto universe.

Protocol (fixed before looking at results):
  * Data: 44 assets incl. LUNC/USTC/FTT/CEL, daily close + volume, 2020-11 -> 2026-11.
  * All indicators are trailing (rolling windows), so weights at t use only data <= t.
  * Rebalance weekly; hold between rebalances; holdings drift with prices.
  * Costs: 10 bps per unit turnover (30 bps stress). Long-short pays 3 bp/day
    short financing. No look-ahead anywhere.
  * Split: train 2021-05-26 -> 2024-01-01; test 2024-01-01 -> 2026-11-07.
    The test window is not used to choose parameters.
  * Strategies are a small pre-specified set of known anomalies, not a grid.
  * Significance: circular block bootstrap (block 20) + Bonferroni across trials.

Run: python research_edge.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 365
COST_BASE = 0.0010      # 10 bps per unit turnover
COST_STRESS = 0.0030    # 30 bps
SHORT_COST_DAILY = 0.0003  # 3 bp/day on short gross
MIN_BARS = 200          # common start so every strategy has full history
REBAL = 7

TRAIN_START = pd.Timestamp('2021-05-26')
TRAIN_END = pd.Timestamp('2024-01-01')
TEST_START = pd.Timestamp('2024-01-01')
TEST_END = pd.Timestamp('2026-11-08')


# ---------------------------------------------------------------------------
# Data / indicators
# ---------------------------------------------------------------------------
def load_data():
    d = pd.read_pickle('broad_data_delisted.pkl')
    close, volume = d[0].ffill(), d[1].ffill()
    return close, volume


def make_indicators(close, volume):
    ret = close.pct_change()
    ind = {
        'close': close,
        'ret': ret,
        'vol20': ret.rolling(20).std() * np.sqrt(ANN),
        'vol60': ret.rolling(60).std() * np.sqrt(ANN),
        'sma50': close.rolling(50).mean(),
        'sma200': close.rolling(200).mean(),
        'mom7': close / close.shift(7) - 1.0,
        'mom30': close / close.shift(30) - 1.0,
        'mom90': close / close.shift(90) - 1.0,
        'mom180': close / close.shift(180) - 1.0,
        'hist': close.notna().cumsum(),
        'adv20': volume.rolling(20).mean(),
    }
    return ind


# ---------------------------------------------------------------------------
# Weight helpers (all causal at row t)
# ---------------------------------------------------------------------------
def eligible(ind, t, min_history=200):
    c, v = ind['close'].loc[t], ind['vol20'].loc[t]
    return ((ind['hist'].loc[t] >= min_history)
            & c.notna() & (c > 0)
            & v.notna() & (v > 0)
            & ind['adv20'].loc[t].fillna(0).gt(0))


def inv_vol_weights(ind, t, sel, target_vol=0.15, max_lev=2.0, max_weight=0.50):
    sel = [s for s in sel if np.isfinite(ind['vol20'].loc[t, s]) and ind['vol20'].loc[t, s] > 0]
    if not sel:
        return pd.Series(0.0, index=ind['close'].columns)
    v = ind['vol20'].loc[t, sel]
    raw = 1.0 / v
    w = raw / raw.sum()
    port_vol = float(np.sqrt(((w * v) ** 2).sum()))
    scale = min(target_vol / port_vol, max_lev) if port_vol > 0 else 1.0
    w = (w * scale).clip(upper=max_weight)
    return w.reindex(ind['close'].columns).fillna(0.0)


def equal_weights(ind, t, sel, target_vol=0.15, max_lev=1.5, max_weight=0.50):
    sel = [s for s in sel if np.isfinite(ind['vol20'].loc[t, s]) and ind['vol20'].loc[t, s] > 0]
    if not sel:
        return pd.Series(0.0, index=ind['close'].columns)
    v = ind['vol20'].loc[t, sel]
    w = pd.Series(1.0 / len(sel), index=sel)
    port_vol = float(np.sqrt(((w * v) ** 2).sum()))
    scale = min(target_vol / port_vol, max_lev) if port_vol > 0 else 1.0
    w = (w * scale).clip(upper=max_weight)
    return w.reindex(ind['close'].columns).fillna(0.0)


def ls_weights(ind, t, longs, shorts, gross_per_side=0.5, max_weight=0.25):
    def side(sel, sign):
        sel = [s for s in sel if np.isfinite(ind['vol20'].loc[t, s]) and ind['vol20'].loc[t, s] > 0]
        if not sel:
            return pd.Series(0.0, index=ind['close'].columns)
        v = ind['vol20'].loc[t, sel]
        raw = 1.0 / v
        w = raw / raw.sum() * gross_per_side * sign
        return w.clip(-max_weight, max_weight).reindex(ind['close'].columns).fillna(0.0)
    return side(longs, 1.0) + side(shorts, -1.0)


# ---------------------------------------------------------------------------
# Strategy factories: each returns (name, weight_func, short_book?)
# ---------------------------------------------------------------------------
def build_strategies(ind):
    cols = ind['close'].columns
    zero = pd.Series(0.0, index=cols)

    def tsmom(target_vol=0.15):
        def f(i):
            t = ind['close'].index[i]
            ok = eligible(ind, t)
            sel = ok.index[ok & (ind['close'].loc[t] > ind['sma50'].loc[t])]
            return inv_vol_weights(ind, t, sel, target_vol=target_vol)
        return f

    def xsmom(lookback='mom90', n=5, trend='sma50', target_vol=0.15):
        def f(i):
            t = ind['close'].index[i]
            ok = eligible(ind, t)
            cand = ok & (ind['close'].loc[t] > ind[trend].loc[t]) if trend else ok
            mom = ind[lookback].loc[t]
            ranked = mom[cand].dropna().sort_values(ascending=False)
            sel = ranked.head(n).index
            return inv_vol_weights(ind, t, sel, target_vol=target_vol)
        return f

    def xsmom_ls(lookback='mom90', n=5):
        def f(i):
            t = ind['close'].index[i]
            ok = eligible(ind, t)
            ranked = ind[lookback].loc[t][ok].dropna().sort_values()
            if len(ranked) < 2 * n:
                return zero
            shorts, longs = ranked.head(n).index, ranked.tail(n).index
            return ls_weights(ind, t, longs, shorts)
        return f

    def reversal(n=5, trend='sma200', target_vol=0.15):
        def f(i):
            t = ind['close'].index[i]
            ok = eligible(ind, t)
            cand = ok & (ind['close'].loc[t] > ind[trend].loc[t])
            ranked = ind['mom7'].loc[t][cand].dropna().sort_values()
            return inv_vol_weights(ind, t, ranked.head(n).index, target_vol=target_vol)
        return f

    def vol_managed_ew(target_vol=0.15):
        def f(i):
            t = ind['close'].index[i]
            ok = eligible(ind, t)
            return equal_weights(ind, t, ok.index[ok], target_vol=target_vol)
        return f

    def btc_eth(target_vol=0.15):
        def f(i):
            t = ind['close'].index[i]
            ok = eligible(ind, t)
            sel = [a for a in ['BTC-USD', 'ETH-USD']
                   if a in cols and ok.get(a, False) and ind['close'].loc[t, a] > ind['sma50'].loc[t, a]]
            return inv_vol_weights(ind, t, sel, target_vol=target_vol, max_weight=1.0)
        return f

    def btc_only(target_vol=0.15):
        def f(i):
            t = ind['close'].index[i]
            ok = eligible(ind, t)
            if (ok.get('BTC-USD', False)
                    and ind['close'].loc[t, 'BTC-USD'] > ind['sma50'].loc[t, 'BTC-USD']):
                return inv_vol_weights(ind, t, ['BTC-USD'], target_vol=target_vol, max_weight=1.0)
            return zero
        return f

    def xsmom_regime(n=5, target_vol=0.15):
        base = xsmom('mom90', n, 'sma50', target_vol)
        def f(i):
            t = ind['close'].index[i]
            btc_ok = (ind['close'].loc[t, 'BTC-USD'] > ind['sma200'].loc[t, 'BTC-USD'])
            return base(i) if btc_ok else zero
        return f

    return [
        ('TSMOM broad (50d)',        tsmom(),           False),
        ('XSMOM top5 (90d)',         xsmom(),           False),
        ('XSMOM top5 + BTC200',      xsmom_regime(),    False),
        ('XSMOM L/S top5',           xsmom_ls(),        True),
        ('Reversal 7d + 200d trend', reversal(),        False),
        ('Dual mom top5 (90/200)',   xsmom('mom90', 5, 'sma200'), False),
        ('Vol-managed equal weight', vol_managed_ew(),  False),
        ('BTC+ETH trend',            btc_eth(),         False),
        ('BTC-only trend',           btc_only(),        False),
    ]


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
def backtest(ind, weight_func, cost=COST_BASE, short_book=False, rebal=REBAL,
             min_bars=MIN_BARS):
    close = ind['close']
    dates = close.index
    rets = ind['ret'].fillna(0.0)
    cols = close.columns
    w = pd.Series(0.0, index=cols)
    nav = 1.0
    navs = []
    for i, t in enumerate(dates):
        r = rets.loc[t]
        short_gross = float(w.clip(upper=0).abs().sum())
        port_r = float((w * r).sum())
        if short_book:
            port_r -= short_gross * SHORT_COST_DAILY
        nav *= (1.0 + port_r)
        if 1.0 + port_r > 1e-12:
            w = w * (1.0 + r) / (1.0 + port_r)
        if i % rebal == 0 and i >= min_bars:
            target = weight_func(i)
            turnover = float((target - w).abs().sum())
            nav *= (1.0 - turnover * cost)
            w = target
        navs.append(nav)
    nav = pd.Series(navs, index=dates)
    daily = nav.pct_change().fillna(0.0)
    daily.iloc[:min_bars] = 0.0
    return daily, nav


# ---------------------------------------------------------------------------
# Metrics / significance
# ---------------------------------------------------------------------------
def metrics(rets):
    rets = rets.dropna()
    if len(rets) < 30:
        return dict(cagr=0.0, vol=0.0, sharpe=0.0, mdd=0.0, hit=0.0, total=0.0)
    total = float((1 + rets).prod() - 1)
    cagr = (1 + total) ** (ANN / len(rets)) - 1
    vol = float(rets.std() * np.sqrt(ANN))
    eq = (1 + rets).cumprod()
    return dict(total=total, cagr=cagr, vol=vol,
                sharpe=(cagr / vol if vol else 0.0),
                mdd=float((1 - eq / eq.cummax()).max()),
                hit=float((rets > 0).mean()))


def alpha_beta(rets, bench):
    df = pd.concat([rets, bench], axis=1).dropna()
    if len(df) < 30:
        return np.nan, np.nan
    cov = np.cov(df.iloc[:, 0], df.iloc[:, 1])
    beta = cov[0, 1] / cov[1, 1]
    alpha = (df.iloc[:, 0].mean() - beta * df.iloc[:, 1].mean()) * ANN
    return beta, alpha


def block_bootstrap(rets, block=20, n_boot=4000, seed=7):
    r = rets.dropna().values
    n = len(r)
    obs = r.mean()
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = []
        while len(idx) < n:
            start = rng.integers(0, n)
            idx.extend([(start + k) % n for k in range(block)])
        means[b] = r[np.array(idx[:n])].mean()
    se = means.std(ddof=1)
    p = float((means <= 0).mean())
    return obs, se, obs / se if se else np.nan, p


def slice_s(x, start, end):
    return x[(x.index >= start) & (x.index < end)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    close, volume = load_data()
    ind = make_indicators(close, volume)
    strategies = build_strategies(ind)
    btc = ind['ret']['BTC-USD']

    print(f"Universe: {close.shape[1]} assets | {len(close)} bars | "
          f"{close.index[0].date()} -> {close.index[-1].date()}")
    print(f"Train: {TRAIN_START.date()} -> {TRAIN_END.date()} | "
          f"Test: {TEST_START.date()} -> {TEST_END.date()}")
    print(f"Costs: {COST_BASE*1e4:.0f} bps/unit turnover (base), "
          f"{COST_STRESS*1e4:.0f} bps stress; short financing 3 bp/day\n")

    results = {}
    for name, fn, short_book in strategies:
        daily, _ = backtest(ind, fn, cost=COST_BASE, short_book=short_book)
        results[name] = dict(
            daily=daily,
            train=metrics(slice_s(daily, TRAIN_START, TRAIN_END)),
            test=metrics(slice_s(daily, TEST_START, TEST_END)),
            short=short_book,
        )

    btc_daily = btc.copy()
    results['BTC buy & hold'] = dict(
        daily=btc_daily,
        train=metrics(slice_s(btc_daily, TRAIN_START, TRAIN_END)),
        test=metrics(slice_s(btc_daily, TEST_START, TEST_END)),
        short=False,
    )

    hdr = (f"{'strategy':<30}{'trCAGR':>8}{'trSh':>6}{'trMDD':>7}   "
           f"{'teCAGR':>8}{'teSh':>6}{'teMDD':>7}{'teAlpha':>9}{'teP':>7}")
    print(hdr)
    print('-' * len(hdr))
    ordered = sorted(results.items(), key=lambda kv: kv[1]['train']['sharpe'], reverse=True)
    for name, r in ordered:
        tr, te = r['train'], r['test']
        beta, alpha = alpha_beta(slice_s(r['daily'], TEST_START, TEST_END),
                                 slice_s(btc_daily, TEST_START, TEST_END))
        _, _, _, p = block_bootstrap(slice_s(r['daily'], TEST_START, TEST_END))
        print(f"{name:<30}{tr['cagr']*100:>8.1f}{tr['sharpe']:>6.2f}{tr['mdd']*100:>7.1f}   "
              f"{te['cagr']*100:>8.1f}{te['sharpe']:>6.2f}{te['mdd']*100:>7.1f}"
              f"{alpha*100:>9.1f}{p:>7.3f}")

    # Multiple testing: Bonferroni across the 9 real strategies (exclude B&H).
    n_trials = len(strategies)
    print(f"\nMultiple-testing correction: {n_trials} strategies tested.")
    print("  A test p-value must be < {:.4f} to survive Bonferroni at 5%."
          .format(0.05 / n_trials))

    print("\nTop candidates by TEST Sharpe (this is diagnostic, not selection):")
    by_test = sorted(((n, r) for n, r in results.items() if n != 'BTC buy & hold'),
                     key=lambda kv: kv[1]['test']['sharpe'], reverse=True)
    for name, r in by_test[:4]:
        te = r['test']
        obs, se, t, p = block_bootstrap(slice_s(r['daily'], TEST_START, TEST_END))
        p_adj = min(1.0, p * n_trials)
        print(f"  {name:<30} test Sharpe {te['sharpe']:5.2f}  "
              f"t={t:5.2f}  raw p={p:.3f}  Bonferroni p={p_adj:.3f}")

    # Best train -> what happens out of sample (honest walk-forward check).
    best_train = ordered[0]
    print(f"\nBest TRAIN strategy: {best_train[0]}  "
          f"(train Sharpe {best_train[1]['train']['sharpe']:.2f})")
    print(f"  Out-of-sample test: CAGR {best_train[1]['test']['cagr']*100:.1f}%  "
          f"Sharpe {best_train[1]['test']['sharpe']:.2f}  "
          f"MDD {best_train[1]['test']['mdd']*100:.1f}%  "
          f"alpha {alpha_beta(slice_s(best_train[1]['daily'], TEST_START, TEST_END), slice_s(btc_daily, TEST_START, TEST_END))[1]*100:.1f}%/yr")

    # Cost stress on the top test candidate.
    top_name, top_r = by_test[0]
    top_fn = dict((n, f) for n, f, _ in strategies)[top_name]
    top_short = dict((n, s) for n, _, s in strategies)[top_name]
    print(f"\nCost stress on top test candidate ({top_name}):")
    for c in [COST_BASE, 0.0020, COST_STRESS]:
        d, _ = backtest(ind, top_fn, cost=c, short_book=top_short)
        m = metrics(slice_s(d, TEST_START, TEST_END))
        print(f"  {c*1e4:>4.0f} bps: CAGR {m['cagr']*100:5.1f}%  "
              f"Sharpe {m['sharpe']:4.2f}  MDD {m['mdd']*100:5.1f}%")

    # Parameter sensitivity for the top train candidate (is it a plateau?).
    print(f"\nParameter sensitivity for {best_train[0]} (test Sharpe):")
    if 'BTC-only' in best_train[0]:
        for lb in [30, 50, 100]:
            fn = (lambda lb=lb: (lambda i: _btc_variant(ind, i, lb)))()
            d, _ = backtest(ind, fn)
            m = metrics(slice_s(d, TEST_START, TEST_END))
            print(f"  lookback {lb:>3}: Sharpe {m['sharpe']:5.2f}  CAGR {m['cagr']*100:5.1f}%")
    elif 'XSMOM' in best_train[0] and 'L/S' not in best_train[0]:
        for lb, n in [('mom30', 3), ('mom30', 5), ('mom90', 3), ('mom90', 5),
                      ('mom90', 8), ('mom180', 5)]:
            d, _ = backtest(ind, xsmom_variant_factory(ind, lb, n))
            m = metrics(slice_s(d, TEST_START, TEST_END))
            print(f"  {lb} n={n}: Sharpe {m['sharpe']:5.2f}  CAGR {m['cagr']*100:5.1f}%")
    else:
        print("  (sensitivity sweep defined for the XSMOM / BTC families only)")

    # Year by year for the top two test candidates.
    print("\nYear-by-year net return (%):")
    years = sorted(set(ind['ret'].index.year))
    years = [y for y in years if 2021 <= y <= 2026]
    table = {}
    for name, r in by_test[:3]:
        table[name] = [float((1 + slice_s(r['daily'], pd.Timestamp(f'{y}-01-01'),
                                          pd.Timestamp(f'{y+1}-01-01')).dropna()
                             ).prod() - 1) * 100 for y in years]
    table['BTC'] = [float((1 + slice_s(btc_daily, pd.Timestamp(f'{y}-01-01'),
                                       pd.Timestamp(f'{y+1}-01-01')).dropna()).prod() - 1) * 100
                    for y in years]
    df = pd.DataFrame(table, index=years).round(1)
    print(df.to_string())


def _btc_variant(ind, i, lookback):
    t = ind['close'].index[i]
    ok = eligible(ind, t)
    sma = ind['close'].rolling(lookback).mean().loc[t]
    if (ok.get('BTC-USD', False) and ind['close'].loc[t, 'BTC-USD'] > sma['BTC-USD']):
        return inv_vol_weights(ind, t, ['BTC-USD'], target_vol=0.15, max_weight=1.0)
    return pd.Series(0.0, index=ind['close'].columns)


def xsmom_variant_factory(ind, lookback, n):
    def f(i):
        t = ind['close'].index[i]
        ok = eligible(ind, t)
        cand = ok & (ind['close'].loc[t] > ind['sma50'].loc[t])
        ranked = ind[lookback].loc[t][cand].dropna().sort_values(ascending=False)
        return inv_vol_weights(ind, t, ranked.head(n).index, target_vol=0.15)
    return f


if __name__ == '__main__':
    main()
