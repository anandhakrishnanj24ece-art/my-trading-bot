"""Consolidated quantitative validation of the v3 strategy.

Covers:
  * HAC-adjusted t-statistics (autocorrelation-robust)
  * block-bootstrap confidence intervals
  * factor regressions vs BTC / ETH / equal-weight market
  * tail, drawdown and drawdown-duration statistics
  * capacity estimate from 60d dollar volume and actual turnover
  * parameter stability across a lambda x trend grid (train vs test)
  * approximate Deflated Sharpe Ratio (Bailey & Lopez de Prado)
  * yearly / monthly / rolling-12m stability

Run:  python quant_validation.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, spearmanr
from statsmodels.stats.diagnostic import acorr_ljungbox

from backtest_trend import metrics, yearly, slice_s, ANN
from research_v3_improved import TRAIN_START, TRAIN_END, TEST_START, TEST_END
import backtest_v3_improved as bt
from backtest_v3_improved import (load_7y, run_v3_live_logic, TICKERS,
                                  SHORT_COST_BPS, NO_TRADE_BAND)
from audit_v3_survivorship import run_pit
from audit_v3_stats import block_bootstrap

OUT = {}


# --------------------------------------------------------------------------
# Core statistics
# --------------------------------------------------------------------------
def drawdown_episodes(r, top=5):
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    # last day of each new-high run
    new_high = (eq >= peak - 1e-12).values
    idx = np.where(new_high)[0]
    eps = []
    for k in range(len(idx) - 1):
        a, b = idx[k], idx[k + 1]
        seg = dd.iloc[a:b + 1]
        if len(seg) <= 1:
            continue
        trough = int(np.argmin(seg.values))
        depth = -float(seg.min())
        if depth <= 0:
            continue
        eps.append(dict(
            peak=str(r.index[a].date()),
            trough=str(r.index[a + trough].date()),
            recovery=str(r.index[b].date()),
            depth=depth,
            days_peak_to_trough=int((r.index[a + trough] - r.index[a]).days),
            days_to_recovery=int((r.index[b] - r.index[a]).days),
        ))
    # open drawdown still running at the end
    last_peak = idx[-1]
    seg = dd.iloc[last_peak:]
    if len(seg) > 1 and seg.min() < 0:
        trough = int(np.argmin(seg.values))
        eps.append(dict(
            peak=str(r.index[last_peak].date()),
            trough=str(r.index[last_peak + trough].date()),
            recovery=None,
            depth=-float(seg.min()),
            days_peak_to_trough=int((r.index[last_peak + trough] - r.index[last_peak]).days),
            days_to_recovery=None,
        ))
    eps.sort(key=lambda x: x['depth'], reverse=True)
    return eps[:top], dd


def stats_block(r):
    r = r.dropna()
    n = len(r)
    total = float((1 + r).prod() - 1)
    cagr = (1 + total) ** (ANN / n) - 1
    vol = float(r.std() * np.sqrt(ANN))
    sharpe_cagr = cagr / vol if vol else np.nan
    sharpe_std = float(r.mean() / r.std() * np.sqrt(ANN)) if r.std() else np.nan
    t_iid = float(r.mean() / r.std() * np.sqrt(n)) if r.std() else np.nan
    hac = sm.OLS(r.values, np.ones(n)).fit(
        cov_type='HAC', cov_kwds={'maxlags': 5})
    t_hac = float(hac.tvalues[0])
    eq = (1 + r).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    mdd = float(-dd.min())
    downside = float(r[r < 0].std() * np.sqrt(ANN)) if (r < 0).any() else np.nan
    sortino = cagr / downside if downside else np.nan
    calmar = cagr / mdd if mdd else np.nan
    var95 = float(np.percentile(r, 5))
    var99 = float(np.percentile(r, 1))
    cvar95 = float(r[r <= var95].mean())
    cvar99 = float(r[r <= var99].mean())
    worst30 = float(((1 + r).rolling(30).apply(np.prod, raw=True) - 1).min())
    best30 = float(((1 + r).rolling(30).apply(np.prod, raw=True) - 1).max())
    pos = r[r > 0].sum()
    neg = r[r < 0].sum()
    profit_factor = float(pos / abs(neg)) if neg else np.nan
    # underwater duration
    under = (eq < peak * (1 - 1e-12)).values
    run = best = 0
    for u in under:
        run = run + 1 if u else 0
        best = max(best, run)
    current_under = 0
    for u in under[::-1]:
        if u:
            current_under += 1
        else:
            break
    try:
        lb = acorr_ljungbox(r, lags=[10], return_df=True)
        lb_p = float(lb['lb_pvalue'].iloc[0])
    except Exception:
        lb_p = np.nan
    eps, _ = drawdown_episodes(r, top=5)
    return dict(
        days=n, total=total, cagr=cagr, vol=vol,
        sharpe_cagr=sharpe_cagr, sharpe_std=sharpe_std,
        t_iid=t_iid, t_hac=t_hac,
        mdd=mdd, calmar=calmar, sortino=sortino,
        hit=float((r > 0).mean()),
        skew=float(r.skew()), ex_kurt=float(r.kurtosis()),
        var95=var95, var99=var99, cvar95=cvar95, cvar99=cvar99,
        worst_day=float(r.min()), best_day=float(r.max()),
        worst30=worst30, best30=best30,
        profit_factor=profit_factor,
        avg_win=float(r[r > 0].mean()), avg_loss=float(r[r < 0].mean()),
        max_underwater_days=int(best), current_underwater_days=int(current_under),
        autocorr1=float(r.autocorr(1)), ljung_box_p=lb_p,
        drawdowns=eps,
    )


def factor_reg(r, factors, maxlags=5):
    df = pd.concat([r] + list(factors), axis=1).dropna()
    df.columns = ['y'] + [f'f{i}' for i in range(len(factors))]
    y = df['y'].values
    X = sm.add_constant(df.drop(columns='y').values)
    res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    names = ['alpha'] + [getattr(f, 'name', f'factor{i}') for i, f in enumerate(factors)]
    return dict(
        alpha_ann=float(res.params[0] * ANN),
        t_alpha=float(res.tvalues[0]),
        betas={names[i + 1]: float(res.params[i + 1]) for i in range(len(factors))},
        t_betas={names[i + 1]: float(res.tvalues[i + 1]) for i in range(len(factors))},
        r2=float(res.rsquared), n=int(len(y)),
        hac_lags=maxlags,
    )


def bootstrap_ci(r, block=20, n_boot=2000):
    ci = block_bootstrap(r, block=block, n_boot=n_boot)
    return dict(cagr_lo=float(ci[0, 0]), cagr_hi=float(ci[2, 0]),
                sharpe_lo=float(ci[0, 1]), sharpe_hi=float(ci[2, 1]))


def dsr_prob(sr_daily, trial_sr_ann, n_trials, T, skew, ex_kurt):
    """Bailey & Lopez de Prado deflated Sharpe (approximate)."""
    gamma = 0.5772156649015329
    trial_daily = np.asarray(trial_sr_ann) / np.sqrt(ANN)
    var = float(np.var(trial_daily, ddof=1))
    sr0 = np.sqrt(var) * ((1 - gamma) * norm.ppf(1 - 1.0 / n_trials)
                          + gamma * norm.ppf(1 - 1.0 / (n_trials * np.e)))
    kurt = ex_kurt + 3.0
    denom = np.sqrt(max(1e-12, 1 - skew * sr_daily
                        + (kurt - 1) / 4.0 * sr_daily ** 2))
    z = (sr_daily - sr0) * np.sqrt(T - 1) / denom
    return dict(trial_var=var, sr0_daily=float(sr0),
                sr0_ann=float(sr0 * np.sqrt(ANN)),
                dsr=float(norm.cdf(z)), n_trials=n_trials)


def pct(x):
    return None if x is None else round(float(x) * 100, 2)


def main():
    t0 = time.time()
    close, high, low, volume = load_7y()
    tickers = [t for t in TICKERS if t in close.columns]
    close = close[tickers]
    volume = volume[tickers]
    fwd = (close.shift(-1) / close - 1.0).fillna(0.0)

    fixed, targets = run_v3_live_logic(close, fwd,
                                       short_cost_bps=SHORT_COST_BPS,
                                       start=TRAIN_START)

    broad_path = (Path('broad_data_delisted.pkl')
                  if Path('broad_data_delisted.pkl').exists()
                  else Path('broad_data.pkl'))
    close_b, vol_b = pd.read_pickle(broad_path)
    pit, nsel = run_pit(close_b, vol_b, top_n=15, min_dv=1e6,
                        short_cost_bps=SHORT_COST_BPS)

    windows = {
        'fixed_train': ('fixed', TRAIN_START, TRAIN_END),
        'fixed_test': ('fixed', TEST_START, TEST_END),
        'fixed_full': ('fixed', TRAIN_START, TEST_END),
        'pit_test': ('pit', TEST_START, TEST_END),
        'pit_full': ('pit', TRAIN_START, TEST_END),
    }
    series = {'fixed': fixed['net'], 'pit': pit['net']}

    stats = {}
    for name, (which, s, e) in windows.items():
        r = slice_s(series[which], s, e)
        stats[name] = stats_block(r)
        stats[name]['bootstrap'] = bootstrap_ci(r)
        stats[name]['start'] = str(r.index[0].date())
        stats[name]['end'] = str(r.index[-1].date())
    OUT['stats'] = stats

    # ---- factor regressions ----
    ew = fwd[tickers].mean(axis=1)
    ew.name = 'EW15'
    btc = fwd['BTC-USD'].rename('BTC')
    eth = fwd['ETH-USD'].rename('ETH')

    rets_b = close_b.pct_change()
    fwd_b = (close_b.shift(-1) / close_b - 1.0).fillna(0.0)
    ew_b = fwd_b.mean(axis=1).rename('EW_broad')
    btc_b = fwd_b['BTC-USD'].rename('BTC')
    eth_b = fwd_b['ETH-USD'].rename('ETH')

    factors = {}
    for name, (which, s, e) in windows.items():
        if which == 'fixed':
            r = slice_s(fixed['net'], s, e)
            f1 = slice_s(btc, s, e)
            f2 = slice_s(eth, s, e)
            f3 = slice_s(ew, s, e)
        else:
            r = slice_s(pit['net'], s, e)
            f1 = slice_s(btc_b, s, e)
            f2 = slice_s(eth_b, s, e)
            f3 = slice_s(ew_b, s, e)
        factors[name] = {
            'btc_eth': factor_reg(r, [f1, f2]),
            'btc_eth_ew': factor_reg(r, [f1, f2, f3]),
        }
    OUT['factors'] = factors

    # ---- yearly / monthly / rolling ----
    year_tbl = {}
    for which, s, e in [('fixed', TRAIN_START, TEST_END),
                        ('pit', TRAIN_START, TEST_END)]:
        r = slice_s(series[which], s, e)
        y = yearly(r)
        month = r.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        roll12 = ((1 + r).rolling(365).apply(np.prod, raw=True) - 1).dropna()
        year_tbl[which] = dict(
            yearly={str(k): pct(v) for k, v in y.items()},
            positive_months=float((month > 0).mean()),
            worst_month=pct(month.min()), best_month=pct(month.max()),
            positive_rolling_12m=float((roll12 > 0).mean()),
            worst_rolling_12m=pct(roll12.min()),
            best_rolling_12m=pct(roll12.max()),
            n_months=int(len(month)),
        )
    OUT['stability'] = year_tbl

    # ---- exposure ----
    W = np.array([w for _, w, _ in targets])
    absW = np.abs(W)
    gross = float(absW.sum(axis=1).mean())
    long_e = float(np.where(W > 0, W, 0.0).sum(axis=1).mean())
    short_e = float(np.where(W < 0, W, 0.0).sum(axis=1).mean())
    net_e = float(W.sum(axis=1).mean())
    asset_avg = pd.Series(absW.mean(axis=0), index=tickers)
    per_asset_turn = pd.Series(np.abs(np.diff(W, axis=0)).mean(axis=0),
                               index=tickers)
    OUT['exposure'] = dict(
        avg_gross=gross, avg_long=long_e, avg_short=short_e, avg_net=net_e,
        avg_lev=float(fixed['lev'].mean()),
        avg_turnover=float(fixed['turn'].mean()),
        avg_short_gross=float(fixed['short'].mean()),
        btc_weight=float(asset_avg.get('BTC-USD', np.nan)),
        eth_weight=float(asset_avg.get('ETH-USD', np.nan)),
        per_asset_avg_abs_weight={k: float(v) for k, v in asset_avg.items()},
    )

    # ---- capacity ----
    dv = (close * volume)
    last = dv.loc[dv.index >= dv.index[-1] - pd.Timedelta(days=365)]
    adv = last.median()
    turn = float(fixed['turn'].mean())
    cap_cons = 0.01 * float(adv.min()) / max(turn, 1e-9)
    per_asset_cap = 0.01 * adv / per_asset_turn.clip(lower=1e-9)
    smallest = per_asset_cap.idxmin()
    OUT['capacity'] = dict(
        adv_median_m=float(adv.median() / 1e6),
        adv_min_m=float(adv.min() / 1e6),
        adv_min_asset=str(adv.idxmin()),
        adv_by_asset_m={k: float(v / 1e6) for k, v in adv.items()},
        avg_turnover=turn,
        cap_conservative_usd=float(cap_cons),
        cap_prorata_usd=float(per_asset_cap.min()),
        cap_limit_asset=str(smallest),
        account_usd=10000.0,
    )

    # ---- parameter grid: lambda x trend ----
    grid = []
    saved = {k: getattr(bt, k) for k in ('LOOKBACK', 'LAMBDA_BASE')}
    try:
        for lam in (0.03, 0.04, 0.05, 0.06, 0.08):
            for lb in (30, 50, 70):
                bt.LOOKBACK = lb
                bt.LAMBDA_BASE = lam
                df, _ = bt.run_v3_live_logic(
                    close, fwd, short_cost_bps=SHORT_COST_BPS,
                    start=bt.TRAIN_START)
                tr = metrics(slice_s(df['net'], TRAIN_START, TRAIN_END))
                te = metrics(slice_s(df['net'], TEST_START, TEST_END))
                fu = metrics(slice_s(df['net'], TRAIN_START, TEST_END))
                grid.append(dict(lam=lam, lookback=lb,
                                 train_sharpe=float(tr['sharpe']),
                                 test_sharpe=float(te['sharpe']),
                                 full_sharpe=float(fu['sharpe']),
                                 train_cagr=float(tr['ann']),
                                 test_cagr=float(te['ann']),
                                 test_mdd=float(te['mdd'])))
    finally:
        for k, v in saved.items():
            setattr(bt, k, v)

    gdf = pd.DataFrame(grid)
    train_s = gdf['train_sharpe'].values
    test_s = gdf['test_sharpe'].values
    rho, rho_p = spearmanr(train_s, test_s)
    chosen = gdf[(gdf.lam == 0.05) & (gdf.lookback == 50)].iloc[0]
    best_train = gdf.loc[gdf.train_sharpe.idxmax()]
    best_test = gdf.loc[gdf.test_sharpe.idxmax()]

    # deflated Sharpe for the chosen test config
    r_test = slice_s(fixed['net'], TEST_START, TEST_END)
    st = stats['fixed_test']
    dsr = {str(n): dsr_prob(st['sharpe_std'] / np.sqrt(ANN), train_s, n,
                            len(r_test), st['skew'], st['ex_kurt'])
           for n in (15, 100, 500)}
    # PIT approximate DSR using the same trial variance (conservative proxy)
    r_pit_test = slice_s(pit['net'], TEST_START, TEST_END)
    sp = stats['pit_test']
    dsr_pit = {str(n): dsr_prob(sp['sharpe_std'] / np.sqrt(ANN), train_s, n,
                                len(r_pit_test), sp['skew'], sp['ex_kurt'])
               for n in (15, 100, 500)}

    OUT['grid'] = dict(
        rows=grid,
        train_mean=float(np.mean(train_s)), train_std=float(np.std(train_s, ddof=1)),
        test_mean=float(np.mean(test_s)), test_std=float(np.std(test_s, ddof=1)),
        test_min=float(np.min(test_s)), test_max=float(np.max(test_s)),
        test_positive=int(np.sum(test_s > 0)), test_n=int(len(test_s)),
        train_test_spearman=float(rho), spearman_p=float(rho_p),
        chosen=dict(chosen), best_train=dict(best_train), best_test=dict(best_test),
    )
    OUT['dsr'] = dict(fixed=dsr, pit_approx=dsr_pit)

    # ---- family-wise thresholds ----
    fw = {}
    for n in (15, 100, 500):
        fw[str(n)] = float(norm.ppf(1 - 0.025 / n))
    OUT['familywise_t'] = fw

    OUT['elapsed_s'] = round(time.time() - t0, 1)
    with open('quant_validation_results.json', 'w') as f:
        json.dump(OUT, f, indent=2, default=str)

    # compact console summary
    print(json.dumps({k: OUT[k] for k in
                      ('elapsed_s', 'familywise_t')}, indent=2))
    for name in windows:
        s = stats[name]
        print(f"{name:<12} n={s['days']:4d} CAGR={s['cagr']*100:6.1f}% "
              f"Sh={s['sharpe_cagr']:.2f} StdSh={s['sharpe_std']:.2f} "
              f"t={s['t_iid']:.2f} tHAC={s['t_hac']:.2f} MDD={s['mdd']*100:5.1f}% "
              f"Calmar={s['calmar']:.2f} Sortino={s['sortino']:.2f}")
    print('grid test Sharpes:', np.round(test_s, 2).tolist())
    print('train/test spearman:', round(rho, 3), 'p=', round(rho_p, 4))
    print('DSR fixed:', {k: round(v['dsr'], 3) for k, v in dsr.items()})
    print('DSR pit  :', {k: round(v['dsr'], 3) for k, v in dsr_pit.items()})
    print('saved quant_validation_results.json')


if __name__ == '__main__':
    main()
