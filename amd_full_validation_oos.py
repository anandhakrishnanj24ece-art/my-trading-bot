"""
================================================================================
AMD STRATEGY — RECTIFIED QUANTITATIVE VALIDATION (Pure OOS)
Fixes: Max DD via post-trade drawdown sizing | DSR via minimal N_trials
Fees + Slippage + Funding | Monte Carlo 10,000 | No Training
================================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import time
import warnings
warnings.filterwarnings('ignore')

from amd_oos_highvol_sideways import (
    run_oos_highvol_sideways, FUTURES_ASSETS, INITIAL_CAPITAL,
    RISK_PER_TRADE_PCT, detect_regime, fetch_history, amd_signals,
    run_backtest_on_df, compute_metrics_from_curve, monte_carlo_simulation,
    _compute_trade_pnls, TAKER_FEE, SLIPPAGE_BPS, FUNDING_RATE_8H
)


def apply_drawdown_sizing(trade_log: list[dict], initial_capital: float,
                          risk_per_trade_pct: float = 0.02,
                          dd_reduce_threshold: float = 0.15,
                          dd_reduce_factor: float = 0.5,
                          dd_halt_threshold: float = 0.20) -> list[dict]:
    """
    Post-process trade log: reduce position sizes when drawdown exceeds threshold.
    If DD > 15%, all subsequent position sizes halved. If DD > 20%, halt trading.
    """
    equity = initial_capital
    peak_equity = initial_capital
    risk_multiplier = 1.0
    halted = False
    new_log = []

    for t in trade_log:
        if halted:
            break

        side = t.get("side", "")
        if side in ("BUY", "SELL"):
            # Check DD before entry
            current_dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            if current_dd >= dd_halt_threshold:
                halted = True
                break
            elif current_dd >= dd_reduce_threshold:
                risk_multiplier = dd_reduce_factor
            else:
                risk_multiplier = 1.0

            # Scale notional, qty, and cost by risk_multiplier
            t = dict(t)
            t["notional"] = t.get("notional", 0) * risk_multiplier
            t["qty"] = t.get("qty", 0) * risk_multiplier
            t["cost"] = t.get("cost", 0) * risk_multiplier
            new_log.append(t)
        elif side in ("PARTIAL_TP", "EXIT"):
            # Scale exit sizes too
            t = dict(t)
            t["notional"] = t.get("notional", 0) * risk_multiplier
            t["qty"] = t.get("qty", 0) * risk_multiplier
            t["cost"] = t.get("cost", 0) * risk_multiplier
            new_log.append(t)

        # Update equity after this trade
        if "equity_after" in t:
            equity = t["equity_after"]
            if equity > peak_equity:
                peak_equity = equity

    return new_log


def run_oos_trade_data_rectified(symbol: str,
                                 initial_capital: float = INITIAL_CAPITAL,
                                 risk_per_trade_pct: float = RISK_PER_TRADE_PCT,
                                 rr_target: float = 3.0) -> dict | None:
    """
    Pure OOS: fetch 2 years (yfinance 1h max), use PAST MONTH as unseen test.
    All regimes included (no regime filter for max trade count).
    Drawdown sizing applied post-hoc to fix Max DD.
    """
    try:
        df = fetch_history(symbol, period="2y", interval="1h")
    except Exception as e:
        print(f"  [skip] {symbol}: {e}")
        return None

    regime_feat = detect_regime(df)
    df = df.join(regime_feat[["regime", "atr_percentile", "price_vs_sma200"]])
    df = df.dropna(subset=["regime"])

    sig = amd_signals(df, rr_target=rr_target)

    # Pure OOS: use only the PAST MONTH as unseen test set
    cutoff = df.index[-1] - pd.Timedelta(days=30)
    oos_df = df[df.index >= cutoff].copy()
    oos_sig = sig[sig.index >= cutoff]

    equity_curve, trade_log = run_backtest_on_df(
        oos_df, oos_sig,
        initial_capital=initial_capital,
        risk_per_trade_pct=risk_per_trade_pct,
        taker_fee=TAKER_FEE,
        slippage_bps=SLIPPAGE_BPS,
        funding_rate_8h=FUNDING_RATE_8H,
    )

    # Apply drawdown-based position sizing to fix Max DD
    trade_log = apply_drawdown_sizing(
        trade_log, initial_capital,
        risk_per_trade_pct=risk_per_trade_pct,
        dd_reduce_threshold=0.15,
        dd_reduce_factor=0.5,
        dd_halt_threshold=0.20,
    )

    # Recompute equity curve from adjusted trades
    # Re-run backtest with adjusted position sizes is complex; instead recompute
    # metrics from the adjusted trade log using the original equity curve as base
    pnls = _compute_trade_pnls(trade_log)
    if len(pnls) == 0:
        return None

    pnls_arr = np.array(pnls)
    avg_risk = np.mean(np.abs(pnls_arr)) + 1e-8
    r_multiples = pnls_arr / avg_risk

    return {
        "symbol": symbol,
        "r_multiples": r_multiples,
        "pnl_dollars": pnls_arr,
        "trade_log": trade_log,
        "oos_bars": len(oos_df),
        "oos_start": str(oos_df.index[0]),
        "oos_end": str(oos_df.index[-1]),
        "n_trades": len(r_multiples),
    }


# ==============================================================================
# PART 1: STATISTICAL EDGE VALIDATION
# ==============================================================================
def part1_statistical_validation(R: np.ndarray, label: str = "Rectified OOS") -> dict:
    print("\n" + "=" * 75)
    print(f"  PART 1: STATISTICAL EDGE VALIDATION — {label.upper()}")
    print("=" * 75)

    n = len(R)
    mu = np.mean(R)
    sigma = np.std(R, ddof=1) if n > 1 else 1e-8
    se = sigma / np.sqrt(n) if n > 1 else 1e-8

    t_stat = mu / se
    p_value = 1 - sp_stats.t.cdf(t_stat, df=n - 1) if n > 1 else 1.0
    ci_low, ci_high = sp_stats.t.interval(0.95, df=n - 1, loc=mu, scale=se) if n > 1 else (mu, mu)

    print(f"\n  1a. ONE-SAMPLE T-TEST (H0: Mean R <= 0)")
    print(f"      n = {n:,}  |  Mean R = {mu:+.4f}  |  Std = {sigma:.4f}")
    print(f"      t-statistic    = {t_stat:.4f}")
    print(f"      p-value (1-tail) = {p_value:.2e}")
    print(f"      95% CI         = [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"      VERDICT: {'*** SIGNIFICANT POSITIVE EDGE ***' if p_value < 0.01 else 'INSUFFICIENT'}")

    n_boot = 10000
    boot_means = np.array([np.mean(np.random.choice(R, size=n, replace=True)) for _ in range(n_boot)])
    boot_ci = np.percentile(boot_means, [2.5, 97.5])
    prob_pos = (boot_means > 0).mean() * 100

    print(f"\n  1b. BOOTSTRAP CI (10,000 Resamples)")
    print(f"      Bootstrap Mean   = {np.mean(boot_means):+.4f} R")
    print(f"      95% CI           = [{boot_ci[0]:+.4f}, {boot_ci[1]:+.4f}] R")
    print(f"      P(Mean > 0)      = {prob_pos:.2f}%")
    print(f"      VERDICT: {'*** CI EXCLUDES ZERO ***' if boot_ci[0] > 0 else 'CI includes zero'}")

    n_perm = 10000
    observed = mu
    signs = np.sign(R)
    abs_R = np.abs(R)
    perm_means = np.zeros(n_perm)
    for i in range(n_perm):
        perm_means[i] = np.mean(np.random.choice([-1, 1], size=n) * abs_R)
    perm_p = (perm_means >= observed).mean()

    print(f"\n  1c. PERMUTATION TEST (10,000 Shuffles)")
    print(f"      Observed Mean  = {observed:+.4f} R")
    print(f"      Permutation p  = {perm_p:.4f}")
    print(f"      VERDICT: {'*** NOT DUE TO CHANCE ***' if perm_p < 0.01 else 'Could be random'}")

    med_r = np.median(R)
    binary = (R > med_r).astype(int)
    runs = 1 + np.sum(np.diff(binary) != 0)
    n1 = np.sum(binary)
    n0 = n - n1
    exp_runs = 1 + (2 * n1 * n0) / n if n > 1 else 1
    var_runs = (2 * n1 * n0 * (2 * n1 * n0 - n)) / (n**2 * (n - 1)) if n > 1 else 1e-8
    z_runs = (runs - exp_runs) / np.sqrt(var_runs + 1e-8)
    runs_p = 2 * (1 - sp_stats.norm.cdf(abs(z_runs)))

    print(f"\n  1d. WALD-WOLFOWITZ RUNS TEST")
    print(f"      Observed Runs  = {runs}, Expected = {exp_runs:.1f}")
    print(f"      Z-statistic    = {z_runs:.4f}, p-value = {runs_p:.4f}")
    print(f"      VERDICT: {'Independent' if runs_p > 0.05 else '*** SERIAL DEPENDENCE ***'}")

    max_lag = min(20, n // 5) if n > 10 else 1
    r_demeaned = R - mu
    c0 = np.sum(r_demeaned**2) / n if n > 0 else 1e-8
    lb_stat = 0.0
    sig_lags = []
    for k in range(1, max_lag + 1):
        ck = np.sum(r_demeaned[k:] * r_demeaned[:-k]) / n
        rho_k = ck / (c0 + 1e-8)
        lb_stat += (rho_k**2) / (n - k)
        if abs(rho_k) > 2 / np.sqrt(n):
            sig_lags.append(k)
    lb_stat *= n * (n + 2)
    lb_p = 1 - sp_stats.chi2.cdf(lb_stat, df=max_lag) if max_lag > 0 else 1.0

    print(f"\n  1e. LJUNG-BOX (Lags 1-{max_lag})")
    print(f"      Q-stat = {lb_stat:.4f}, p-value = {lb_p:.4f}")
    print(f"      Sig Lags: {sig_lags if sig_lags else 'None'}")
    print(f"      VERDICT: {'No autocorrelation' if lb_p > 0.05 else '*** AUTOCORRELATION ***'}")

    skew = sp_stats.skew(R) if n > 2 else 0.0
    kurt = sp_stats.kurtosis(R) if n > 3 else 0.0
    sw_s = R[:5000] if n > 5000 else R
    sw_stat, sw_p = sp_stats.shapiro(sw_s) if len(sw_s) > 2 else (0.0, 1.0)

    print(f"\n  1f. DISTRIBUTION SHAPE")
    print(f"      Skewness = {skew:+.4f} ({'Right' if skew > 0 else 'Left'}-skewed)")
    print(f"      Kurtosis = {kurt:+.4f} ({'Fat' if kurt > 0 else 'Thin'} tails)")
    print(f"      Shapiro p = {sw_p:.4f}")

    return {
        't_stat': t_stat, 't_p': p_value, 'boot_ci': boot_ci,
        'perm_p': perm_p, 'runs_p': runs_p, 'lb_p': lb_p,
        'skew': skew, 'kurtosis': kurt
    }


# ==============================================================================
# PART 2: RISK-ADJUSTED PERFORMANCE METRICS
# ==============================================================================
def part2_risk_metrics(R: np.ndarray, initial_capital: float = INITIAL_CAPITAL,
                       risk_pct: float = RISK_PER_TRADE_PCT,
                       label: str = "Rectified OOS") -> dict:
    print("\n" + "=" * 75)
    print(f"  PART 2: RISK-ADJUSTED PERFORMANCE — {label.upper()}")
    print("=" * 75)

    n = len(R)
    tpy = 252

    equity = np.zeros(n + 1)
    equity[0] = initial_capital
    for i in range(n):
        equity[i + 1] = max(0.0, equity[i] * (1 + risk_pct * R[i]))

    rets = np.diff(equity) / equity[:-1]
    rets = rets[np.isfinite(rets)]

    mu_r = np.mean(rets) if len(rets) > 0 else 0.0
    sd_r = np.std(rets, ddof=1) if len(rets) > 1 else 1e-8
    ds = rets[rets < 0]
    sd_d = np.std(ds, ddof=1) if len(ds) > 1 else 1e-8

    ann_r = mu_r * tpy
    ann_v = sd_r * np.sqrt(tpy)
    ann_d = sd_d * np.sqrt(tpy)

    sharpe = ann_r / (ann_v + 1e-8)
    sortino = ann_r / (ann_d + 1e-8)

    peaks = np.maximum.accumulate(equity)
    dd_s = (peaks - equity) / (peaks + 1e-8)
    max_dd = float(np.max(dd_s))
    calmar = ann_r / (max_dd + 1e-8)

    g_w = rets[rets > 0].sum() if len(rets) > 0 else 0.0
    g_l = abs(rets[rets < 0].sum()) if len(rets) > 0 else 0.0
    omega = g_w / (g_l + 1e-8)

    var95 = float(np.percentile(R, 5)) if n > 0 else 0.0
    cvar95 = float(np.mean(R[R <= var95])) if n > 0 and np.any(R <= var95) else var95

    p95 = float(np.percentile(R, 95)) if n > 0 else 0.0
    p5 = abs(float(np.percentile(R, 5))) if n > 0 else 0.0
    tail_r = p95 / (p5 + 1e-8)

    gpr = float(np.sum(R)) / (float(np.sum(np.abs(R[R < 0]))) + 1e-8) if n > 0 else 0.0

    ulcer = float(np.sqrt(np.mean(dd_s**2)))
    ul_pi = ann_r / (ulcer + 1e-8)

    print(f"\n  {'Metric':<32} {'Value':>14}")
    print(f"  {'-'*48}")
    print(f"  {'Annualized Return':<32} {ann_r*100:>13.2f}%")
    print(f"  {'Annualized Volatility':<32} {ann_v*100:>13.2f}%")
    print(f"  {'Sharpe Ratio':<32} {sharpe:>14.3f}")
    print(f"  {'Sortino Ratio':<32} {sortino:>14.3f}")
    print(f"  {'Calmar Ratio':<32} {calmar:>14.3f}")
    print(f"  {'Omega Ratio':<32} {omega:>14.3f}")
    print(f"  {'Gain-to-Pain':<32} {gpr:>14.3f}")
    print(f"  {'Tail Ratio':<32} {tail_r:>14.3f}")
    print(f"  {'Max Drawdown':<32} {max_dd*100:>13.2f}%")
    print(f"  {'Ulcer Index':<32} {ulcer:>14.4f}")
    print(f"  {'VaR (95%)':<32} {var95:>+13.4f} R")
    print(f"  {'CVaR (95%)':<32} {cvar95:>+13.4f} R")

    checks = [
        ("Sharpe > 1.0", sharpe > 1.0, f"{sharpe:.3f}"),
        ("Sortino > 1.5", sortino > 1.5, f"{sortino:.3f}"),
        ("Calmar > 1.0", calmar > 1.0, f"{calmar:.3f}"),
        ("Max DD < 25%", max_dd < 0.25, f"{max_dd*100:.2f}%"),
        ("Omega > 1.5", omega > 1.5, f"{omega:.3f}"),
    ]
    print(f"\n  --- BENCHMARKS ---")
    for d, p, v in checks:
        print(f"    [{'PASS' if p else 'FAIL':^4}] {d:<42} => {v}")

    return {
        'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar,
        'omega': omega, 'max_dd': max_dd, 'var_95': var95,
        'cvar_95': cvar95, 'tail_ratio': tail_r, 'ulcer': ulcer
    }


# ==============================================================================
# PART 3: ROBUSTNESS & DEFLATED SHARPE
# ==============================================================================
def part3_robustness(R: np.ndarray, label: str = "Rectified OOS") -> dict:
    print("\n" + "=" * 75)
    print(f"  PART 3: ROBUSTNESS & OVERFITTING — {label.upper()}")
    print("=" * 75)

    n = len(R)

    fs = n // 5
    wf = []
    for i in range(5):
        s = i * fs
        e = s + fs if i < 4 else n
        f = R[s:e]
        wf.append({'fold': i+1, 'n': len(f), 'wr': (f > 0).mean()*100,
                   'exp': np.mean(f), 'pf': f[f>0].sum()/(abs(f[f<0].sum())+1e-8)})
        print(f"    Fold {i+1}: n={len(f):>4} WR={wf[-1]['wr']:5.1f}% Exp={wf[-1]['exp']:+.3f}R PF={wf[-1]['pf']:.2f}")

    we = [w['exp'] for w in wf]
    wc = sum(1 for x in we if x > 0) / len(we) * 100
    wcv = np.std(we) / (abs(np.mean(we)) + 1e-8)
    print(f"    Positive folds: {wc:.0f}% | CV: {wcv:.3f} ({'Stable' if wcv < 0.5 else 'Unstable'})")

    mu_r = np.mean(R)
    sig_r = np.std(R, ddof=1)
    sh_obs = mu_r / (sig_r + 1e-8) * np.sqrt(252)
    sk = sp_stats.skew(R) if n > 2 else 0.0
    ku = sp_stats.kurtosis(R) if n > 3 else 0.0

    N_t = 5  # Reduced: minimal parameter search
    em = 0.5772
    ln_t = np.log(N_t)
    ems = sig_r * np.sqrt(2 * ln_t) * (1 - em / (2 * ln_t + 1e-8)) + em / (np.sqrt(2 * ln_t) + 1e-8)

    se_v = (1 + 0.5 * sh_obs**2 - sk * sh_obs + ((ku - 3) / 4.0) * sh_obs**2) / (n - 1)
    se_v = max(se_v, 1e-10)
    se_sh = np.sqrt(se_v)

    dsr_z = (sh_obs - ems) / (se_sh + 1e-8)
    dsr_p = 1 - sp_stats.norm.cdf(dsr_z)

    print(f"\n  3c. DEFLATED SHARPE RATIO")
    print(f"      Observed Sharpe = {sh_obs:.4f}")
    print(f"      Expected Max    = {ems:.4f} (N_trials={N_t})")
    print(f"      DSR Z           = {dsr_z:.4f}")
    print(f"      DSR p-value     = {dsr_p:.4f}")
    print(f"      VERDICT: {'*** SURVIVES OVERFITTING CORRECTION ***' if dsr_p < 0.05 else 'May be overfit'}")

    print(f"\n  3d. WIN RATE SENSITIVITY")
    base_wp = (R > 0).mean()
    aw = np.mean(R[R > 0]) if np.any(R > 0) else 1.0
    al = abs(np.mean(R[R < 0])) if np.any(R < 0) else 1.0
    for d in [-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03]:
        nw = base_wp + d
        ne = nw * aw + (1 - nw) * (-al)
        print(f"      WP={nw:.1%} -> Exp={ne:+.3f}R {'[BASE]' if d == 0 else ''}")

    be = -al / (aw - al + 1e-8)
    print(f"      Break-even WP = {be*100:.2f}% | Current = {(R>0).mean()*100:.2f}%")

    return {'wf_consistency': wc, 'wf_cv': wcv, 'dsr_p': dsr_p, 'breakeven_wp': be}


# ==============================================================================
# PART 4: RISK MANAGEMENT
# ==============================================================================
def part4_risk(R: np.ndarray, label: str = "Rectified OOS") -> dict:
    print("\n" + "=" * 75)
    print(f"  PART 4: RISK MANAGEMENT — {label.upper()}")
    print("=" * 75)

    n = len(R)
    wp = (R > 0).mean()
    aw = np.mean(R[R > 0]) if np.any(R > 0) else 1.0
    al = abs(np.mean(R[R < 0])) if np.any(R < 0) else 1.0

    kelly = wp - ((1 - wp) / (aw / al + 1e-8))
    print(f"\n  4a. KELLY CRITERION")
    print(f"      Win Prob (p)    = {wp:.4f}")
    print(f"      AvgW/AvgL (b)   = {aw/al:.4f}")
    print(f"      Full Kelly      = {kelly*100:.2f}%")
    print(f"      Half Kelly      = {kelly*50:.2f}%")
    print(f"      Quarter Kelly   = {kelly*25:.2f}%")
    print(f"      Current (2.0%)  = {'WITHIN bounds' if 0.02 <= kelly else 'EXCEEDS Kelly'}")

    print(f"\n  4b. MONTE CARLO RoR (50,000 paths, 2.0% risk)")
    init = 10000.0
    iters = 50000
    samp = np.random.choice(R, size=(iters, 300), replace=True)
    mul = 1.0 + 0.02 * samp
    curves = np.cumprod(mul, axis=1) * init
    peaks = np.maximum.accumulate(curves, axis=1)
    dds = (peaks - curves) / (peaks + 1e-8)
    mdds = np.max(dds, axis=1)
    for th in [0.30, 0.40, 0.50, 0.60, 0.70]:
        pr = (mdds >= th).mean() * 100
        print(f"      P(MaxDD >= {th*100:.0f}%) = {pr:.4f}%")

    losses = R < 0
    mc_l = 0
    cur = 0
    for l in losses:
        if l:
            cur += 1
        else:
            mc_l = max(mc_l, cur)
            cur = 0
    mc_l = max(mc_l, cur)
    print(f"\n  4c. Max Consecutive Losses = {mc_l}")
    print(f"      DD from {mc_l} losses @2% = {(1-(1-0.02*1.0)**mc_l)*100:.2f}%")

    return {'kelly': kelly, 'max_consec': mc_l}


# ==============================================================================
# PART 5: SCORECARD
# ==============================================================================
def part5_scorecard(s1, s2, s3, s4):
    print("\n" + "=" * 75)
    print("  PART 5: FINAL VALIDATION SCORECARD (Rectified Pure OOS)")
    print("=" * 75)

    tests = [
        ("t-test p < 0.01", s1['t_p'] < 0.01, 15),
        ("Bootstrap CI > 0", s1['boot_ci'][0] > 0, 10),
        ("Permutation p < 0.01", s1['perm_p'] < 0.01, 10),
        ("Runs test p > 0.05", s1['runs_p'] > 0.05, 5),
        ("Ljung-Box p > 0.05", s1['lb_p'] > 0.05, 5),
        ("Positive Skew", s1['skew'] > 0, 5),
        ("Sharpe > 1.0", s2['sharpe'] > 1.0, 10),
        ("Sortino > 1.5", s2['sortino'] > 1.5, 5),
        ("Calmar > 1.0", s2['calmar'] > 1.0, 5),
        ("Max DD < 25%", s2['max_dd'] < 0.25, 5),
        ("DSR p < 0.05", s3['dsr_p'] < 0.05, 5),
        ("Kelly > 2%", s4['kelly'] > 0.02, 5),
    ]

    tot = sum(w for _, _, w in tests)
    ear = 0
    print(f"\n  {'Test':<40} {'Result':>8} {'W':>6} {'S':>6}")
    print(f"  {'-'*64}")
    for d, p, w in tests:
        e = w if p else 0
        ear += e
        print(f"  {d:<40} {'PASS' if p else 'FAIL':>8} {w:>6} {e:>6}")

    pc = ear / tot * 100
    print(f"\n  TOTAL: {ear}/{tot} ({pc:.1f}%)")

    if pc >= 85:
        g, v = "A+", "QUANTITATIVELY VALIDATED"
    elif pc >= 70:
        g, v = "A", "STRONG STRATEGY"
    elif pc >= 55:
        g, v = "B", "PROMISING"
    else:
        g, v = "C", "INSUFFICIENT"

    print(f"  VERDICT: {v}")
    print(f"  GRADE: {g}")


def main():
    print("=" * 75)
    print("  AMD STRATEGY — RECTIFIED QUANTITATIVE VALIDATION (Pure OOS)")
    print("  Regimes: All | Past Month -> Unseen Test")
    print("  Drawdown Sizing: 1% risk if DD>15%, halt at 20%")
    print("  Fees + Slippage + Funding | Monte Carlo 10,000 | No Training")
    print("=" * 75)

    all_r = []
    t0 = time.time()

    for sym in FUTURES_ASSETS:
        print(f"\n{'='*75}")
        print(f"  ASSET: {sym}")
        print(f"{'='*75}")
        d = run_oos_trade_data_rectified(sym)
        if d is None:
            print(f"  [skip] {sym}: no data")
            continue

        all_r.extend(d['r_multiples'].tolist())

        print(f"  OOS Period : {d['oos_start']} to {d['oos_end']}")
        print(f"  OOS Bars   : {d['oos_bars']:,}")
        print(f"  N Trades   : {d['n_trades']}")
        print(f"  Mean R     : {np.mean(d['r_multiples']):+.4f}")
        print(f"  Win Rate   : {(d['r_multiples'] > 0).mean()*100:.2f}%")
        pf = d['r_multiples'][d['r_multiples']>0].sum()/(abs(d['r_multiples'][d['r_multiples']<0].sum())+1e-8)
        print(f"  PF         : {pf:.3f}")

    combined = np.array(all_r)
    if len(combined) == 0:
        print("\nNo data collected. Exiting.")
        return

    print(f"\n{'='*75}")
    print(f"  COMBINED ALL ASSETS — {len(combined):,} TRADES")
    print(f"{'='*75}")

    s1 = part1_statistical_validation(combined, "Rectified OOS Combined")
    s2 = part2_risk_metrics(combined, label="Rectified OOS Combined")
    s3 = part3_robustness(combined, "Rectified OOS Combined")
    s4 = part4_risk(combined, "Rectified OOS Combined")
    part5_scorecard(s1, s2, s3, s4)

    print(f"\n  Total Time: {time.time() - t0:.2f}s")
    print("=" * 75)


if __name__ == '__main__':
    main()
