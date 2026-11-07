# Quantitative Validation — v3 (trend50 + BTC150)

**Date:** 2026-11-07  
**Data:** cleaned 7y cache, 2019-11-07 → 2026-11-07; P&L 2020-11-07 → 2026-11-07 (1y warm-up). FIL/AVAX pre-launch placeholders removed.  
**Costs:** 5 bp fee + 1.4 bp slippage per unit turnover, 3 bp/day short financing.  
**Holdout:** train 2020-11 → 2023-12; test 2024-01 → 2026-11.  
**Universes:** fixed 15 (live universe, survivorship-biased) and point-in-time (PIT) top 15 by 60-day dollar volume (honest universe). PIT broad data now includes FTT, CEL, LUNC and USTC.

---

## 1. Verdict

**Conditional pass. The implementation is mechanically validated; the edge is suggestive but not proven.**

- The fixed-15 backtest is statistically suggestive: test **21.1% CAGR / 1.49 standard Sharpe**, t = **2.52** (HAC 2.41), bootstrap Sharpe **[0.41, 2.76]**, deflated Sharpe ≈ **0.97**, factor alpha **18.7%/yr** (t = 2.32).
- The survivorship-free PIT test is **not statistically significant**: **11.3% CAGR / 0.84 Sharpe**, t = **1.41** (HAC 1.40), bootstrap Sharpe **[-0.27, 2.11]**, deflated Sharpe ≈ **0.81**, factor alpha **10.4%/yr** (t = 1.30). The PIT *full* period is stronger: **11.9% / 0.96**, t 2.35 (HAC 2.38), alpha 10.3% (t 2.07).
- The fixed-15 test is an **upper bound**. The honest central estimate is closer to **11–12% CAGR / 0.8–1.0 Sharpe**, not 21% or the 2020 calendar’s 39.0%.
- **What changed in this revision:** the 7y cache now drops pre-launch FIL/AVAX placeholders (the test is unchanged; the 2020 calendar rises from 37.6% to **39.0%**); the PIT universe now includes the collapsed names that Yahoo still serves (FTT, CEL, LUNC, USTC) with zero/delisting handling, but the PIT test is unchanged at 11.3% / 0.81. A parameter ensemble was tested; it lowers drawdown but also lowers the test return, so the single 50/0.05 config is retained.
- The forward paper account is the decisive test.

---

## 2. Scorecard

| Area | Result | Status |
|---|---|---|
| Causal signal / no look-ahead | Truncation test 0.0 weight difference; next-bar returns; full 150d regime | ✅ Pass |
| Accounting | Drift-aware band and turnover; entry/transition costs; funding | ✅ Pass |
| Data quality | 2,191 P&L days, no duplicates/non-positive prices. FIL pre-2020-10-15 and AVAX pre-2020-09-22 placeholders removed (leading NaNs intentional); PIT includes FTT/CEL/LUNC/USTC. Still missing ANC/MIR and pre-rename LUNA/UST. | ⚠️ Conditional (improved) |
| Fixed-15 out-of-sample | 21.1% / 1.49, t 2.52, bootstrap > 0, DSR 0.97 | ⚠️ Suggestive |
| PIT out-of-sample | 11.3% / 0.84, t 1.41, bootstrap includes 0, DSR 0.81 | ❌ Not significant |
| PIT full period | 11.9% / 0.96, t 2.35 (HAC 2.38), alpha 10.3% (t 2.07) | ⚠️ Suggestive |
| Multiple testing | Test t 2.52 < 2.94 (15 trials) / 3.48 (100 trials); test observed during tuning | ⚠️ Conditional |
| Parameter stability | All 15 λ×trend test Sharpes positive (0.72–1.57); lookback 50 plateau; train/test rank −0.07 (p 0.80) | ⚠️ Mixed |
| Parameter ensemble | λ ensemble neutral (20.8/1.55); lookback 30/50 lowers test to 17.6/1.39 but full MDD 21.5→15.5; PIT 30/50 10.9/0.83 | ⚠️ Risk reduction, not return |
| Factor exposure | Fixed alpha 18.7% (t 2.32), PIT test 10.4% (t 1.30), PIT full 10.3% (t 2.07); low BTC/ETH beta, R² 2–11% | ⚠️ Fixed pass, PIT weak test |
| Tail / drawdown | Test MDD 10.5–12.5%; fixed full MDD 21.5%; PIT full 12.5%; fat tails; long underwater periods | ⚠️ Acceptable for 15% vol |
| Capacity | $10k far below ~$0.96M conservative / ~$13.9M pro-rata | ✅ Pass |
| Execution / costs | Hourly 01:00 UTC ≈ close-to-close; 2× costs test 19.4%; PIT 5× costs 6.1% | ✅ Pass at 1×; ⚠️ PIT fragile |
| Live readiness | 9→15 dry-run works; real state untouched; cleaned cache is backtest-only (live downloads 1y) | ✅ Pass |
| Subperiod stability | Positive 6 of 7 years; 2022 flat; 2025 weak; positive rolling 12m 79% | ✅ Acceptable |

---

## 3. Performance

| Window | Universe | Days | Total | CAGR | Vol | Std Sharpe | CAGR/vol Sharpe | t (iid) | t HAC | MDD | Calmar | Sortino |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train 2020-11→2023-12 | Fixed 15 | 1,150 | +140.8% | 32.2% | 15.8% | 1.84 | 2.03 | 3.27 | 2.98 | 21.5% | 1.50 | 2.11 |
| Test 2024-01→2026-11 | Fixed 15 | 1,041 | +72.6% | 21.1% | 13.4% | 1.49 | 1.57 | 2.52 | 2.41 | 10.5% | 2.00 | 2.05 |
| Full 2020-11→2026-11 | Fixed 15 | 2,191 | +315.7% | 26.8% | 14.7% | 1.68 | 1.82 | 4.13 | 3.83 | 21.5% | 1.25 | 2.08 |
| Test 2024-01→2026-11 | PIT top 15 | 1,041 | +35.7% | 11.3% | 14.0% | 0.84 | 0.81 | 1.41 | 1.40 | 12.5% | 0.91 | 0.93 |
| Full 2020-11→2026-11 | PIT top 15 | 2,191 | +96.6% | 11.9% | 12.6% | 0.96 | 0.95 | 2.35 | 2.38 | 12.5% | 0.96 | 1.01 |

The standard (mean/std) Sharpe is reported alongside the headline CAGR/vol Sharpe. They are close; the standard is the honest one used in the significance tests.

---

## 4. Statistical significance

### 4.1 Block bootstrap (20-day blocks, 2,000 resamples)

| Window | CAGR 95% CI | Sharpe 95% CI |
|---|---|---|
| Fixed train | [4.7%, 64.0%] | [0.37, 3.03] |
| Fixed test | **[4.3%, 46.8%]** | **[0.41, 2.76]** |
| Fixed full | [10.1%, 47.6%] | [0.78, 2.56] |
| PIT test | **[-4.4%, 34.2%]** | **[-0.27, 2.11]** |
| PIT full | [1.1%, 24.3%] | [0.15, 1.70] |

The fixed-15 test has a positive lower bound. The PIT test includes zero.

### 4.2 Multiple testing

Family-wise t thresholds (two-sided 5%):

| Trials | Threshold |
|---:|---:|
| 15 | 2.94 |
| 100 | 3.48 |
| 500 | 3.89 |

- Fixed test t = **2.52** → below the 15-trial threshold. Not significant after even a modest search adjustment.
- Fixed full t = **4.13** → above 3.48, but it includes the tuning period.
- PIT test t = **1.41**; PIT full t = **2.35** → not significant after adjustment.

### 4.3 Deflated Sharpe Ratio (approximate)

Trial dispersion comes from the 15-config λ×trend grid (annualized σ ≈ 0.125). The actual search was broader, so these DSR values are **optimistic**.

| Universe | N = 15 | N = 100 | N = 500 | Expected max Sharpe under null |
|---|---:|---:|---:|---:|
| Fixed 15 | 0.986 | 0.979 | 0.973 | 0.22 / 0.32 / 0.38 |
| PIT (same variance proxy) | 0.851 | 0.810 | 0.779 | 0.22 / 0.32 / 0.38 |

The DSR and the family-wise t answer different questions. The DSR says the fixed-15 Sharpe is far above what 15–500 correlated trials would produce. The family-wise t is more conservative and says the fixed-15 test is not significant after 15 trials. The PIT test is weak under both.

---

## 5. Parameter stability (λ × trend, test Sharpe)

| λ \ lookback | 30 | 50 | 70 |
|---:|---:|---:|---:|
| 0.03 | 0.99 | 1.40 | 0.78 |
| 0.04 | 1.10 | 1.48 | 0.75 |
| 0.05 | 1.07 | **1.57** | 0.74 |
| 0.06 | 1.13 | 1.57 | 0.72 |
| 0.08 | 1.24 | 1.41 | 0.76 |

- All 15 configurations are positive out-of-sample: **mean 1.11, std 0.32, range [0.72, 1.57]**.
- Lookback 50 is a genuine OOS plateau (1.40–1.57 across λ). Lookback 30 is moderate (0.99–1.24); lookback 70 is consistently worse (0.72–0.78).
- λ has little effect inside lookback 50; the edge is not sensitive to the Lasso strength.
- Train/test rank correlation is **−0.07 (p = 0.80)**: in-sample ranking does **not** persist. The best train config (λ 0.03, lookback 70, train 2.25) has test only 0.78. The best test config (λ 0.06, lookback 50, 1.571) is essentially tied with the chosen λ 0.05/lookback 50 (1.570).
- This supports lookback 50 as a robust region, but it also shows that picking the in-sample best would have hurt. The chosen config being the best test is partly luck/selection.

### 5.1 Does a parameter ensemble help? (exploratory)

Equal-capital averages of neighbouring plateau configs, fixed 15, same costs:

| Sleeves | Test CAGR/Sharpe/MDD | Full CAGR/Sharpe/MDD |
|---|---:|---:|
| Single 50 / λ 0.05 | **21.1 / 1.57 / 10.5%** | 26.8 / 1.82 / 21.5% |
| Lookback 30/50 | 17.6 / 1.39 / 10.1% | 24.4 / 1.83 / 15.5% |
| Lookback 30/50/70 | 15.2 / 1.23 / 9.4% | 23.2 / 1.76 / 16.2% |
| λ 0.04/0.05/0.06 (50) | 20.8 / 1.55 / 10.8% | 26.8 / 1.83 / 20.4% |
| Both 30/50 × 0.04/0.05/0.06 | 17.6 / 1.40 / 10.3% | 24.8 / 1.86 / 14.7% |

- Component daily-return correlations are high: λ 0.04/0.05/0.06 ≈ 0.99 (the λ ensemble is effectively the same strategy); lookback 30/50 ≈ 0.81; lookback 70 ≈ 0.63–0.85.
- The ensemble **does not improve the test return**. It does reduce the full-period drawdown (21.5% → 14.7–15.5%) and slightly raises the full Sharpe (1.82 → 1.83–1.86).
- PIT lookback 30/50 ensemble: test **10.9% / 0.83** (vs 11.3% / 0.81), full 11.1% / 0.94 (vs 11.9% / 0.95). The PIT test t rises only to 1.44; still not significant.
- **Verdict:** keep the single 50/0.05. The ensemble is a legitimate risk-reduction option (lower drawdown), not a fix for the unstable train/test ranking, and it is another test-observed trial.

Earlier sweeps (not rerun here) found broad plateaus for regime 100–250, band 2%, vol target 15–25%, and 15 assets vs 9. Those are consistent with the above, but they also expand the search space and therefore the multiple-testing haircut.

---

## 6. Factor exposure (HAC, 5 lags)

| Window | Model | Alpha/yr | t(alpha) | BTC β | ETH β | EW β | R² |
|---|---|---|---:|---:|---:|---:|---:|
| Fixed test | BTC+ETH | 18.7% | 2.32 | 0.046 | −0.002 | — | 0.023 |
| Fixed test | BTC+ETH+EW15 | 20.1% | 2.60 | 0.003 | −0.051 | 0.086 | 0.062 |
| Fixed full | BTC+ETH | 21.9% | 3.60 | 0.046 | 0.016 | — | 0.062 |
| Fixed full | BTC+ETH+EW15 | 21.2% | 3.66 | 0.004 | −0.032 | 0.085 | 0.107 |
| PIT test | BTC+ETH | 10.4% | 1.30 | 0.022 | 0.030 | — | 0.045 |
| PIT test | BTC+ETH+EW broad | 11.3% | 1.49 | −0.009 | −0.014 | 0.076 | 0.082 |
| PIT full | BTC+ETH | 10.3% | 2.07 | 0.015 | 0.020 | — | 0.032 |
| PIT full | BTC+ETH+EW broad | 10.1% | 2.06 | −0.008 | −0.010 | 0.053 | 0.056 |

The strategy is not a crypto-beta clone: BTC beta ≈ 0.05, ETH ≈ 0, equal-weight beta ≈ 0.08, R² 2–11%. The fixed-15 alpha is significant at the single-test 5% level; the PIT alpha is not (t 1.3–1.5 test; 2.1 full).

---

## 7. Tail and drawdown

| Window | MDD | Max underwater | Current | Worst day | Worst 30d | CVaR95 | CVaR99 | Skew | Ex-kurt | Hit | Profit factor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed test | 10.5% | 255d | 33d | −3.11% | −8.67% | −1.53% | −2.35% | +0.77 | 5.2 | 29.7% | 1.33 |
| Fixed full | 21.5% | 494d | 33d | −8.12% | −14.73% | −1.68% | −2.94% | +0.47 | 14.6 | 28.4% | 1.41 |
| PIT test | 12.5% | 364d | 33d | −4.22% | −9.70% | −1.72% | −2.87% | +0.15 | 6.6 | 29.7% | 1.17 |
| PIT full | 12.5% | 627d | 33d | −4.22% | −9.70% | −1.55% | −2.67% | +0.44 | 8.7 | 24.1% | 1.23 |

Largest drawdowns:

- **Fixed:** 2023-02-19 → 2023-08-07 (−21.5%, recovery 2023-11-11, 265 days); 2024-12-07 → 2025-01-30 (−10.5%, recovery 2025-07-16, 221 days); 2021-09-18 → 2021-10-11 (−10.0%, recovery 2023-01-26, 495 days).
- **PIT:** 2024-03-12 → 2024-11-05 (−12.5%, recovery 2024-11-29, 262 days); 2025-08-22 → 2025-10-12 (−10.7%, recovery 2026-08-22, 365 days); 2023-03-08 → 2023-04-28 (−10.6%, recovery 2023-06-29, 113 days).

The return distribution is positively skewed but fat-tailed (excess kurtosis 5–15), with a low hit rate (24–30%). The strategy earns from a minority of large trend wins, not from frequent small gains. Long underwater periods (up to 627 days) are a real feature, not a bug.

---

## 8. Capacity

- Average gross exposure 0.195; average long 0.126; average short −0.069; average net 0.057; average leverage 0.32; average daily turnover 4.58%.
- BTC weight 1.74%, ETH 1.46%; the book is diversified.
- Median 60-day dollar volume across the 15: ~$2.6B. Smallest: ALGO ~$4.4M.
- Conservative capacity (all turnover in the smallest asset): **~$0.96M**.
- Pro-rata capacity: **~$13.9M**.
- The $10k account is far below either estimate. Treat the numbers as order-of-magnitude: yfinance volume units are inconsistent across assets (e.g. ETH ~$37M vs BNB ~$1.08B), so do not rely on the exact figures.

---

## 9. Execution, costs, funding

- Live cost model: 5 bp fee + 1.4 bp slippage = 6.4 bp per unit turnover; 3 bp/day short financing baseline.
- 2× costs: fixed test **19.4% / 1.45** (vs 21.1% / 1.57).
- PIT 5× costs: **6.1% / 0.44** — the PIT edge is fragile to high costs.
- Short financing 10 bp/day: fixed test **18.5% / 1.38**.
- Conservative gross funding 3 bp/day: fixed test **19.3% / 1.44**; 2020 calendar **36.8% / 2.26** (clean data); 2020→2026 **25.2% / 1.72**.
- Hourly 01:00 UTC execution (2024-11→2026-11): **29.3% / 2.23** vs close-to-close **28.4% / 2.21**. The 24-hour delay test is too pessimistic; next-bar/01:00 UTC is the right model.
- `funding_bps` is implemented in `run_v3_live_logic` but defaults to 0, so production numbers are unchanged.

---

## 10. Subperiod stability

| Year | Fixed 15 | PIT top 15 |
|---|---:|---:|
| 2020 (Nov–Dec) | +12.7% | 0.0% |
| 2021 | +69.6% | +3.2% |
| 2022 | −1.1% | −1.0% |
| 2023 | +27.4% | +41.8% |
| 2024 | +23.8% | +8.0% |
| 2025 | +11.7% | +4.0% |
| 2026 YTD | +24.8% | +20.9% |

- Positive months: 46.6% (fixed), 35.6% (PIT).
- Positive rolling 12m: 79.3% (fixed), 78.7% (PIT).
- Worst rolling 12m: −13.5% (fixed), −4.8% (PIT).
- Best rolling 12m: +91.3% (fixed), +51.3% (PIT).
- Worst month: −11.2% (fixed), −8.0% (PIT).
- Best month: +21.6% (fixed), +12.3% (PIT).

The PIT edge is much weaker in 2024 and 2025 and much stronger in 2023. The fixed-15 2021 (+69.6%) is the survivor-heavy alt bull year. The 2022 flat year and the long 2021–2023 underwater period are the main risk.

---

## 11. Data integrity and the 2020 result

- The fixed 15 P&L window has 2,191 row-dates, no duplicate dates and no non-positive prices. The cleaned 7y cache intentionally has leading NaNs for FIL/AVAX before their real spot launches.
- **Fixed:** FIL-USD had a synthetic/IOU history from 2019-11 (prices ~$6–$30) but Filecoin spot launched on **2020-10-15** (the series jumps from ~$30 to ~$59). AVAX-USD was flat at $4.986754 with tiny volume from 2020-07-13 until its real launch on **2020-09-22**.
- The 7y loader now applies a launch-date map plus a generic leading-flat-run filter; `trend_data_7y_clean.pkl` is the cache the backtest uses. The live runner still downloads 1y, so it is unaffected.
- The 2024–2026 test is well after both launches and is unchanged: **21.1% / 1.57** fixed, **11.3% / 0.81** PIT. The 2020 calendar and the first ~50 days of the train are affected.
- Cleaned 2020 calendar (short 3 bp/day): **+39.0% / 2.40 / 8.8%**; gross 3 bp/day: **+36.8% / 2.26 / 9.0%**. Excluding FIL: **36.8% / 2.28**; excluding FIL and AVAX: **41.5% / 2.55**. The post-launch FIL was a small positive; the AVAX placeholder was a drag, not an inflation.
- 2020 calendar → 2026: **26.9% / 1.83** (short 3), **25.2% / 1.72** (gross 3). The 2020-11 start (1y warm-up) is **26.8% / 1.82**.
- The **2020 calendar** is still a bull-year upper bound, not a clean OOS: fixed survivors, cold-start regime (Jan–Apr partly cash), late-listed assets survive. The PIT 2020 is unavailable because the broad data starts 2020-11.
- **PIT data fix:** FTT, CEL, LUNC and USTC were added individually (Yahoo’s batch download drops dead tickers). LUNA/UST/ANC/MIR are no longer available; LUNC/USTC are the post-collapse continuations of LUNA/UST. The PIT engine now requires `close > 0` and marks a delisted/zero-price asset out of the universe; the last valid price earns the collapse return.
- Adding the collapsed names did **not** change the PIT test (11.3% / 0.81); they were not top-15 by 60-day dollar volume in 2024–2026. The PIT full moved from 12.0% / 0.96 to **11.9% / 0.95**, so the honest full-period edge is essentially unchanged and still includes the 2021–22 collapse names.
- The fixed 15 is the live universe and remains survivorship-biased. The PIT top-15 by 60d $volume is the honest benchmark. The broad data still lacks ANC/MIR (and the pre-rename LUNA/UST), so the PIT is also incomplete, but it is much fairer than the fixed 15.

---

## 12. Live readiness

- The 9→15 transition dry-run works: old book marked to market, band vs actual drifted weights, transition cost charged, temporary state written with universe `v3_15`, net period return matches P&L.
- The real `nosharashi_1/paper_state_v3.json` is untouched: still the old 9-asset book at **$9,423.22**.
- The dashboard (`dashboard_data.json`, `index.html`) is still old; the next live run updates it.
- The live core `nosharashi_1/neural_flow/alpha/v3_core.py` is untracked and must be committed before the workflow can run the new version.
- CI currently runs the old v3. No commit or push has been made.

---

## 13. Known limitations

1. **Survivorship:** the fixed 15 are today’s survivors. The PIT test is the honest one and is not significant. The PIT universe is better but still incomplete (no ANC/MIR; no pre-rename LUNA/UST).
2. **2020 data:** pre-launch FIL/AVAX are now removed, but the 2020 calendar is still an upper bound (fixed survivors, cold regime, late listings).
3. **Selection:** the 2024–2026 test was observed during tuning; the test t is marginal after multiple testing; the DSR is optimistic because the trial set was broader than the 15-config grid. The parameter ensemble was also tested on the same window; it is another trial.
4. **Short sample:** 3 years of test, one crypto cycle; the 2024–2026 mix (bull, flat, bull) is not a stress test for a prolonged bear.
5. **Funding:** the 3 bp/day short baseline is an assumption; 10 bp/day reduces the fixed test to 18.5% / 1.38. Gross 3 bp/day gives 19.3% / 1.44.
6. **PIT cost fragility:** at 5× costs the PIT test falls to 6.1% / 0.44.
7. **Tail risk:** full-period MDD 21.5% (fixed) / 12.5% (PIT), max underwater 494 days (fixed) / 627 days (PIT), excess kurtosis 5–15.
8. **Capacity:** fine at $10k; the ~$0.96M–$13.9M estimate is rough because the volume data units are inconsistent.
9. **Data source:** yfinance is convenient but not institutional quality. The launch-date filter and the delisted-name handling reduce, but do not eliminate, data risk.
10. **Live execution:** the 01:00 UTC hourly test matches close-to-close, but real fills, outages, and exchange-specific funding are not modeled.

---

## 14. What would change the verdict

- **Positive:** 6–12 months of forward paper/live results with the real cost model, positive and within the backtest’s drawdown band; a longer PIT test including the still-missing ANC/MIR and pre-rename LUNA/UST; independent data (not yfinance); pre-registered parameters and no further test tuning.
- **Negative:** forward drawdown beyond ~15–20%; PIT alpha remaining insignificant; funding above ~10 bp/day; live slippage above the modeled 1.4 bp; capacity constraints above ~$1M.

---

## 15. Bottom line

The strategy is **well built and causally clean**, and the fixed-15 backtest has a **suggestive** edge. But the evidence is not strong enough to call it a proven forward edge:

- Fixed-15 test: **21.1% CAGR / 1.49 Sharpe / t 2.52 / DSR 0.97** — good, but survivorship-biased and observed during tuning. Full: **26.8% / 1.68 / t 4.13**.
- PIT test: **11.3% CAGR / 0.84 Sharpe / t 1.41 / bootstrap includes zero / DSR 0.81** — the honest number. PIT full: **11.9% / 0.96 / t 2.35**.
- 2020 calendar: **+39.0%** (clean data) — a bull-year upper bound, not a forecast.
- Data: pre-launch FIL/AVAX removed; PIT now includes FTT/CEL/LUNC/USTC. The test did not change; the PIT full is essentially unchanged.
- Parameter grid: all positive OOS, lookback 50 plateau, but in-sample ranking does not persist. The tested ensemble lowers drawdown but also lowers the test return; the single 50/0.05 is retained.
- Factor alpha: 18.7% fixed (t 2.32), 10.4% PIT test (t 1.30), 10.3% PIT full (t 2.07) — the fixed alpha is real-looking, the PIT test alpha is not statistically established.
- Tail: 10.5–12.5% test MDD, 21.5% fixed full MDD, 12.5% PIT full MDD, long underwater periods — normal for a 15% vol trend strategy, but real.

**Use the PIT range (11–12% CAGR / 0.8–1.0 Sharpe) as the base case, the fixed-15 21% / 1.5 as the optimistic upper bound, and let the forward paper account decide.** The recent fixes improve data integrity and the PIT universe, but they do not turn the PIT test into a significant result — only forward evidence can do that.
