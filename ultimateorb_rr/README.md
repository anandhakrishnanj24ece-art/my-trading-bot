# UltimateORB — fixed-stop / R:R replay

Status: **research only. Not a deployment candidate. Paper trading only.**

Source report: `../UltimateORB-Backtest-Report.pdf` (8 Nov 2026).

The report references `UltimateORB_NT_2026-11-07.csv`, but that trade file is not on disk. Page 3 of the PDF contains a vector scatter of all trades' MAE/MFE. I extracted those points and replayed **fixed-stop counterfactuals**. This recovers the excursion distribution, not the original trade order.

## What was extracted

- 1,526 excursion points: **1,452 winners, 74 losers** (the report has 1,528 trades; 2 hand-added trades have no excursion data).
- Winner MFE mean: **6.62 pts** (report: 6.6).
- Loser MAE mean: **54.41 pts** (report: 54).
- Replay gross: **3.66 pts/trade** (report: 3.8).
- Replay net after $1.90 round-turn commission: **2.71 pts/trade** (report: 2.88).

The extraction therefore reproduces the report's aggregate excursion economics closely.

## Method

For a fixed stop `S`:

- Winner: if `MAE > S`, the stop is hit first and the result is `-S`; otherwise the winner's original target/MFE is kept.
- Loser: if `MAE > S`, the result is `-S`; otherwise the original loss `-MAE` is kept.
- Base costs: **0.95 points** commission per trade ($1.90 / $2 per point per MNQ).
- Stop-slippage stress: add `slip` points to every stop exit.

This is a valid tightening test under the assumption that the current exit is a fixed/dynamic stop and that the reported MAE occurred before the final exit. If the live stop is trailing or time-based, treat the numbers as an approximation.

**Not estimable from the PDF:** raising the target. Winners captured 100% of MFE, so their MFE is censored at the original target. The PDF cannot tell us how much further a winner would have run. Chronological max drawdown is also not estimable because the scatter has no trade order.

## Fixed-stop results

Target rule unchanged (average target ≈ 6.6 pts). R:R below is **average win / average loss**, not the target/stop ratio.

| Stop | Avg win / avg loss | Win rate | Avg net after commission | PF (net) | Est. net, 1 MNQ | Stop rate | Worst trade | Net after +3 pts uniform slippage |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1.40 | 91.8% | **+4.68 pts** | 11.12 | $14,269 | 7.6% | -5 | +1.68 |
| 10 | 0.70 | 91.9% | **+4.31 pts** | 6.16 | $13,140 | 7.5% | -10 | +1.31 |
| 15 | 0.47 | 91.9% | +3.95 pts | 4.31 | $12,055 | 7.3% | -15 | +0.95 |
| 20 | 0.36 | 92.1% | **+3.63 pts** | 3.39 | $11,091 | 7.1% | -20 | +0.63 |
| 25 | 0.29 | 92.5% | +3.41 pts | 2.93 | $10,401 | 6.5% | -25 | +0.41 |
| 30 | 0.24 | 92.9% | **+3.22 pts** | 2.62 | $9,826 | 5.8% | -30 | +0.22 |
| 40 | 0.19 | 93.5% | +2.96 pts | 2.30 | $9,021 | 4.7% | -40 | -0.04 |
| 50 | 0.16 | 93.9% | +2.75 pts | 2.09 | $8,379 | 3.9% | -50 | -0.26 |
| 60.7 (report avg stop) | 0.14 | 94.0% | +2.42 pts | 1.84 | $7,372 | 3.1% | -60.7 | -0.59 |
| 80 | 0.13 | 94.6% | +2.42 pts | 1.83 | $7,390 | 1.7% | -80 | -0.58 |
| 100 | 0.12 | 95.0% | +2.66 pts | 1.98 | $8,103 | 0.6% | -100 | -0.34 |
| 120 | 0.12 | 95.1% | +2.65 pts | 1.97 | $8,076 | 0.3% | -120 | -0.35 |

Key points:

- The current report edge is roughly **6.6 / 60.7 ≈ 0.11–0.14** average win/loss.
- Tightening to **30** gives roughly **6.6 / 30 ≈ 0.24**; **20** gives **0.36**; **10** gives **0.70**; **5** gives **1.40**.
- The estimated net improvement is monotonic in this replay: tighter is better, all the way to 5 points. That is a warning sign, not a green light. The winner MAE data is extremely clean (95% of winners have MAE ≈ 0), so sub-10-point stops are likely overstated by tick noise, spread, latency, and stop slippage.
- A fixed 60.7 stop is slightly **worse** than the current replay because ~18 winners had MAE > 60.7 and the current dynamic stop let them recover. The current stop is not equivalent to a hard 60.7-point stop.
- **20–30 points** is the most defensible research range: it cuts the loss tail, keeps the stop rate near 6–7%, and remains positive after a 3-point uniform slippage assumption.
- **10 points** has the best base-case net among the plausible stops, but needs tick-level validation before it can be considered.
- **5 points** should be treated as an artifact until proven with tick data.

## Stop-slippage stress

Average net points per trade after commission, with extra slippage applied only when a stop is hit:

| Stop | 0 | 2 | 5 | 10 | 20 | 30 |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 4.68 | 4.52 | 4.30 | 3.92 | 3.16 | 2.40 |
| 10 | 4.31 | 4.16 | 3.93 | 3.56 | 2.81 | 2.06 |
| 20 | 3.63 | 3.49 | 3.28 | 2.93 | 2.22 | 1.51 |
| 30 | 3.22 | 3.10 | 2.93 | 2.64 | 2.05 | 1.47 |
| 60.7 | 2.42 | 2.35 | 2.26 | 2.11 | 1.80 | 1.49 |

Even 10 points of stop slippage leaves 10/20/30 positive and above the current fixed-60.7 replay. At 30 points of slippage, the advantage mostly disappears. Gap risk through the stop is not modeled; that is the main remaining tail risk.

## Target widening: exact backtest not runnable

The requested variants **10/30** and **12/30** require raising the target. The PDF's MFE is censored at the original target (100% MFE capture), so the hit rate at a higher target cannot be recovered exactly.

**Correction:** the 87.2% / 84.8% break-even numbers are for **10/60.7** and **12/60.7**, not 10/30 and 12/30. Correct break-even win rates:

| Target / stop | Break-even win rate |
|---|---:|
| 10 / 30 | **77.4%** |
| 12 / 30 | **73.7%** |
| 10 / 60.7 | 87.2% |
| 12 / 60.7 | 84.8% |

Break-even with the current 60.7 average stop and commission:

| Target | Stop | Break-even win rate | Current win rate | Buffer |
|---:|---:|---:|---:|---:|
| 8 | 60.7 | 89.7% | 95.2% | +5.5 pts |
| 10 | 60.7 | **87.2%** | 95.2% | +8.0 pts |
| 12 | 60.7 | **84.8%** | 95.2% | +10.4 pts |
| 14 | 60.7 | 82.5% | 95.2% | +12.7 pts |
| 20 | 60.7 | 76.4% | 95.2% | +18.8 pts |

The current target distribution makes widening a different strategy: the 75th percentile of winners is only **8 points**, the 90th is **12**, and the 95th is **14**. A fixed 10-point target is above ~75% of current targets; a fixed 12-point target is above ~90%. Widening is not a small parameter tweak, and the PDF cannot prove the extra continuation exists.

## Target-widening scenario (10/30 and 12/30)

To run the variants anyway, I used the smallest extension assumption available:

- After a winner reaches its original target `M`, the remaining path is a **driftless random walk** between the new target `T` and the stop `-S`.
- Probability of reaching `T` before `-S`: `p = (M + S) / (T + S)` (gambler's ruin).
- If `MAE > S`, the trade is stopped before it ever reaches `M`; it is a loss at `-S`.
- Losers cannot reach a wider target.
- Commission 0.95 points.

This is a neutral no-post-target-edge scenario. It is not an exact backtest.

| Target/stop | Break-even | Win rate | Avg win | Avg loss | Avg net | PF | Est. net, 1 MNQ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10/30 | 77.4% | 83.5% | 9.05 | -29.59 | **+2.67 pts** | 1.55 | $8,147 |
| 12/30 | 73.7% | 80.1% | 11.05 | -29.83 | **+2.90 pts** | 1.49 | $8,841 |
| 10/60.7 | 87.2% | 88.6% | 9.05 | -54.15 | +1.86 pts | 1.30 | $5,675 |
| 12/60.7 | 84.8% | 86.5% | 11.05 | -55.33 | +2.09 pts | 1.28 | $6,379 |

Reference points:

- Current replay: **+2.71 pts/trade**, 94.0% win, PF 1.84.
- 6.6/30 stop-only (target unchanged): **+3.22 pts/trade**, 92.9% win, PF 2.62.

So under the neutral assumption, 10/30 and 12/30 are **not** better than simply tightening the stop to 30 and keeping the target rule. They are roughly current-edge or slightly better, with much lower win rates and higher loss rates.

### What continuation edge would be needed?

The scenario can be re-run with a `continuation_boost` added to `p` (percentage points). A boost of +2 pp makes both variants beat the 6.6/30 stop-only reference; +5 pp makes them clearly better:

| Variant | Boost | Win rate | Avg net | PF |
|---|---:|---:|---:|---:|
| 10/30 | 0 pp | 83.5% | +2.67 | 1.55 |
| 10/30 | +2 pp | 85.0% | +3.29 | 1.75 |
| 10/30 | +5 pp | 87.2% | +4.16 | 2.12 |
| 12/30 | 0 pp | 80.1% | +2.90 | 1.49 |
| 12/30 | +2 pp | 81.7% | +3.59 | 1.66 |
| 12/30 | +5 pp | 84.1% | +4.61 | 1.98 |

The +2 pp / +5 pp continuation is an **assumption**, not a measured edge. The PDF cannot tell us whether it exists. To validate it, we need the trade CSV or tick paths.

## Past-year status (Nov 2025 – Nov 2026)

A true past-year backtest of the modified stops is **blocked**:

- The PDF scatter has MAE/MFE but no dates, so the last 12 months cannot be isolated exactly.
- `UltimateORB_NT_2026-11-07.csv` is still missing.
- Free NQ 15-minute history is not available for a full year (Yahoo caps 15m at ~60 days; only 1h is available for a year).

### Best-available 365-trade proxy

The PDF groups winners and losers, but within each group the marker order follows the plotted trade order. The report's 2026 slice has 365 trades and a 94% win rate, so the proxy uses the **last 343 winners + last 22 losers**.

Validation against the report's 2026 slice:

- Proxy win rate: **93.97%** vs report 94%.
- Proxy avg win: **8.61 pts** vs OOS report ~8 pts.
- Proxy avg loss: **-82.87 pts** vs OOS report ~-72 pts.
- Proxy current net: **3.09 pts/trade ($2,257)** vs reported 2026 $2,704.

The proxy's current net is ~17% below the reported 2026 net, so treat absolute dollars as approximate and likely conservative.

| Setting | Win rate | Avg net/trade | PF | Est. 365-trade net | Worst trade |
|---|---:|---:|---:|---:|---:|
| Current | 93.97% | +3.09 pts | 1.62 | $2,257 | -143.88 |
| **6.6/10** | 90.41% | **+6.69 pts** | 7.48 | **$4,883** | -10.95 |
| **6.6/20** | 90.41% | +5.76 pts | 3.93 | $4,203 | -20.95 |
| **6.6/30** | 90.69% | **+4.93 pts** | 2.76 | **$3,601** | -30.95 |
| 6.6/40 | 90.96% | +4.19 pts | 2.17 | $3,060 | -40.95 |
| 6.6/60.7 | 91.78% | +3.10 pts | 1.65 | $2,264 | -61.65 |
| 10/30 | 85.87% | +3.47 pts | 1.81 | $2,535 | -30.95 |
| 12/30 | 83.20% | +4.07 pts | 1.79 | $2,970 | -30.95 |

Key read:

- **6.6/30** is the best evidence-backed past-year proxy: ~$3,601 vs ~$2,257 current, PF 2.76 vs 1.62.
- **6.6/10** and **6.6/20** look better on net, but they rely on the very clean winner-MAE data; 10 points needs tick-level validation.
- **12/30** and **10/30** are both worse than 6.6/30. Target widening still does not pay for its extra stop-outs.
- Max drawdown is **not** estimated from the proxy because the exact chronological order of winners and losers is unavailable.

The real past-year test still needs the dated trade CSV.

## Recommendation

1. Keep UltimateORB **paper-only**; do not change deployment.
2. Obtain `UltimateORB_NT_2026-11-07.csv` (or the NinjaTrader export with entry/exit timestamps and MAE/MFE). Without it, the exact 10/30 and 12/30 grid cannot be run.
3. If the CSV is obtained, replay **chronologically** with:
   - actual entry/exit timestamps and intrabar path,
   - fixed stops 10/20/30/40/60,
   - targets current/8/10/12/14,
   - 0/2/5/10-point stop slippage and commissions,
   - max DD, worst trade, stop rate, and winners sacrificed.
4. Do not adopt a sub-10-point stop from this PDF-only replay. The clean winner-MAE data is not enough to prove that a 5–10-point stop survives real NQ microstructure.
5. The best research candidate, if one must be chosen now, is a **20–30 point fixed stop** with the existing target rule. It has a better cost buffer than the current setup and is less dependent on perfect stop fills than 10 points.

## Files

- `extract_and_replay.py` — extraction and replay code.
- `target_widening_scenario.py` — 10/30 and 12/30 scenario replay.
- `past_year_proxy.py` — 365-trade past-year proxy from the last excursion points.
- `ultimateorb_excursions.csv` — 1,526 recovered MAE/MFE pairs.
- `past_year_proxy_trades.csv` — 365 proxy trades.
- `past_year_proxy_results.csv` — past-year proxy setting results.
- `stop_grid.csv` — fixed-stop results.
- `stop_slippage.csv` — stop-slippage stress.
- `break_even.csv` — break-even win rates for stop tightening and target widening.
- `target_widening_grid.csv` — 10/30, 12/30, 10/60.7, 12/60.7 scenario results.
- `target_widening_continuation.csv` — continuation-edge sensitivity.
