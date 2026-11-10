# No-Sharashi — Multi-Strategy Paper Trading Bot

Four paper-trading strategies run on GitHub Actions and publish a static dashboard:

| # | Strategy | Runner | State |
|---|----------|--------|-------|
| 1 | Adaptive Weighted (LiquidityAlpha) | `nosharashi_1/neural_flow/run_adaptive.py` | `nosharashi_1/paper_state_adaptive.json` |
| 2 | Adaptive v3 (trend + regime + Lasso) | `nosharashi_1/neural_flow/run_adaptive_v3.py` | `nosharashi_1/paper_state_v3.json` |
| 3 | TrendVol (50d time-series momentum) | `nosharashi_1/neural_flow/run_trend.py` | `nosharashi_1/paper_state_trend.json` |
| 4 | AMD Smart Money (PO3) | `amd_paper_trader.py` | `paper_state.json` |

## Workflows

| File | Schedule | Purpose |
|------|----------|---------|
| `.github/workflows/adaptive_daily.yml` | Daily at 01:00 UTC | Runs strategies 1–3 after the crypto daily candle closes |
| `.github/workflows/live_trade.yml` | Every 6 minutes | Runs strategy 4 (AMD) |

Both workflows share the `paper-trading-state` concurrency group, so they cannot
push state at the same time. They commit their own state files back to `main`.
No Personal Access Token is required: the workflows use the repository
`GITHUB_TOKEN` with `contents: write`.

## Dashboard

`index.html` is a static dashboard that reads `dashboard_data.json`. Regenerate
the feed after running any strategy:

```bash
python build_dashboard.py
python nosharashi_1/update_history_md.py
```

The dashboard shows all four strategies, a union of monitored assets, and the
last strategy comparison cards from `dashboard_data.json`.

## UltimateORB paper ledger (separate, not deployed)

`ultimateorb_paper/` is a **separate paper-only ledger** for the UltimateORB
6.6/10 research setting. It is not one of the four deployed strategies and has
no broker adapter, API keys, or live order path.

```bash
python -m ultimateorb_paper.app
```

Open <http://127.0.0.1:8787>. The dashboard has a **HALT** button that blocks
new paper entries by writing `ultimateorb_paper/state/control.json`; marking
and closing an existing paper position remain possible. The server binds to
loopback only. See [`ultimateorb_paper/README.md`](ultimateorb_paper/README.md)
for the CLI and safety notes.

## Tests

The live runners have synthetic-data regression tests. They use temporary state
files and never touch the deployed accounts:

```bash
python -m unittest discover -s tests -v
```

## Running locally

```bash
pip install -r requirements.txt

cd nosharashi_1
export PYTHONPATH=$(pwd)          # PowerShell: $env:PYTHONPATH = (Get-Location).Path
python neural_flow/run_adaptive.py
python neural_flow/run_adaptive_v3.py
python neural_flow/run_trend.py

cd ..
python amd_paper_trader.py --live --capital 10000 --symbols BTC-USD ETH-USD SOL-USD BNB-USD XRP-USD
python build_dashboard.py
python nosharashi_1/update_history_md.py
```

## Notes

- Market data comes from yfinance (free, no API key) and can be rate-limited or
  return incomplete data. Runners forward-fill gaps and never back-fill, so
  missing history cannot leak future prices into a signal.
- Entry costs, turnover costs, and short financing are charged in the live
  accounting. Period returns shown on the dashboard are net of those costs.
- Strategies 1 and 3 use a 2% no-trade band and compare targets with the actual
  price-drifted book. Small same-direction rebalances are skipped, but full
  exits and long/short reversals always execute.
- Crypto volatility is annualised with 365 days, not 252. Zero or non-finite
  volatility returns 1x leverage instead of levering up into bad data.
- LiquidityAlpha requires a full 200-day SMA warm-up before it can signal, and
  the runners refuse to trade if too little common history is available.
- Local caches, model pickles, account artefacts, and `config.json` are ignored
  by git; see `.gitignore`.
- `TRADE_HISTORY.md` is generated from the live state files by
  `nosharashi_1/update_history_md.py`.

## Backtest validation (honest, post-cost)

All figures below include execution costs (0.05% fee + 0.014% effective slippage
per unit turnover, 6.4 bps total) and short financing where shorting is used.
They are point-in-time estimates, not guarantees.

| Strategy | Window | CAGR/yr | Sharpe | Max DD | Alpha vs BTC | Assessment |
|----------|--------|---------|--------|--------|--------------|------------|
| Adaptive Weighted | 2024-11 → 2026-11 | -11.1% | -0.53 | 39.0% | — | **No edge.** Negative after look-ahead fix + costs. Do not trade. |
| Adaptive v3 | 2024-11 → 2026-11 | -5.0% | -0.43 | 20.7% | — | **No robust edge.** Bootstrap interval includes zero. |
| TrendVol BTC+ETH (no overlay) | 2020-11 → 2026-11 | +18.4% | 1.09 | 26.7% | +10.5%/yr | Modest full-period edge, high drawdown. |
| TrendVol BTC+ETH (no overlay) | 2024-01 → 2026-11 (test) | +16.8% | 0.98 | 16.4% | +10.2%/yr | Positive out-of-sample, but naive t≈1.7 (not significant). |
| TrendVol BTC+ETH (causal overlay) | 2024-01 → 2026-11 (test) | +10.2% | 0.76 | 13.5% | +5.4%/yr | Overlay cuts MDD 16.4→13.5% but also cuts return. |
| TrendVol 15 assets (causal overlay) | 2020-11 → 2026-11 | +28.0% | 1.35 | 22.8% | +18.4%/yr | In-sample only; not robust. |
| TrendVol 15 assets (causal overlay) | 2024-01 → 2026-11 (test) | +4.9% | 0.29 | 22.8% | +0.5%/yr | **Fails out-of-sample.** Rejected. |

> **Correction (2026-11-08).** An earlier version of this file reported
> TrendVol (15 assets) at +58.3% CAGR / Sharpe 2.55 / MDD 17.4%. That was
> **wrong**: the drawdown overlay was applied using an equity curve that
> already contained the next bar's return (look-ahead). The corrected, fully
> causal numbers are +28.0% / 1.35 / 22.8% in-sample and +4.9% / 0.29 /
> 22.8% out-of-sample. The live runner was always causal; only the backtest
> was flawed. The 15-asset expansion is rejected; the live runner trades
> BTC+ETH.

Interpretation:
- **Adaptive Weighted** loses money once look-ahead bias and transaction costs
  are removed (-11% CAGR). It should not be traded expecting positive edge.
- **Adaptive v3** shows no robust edge (bootstrap interval includes zero; 2y
  window is negative). Do not trade expecting positive returns.
- **TrendVol BTC+ETH** is the only configuration with a positive out-of-sample
  result (+16.8% CAGR / 0.98 Sharpe over ~2.8 years), but the naive t-stat is
  only ≈1.7 and block-bootstrap significance is weak. The causal drawdown
  overlay reduces max drawdown from 16.4% to 13.5% at the cost of roughly
  6.6%/yr of CAGR out-of-sample. Treat this as a modest, unproven edge.
- **The 15-asset expansion is rejected.** It only outperformed during the
  2020-2023 altcoin bull market; out-of-sample it returned +4.9% CAGR with
  0.29 Sharpe. Adding assets did not add durable edge.

None of the strategies demonstrates a statistically robust edge after all
costs with sufficient live evidence. Position sizes should be kept small and
expectations modest. Keep everything paper-only until a longer live record
exists.

### Broader edge search (2026-11-08)

A stricter 9-strategy search on 44 assets (including the collapsed coins LUNC,
USTC, FTT and CEL), with a held-out 2024-2026 test window, realistic costs and
Bonferroni correction, found **no statistically significant edge**. See
[`RESEARCH_EDGE.md`](RESEARCH_EDGE.md) for the full protocol and results.

The least-bad candidate is still BTC+ETH trend, but its test p-value is 0.115,
it does not survive multiple-testing correction, and it is fragile across
lookbacks (75/100-day lookbacks have negative or near-zero train CAGR; monthly
rebalancing fails). It also underperformed BTC buy-and-hold in the test window
(+14.4% vs +22.7% CAGR); its value is lower drawdown (19.1% vs 53.1%), not
higher return.

Consequence: keep TrendVol paper-only at BTC+ETH. Do not add capital and do
not deploy any of the new research strategies.

## Risk controls (live)

- **Drawdown overlay:** exposure is scaled from 1.0x down to a 0.25x floor as
  drawdown moves from 5% to 15% of the high-water mark.
- **No-trade band (2%):** small same-direction rebalances are skipped; full
  exits and reversals always execute.
- **Short financing:** 3 bp/day charged on short exposure (Adaptive Weighted
  and v3).
- **Data quality:** full SMA warm-up required; minimum common history enforced;
  stale/missing tickers are closed rather than silently dropped.
- **Idempotency:** the same daily bar is never processed twice for the same
  book.

## Manual trigger

Open the **Actions** tab, select either workflow, and choose **Run workflow**.
