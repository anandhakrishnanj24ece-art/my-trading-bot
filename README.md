# Paper Trading — UltimateORB, Multi-Filter ORB, FVG Duo, Trend Trio

**Paper-only. No broker adapter, no API keys, no live order path.**

This repository contains paper proxies and their supporting GitHub Actions.

## FVG Duo paper runner

`fvg_duo_paper/` is a paper-only port of the video-derived FVG Duo rules:

- **Nasdaq-100 CFD** (`USATECH.IDX/USD`), Dukascopy 1-minute feed resampled to
  5-minute candles, New York time. Free NQ futures feeds are delayed ~10 minutes,
  which would make the 09:40/09:50 Opening Drive untradeable. This is the same
  Nasdaq-100 feed used in the five-year validation, not NQ futures.
- **Opening Drive:** 10-minute FVG at 09:40/09:50, 1.5% stop, exit 16:00
- **Trend FVG:** 15-minute bullish FVG, EMA50 > EMA100, middle range >= 1.5 x
  ATR14, stop under the first candle low, 3R target, flat 16:00
- one position at a time; OD overrides Trend that day
- **1 CFD unit** ($1/point), $50,000 starting paper capital, $1/side all-in cost

It has **no broker adapter and no live-order path**. The GitHub workflow
`fvg_duo_paper_runner.yml` runs every five minutes during the US session and
commits `fvg_duo_paper/state/`.

Paper fills use the latest public price at the tick; the $1/side cost is all-in
spread and slippage. A signal older than 5.5 minutes is logged as missed, never
backfilled. Stop and target checks use completed 5-minute bars; the 15:55 exit
uses the latest price.
Run **Actions → FVG Duo paper halt switch** to block new entries.

> **Evidence caveat:** the frozen rules passed a five-year Nasdaq test, but the
> out-of-sample test on S&P 500, Dow, and Russell 2000 did **not** confirm a
> transferable edge. This is a paper monitoring deployment, not a validated
> live strategy.

## Trend Trio paper runner

`trend_trio_paper/` is a paper-only **proxy** for the video-derived Trend Trio
rules (`https://youtu.be/7gx3OPJYazQ`):

- **Data:** Dukascopy `USATECH.IDX/USD` Nasdaq-100 CFD 1-minute feed, 30 calendar
  days loaded so the 1-hour EMA200 has enough completed bars.
- **Setup 1:** SuperTrend 5m flips up, 15m SuperTrend green, long only,
  09:30–15:30 New York. Stop 2 x ATR14(5m) below entry, exit on 5m flip down,
  flat 15:59.
- **Setup 2:** Ultimate MA 4m turns green, 15m Sling Shot EMA38 > EMA62, long
  only, any hour. Exit when Chandelier Exit 4m turns down. No initial stop.
- **Setup 3:** AlphaTrend 2m buy/sell cross, 2m ADX14 >= 20, 09:30–11:30 New
  York. Long and short. No initial stop; exit on the opposite 2m signal or 16:00.
- **Setup 4:** AlphaTrend 1h buy/sell cross, 1h EMA200 filter, 10:00–15:00 New
  York. Long and short. Stop 2 x ATR14(1h); exit on the opposite 1h signal or
  16:00.
- One position at a time; the first fresh signal takes the trade and owns the
  exit. UMA positions are flattened Friday 16:45 to avoid weekend gaps.
- **Sizing:** 1 MNQ-equivalent at `$2/point`, $50,000 starting capital,
  `$2/side` all-in cost.

The video does not print every input. The implementation uses the open-source
defaults: SuperTrend 10/3.0 RMA, Ultimate MA = SMA20 with 2-bar colour
smoothing, Sling Shot 38/62, Chandelier 22/3.0 on closes, AlphaTrend period 14
with coefficient 1.0 and MFI, ADX14, EMA200. It is therefore a **proxy**, not a
verified duplicate.

> **Evidence caveat:** on the available 3-year NQ 1-minute history, this proxy
> produced 3,167 trades:
>
> | cost model | net | PF | max drawdown |
> |---|---:|---:|---:|
> | paper MNQ-equivalent ($2/pt, $2/side) | **+$6,875** | 1.05 | 12.8% |
> | NQ economics ($20/pt, $7.14/side) | **+$150,209** | 1.11 | 45.2% |
>
> A 30-day replay of the live Dukascopy CFD feed was **-$2,749** over 76 trades
> (PF 0.67). The video's 16-year result is not reproduced. This is
> observation-only, not a validated live strategy.

Run **Actions → Trend Trio paper halt switch** to block new entries. The runner
schedules every five minutes around the clock on weekdays because the UMA setup
is an any-hour setup.

## Other paper strategies

The original UltimateORB rules and dated trade CSV are still missing. The
scheduled runner is therefore an **opening-range-breakout proxy**, not an exact
replication:

- instrument: NQ futures price (`NQ=F`), sized as **1 MNQ** at **$2/point**
- timeframe: **5-minute** bars
- opening range: **08:30–09:00 CT**
- entry: close outside the opening range, **09:00–14:40 CT**
- target / stop: **6.6 / 10.0 points**
- one paper trade per session
- paper exit: **14:45 CT**
- commission: **0.95 points** per round turn
- starting paper capital: **$10,000**

## GitHub runner

The **UltimateORB paper session runner** starts near **08:55 CT** and loops every
6 minutes until **14:45 CT**. It does not depend on GitHub firing a cron every 6
minutes. Two UTC cron entries cover daylight-saving time.

It commits the paper state back to `main`. The state files are:

- `ultimateorb_paper/state/control.json`
- `ultimateorb_paper/state/paper_state.json`
- `ultimateorb_paper/state/paper_trades.csv`

GitHub may still delay or drop the initial session start. The robot is not a
real-time execution system.

## Halt

Run **Actions → UltimateORB paper halt switch → Run workflow** and choose
`halt` or `resume`.

Halt blocks new paper entries. It does **not** prevent marking or closing an open
paper position. The local dashboard button and CLI use the same control file.

## Web monitor

GitHub Pages serves `index.html` from this repository. It reads the committed
paper state and refreshes every 60 seconds:

`https://anandhakrishnanj24ece-art.github.io/my-trading-bot/`

## Local dashboard

```bash
python -m ultimateorb_paper.app
```

Open <http://127.0.0.1:8787>. The server binds to loopback only.

## CLI

```bash
python -m ultimateorb_paper.paper_runner status
python -m ultimateorb_paper.paper_runner halt
python -m ultimateorb_paper.paper_runner resume
python -m ultimateorb_paper.paper_runner open --direction long --price 28000
python -m ultimateorb_paper.paper_runner mark --high 28012 --low 27988 --close 28005
python -m ultimateorb_paper.paper_runner close --price 28004
```

## Tests

```bash
python -m unittest discover -s ultimateorb_paper/tests -v
python -m unittest discover -s fvg_duo_paper/tests -t . -v
python -m unittest discover -s trend_trio_paper/tests -t . -v
```

## Safety

- Paper only until the original rules, dated trade CSV, and tick-level replay
  exist.
- Do not add a broker adapter, API secret, or withdrawal permission to this
  folder.
- Keep `allow_live_orders` false.
- Keep the local server on `127.0.0.1`.
