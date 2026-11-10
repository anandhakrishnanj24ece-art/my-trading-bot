# UltimateORB — Paper Trading

**Paper-only. No broker adapter, no API keys, no live order path.**

This repository now contains only the UltimateORB paper proxy and its supporting
GitHub Actions.

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
```

## Safety

- Paper only until the original rules, dated trade CSV, and tick-level replay
  exist.
- Do not add a broker adapter, API secret, or withdrawal permission to this
  folder.
- Keep `allow_live_orders` false.
- Keep the local server on `127.0.0.1`.
