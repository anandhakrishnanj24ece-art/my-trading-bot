# UltimateORB — Paper Mode with Halt Button

**Paper-only research ledger. No broker. No API keys. No live orders.**

This is the execution/risk layer for the UltimateORB 6.6/10 research candidate:

- target: **6.6 points**
- stop: **10.0 points**
- 1 MNQ
- commission: **0.95 points** per round turn
- starting paper capital: **$10,000**
- one open paper position at a time

The **HALT** button is an entry kill switch. It writes `state/control.json`; the engine refuses any new paper entry while halted. Marking and closing an existing paper position are still allowed.

> 6.6/10 is the best result in the PDF proxy, but it is not proven with tick-level fills. This app exists to collect that evidence, not to imply the edge is validated.

## Run

```bash
python -m ultimateorb_paper.app
```

Open <http://127.0.0.1:8787>.

The server binds to loopback only. It refuses non-loopback hosts because it has no authentication.

## GitHub 6-minute paper runner

`.github/workflows/ultimateorb_paper_runner.yml` runs the paper proxy every 6 minutes (GitHub may delay scheduled runs).

It uses **NQ=F 5-minute bars**, converts the price to MNQ dollars at **$2/point**, and:

- builds the opening range from the first 30 minutes of the US cash session (08:30–09:00 CT);
- goes long on a close above the range, short on a close below it;
- uses the **6.6-point target / 10-point stop**, 1 MNQ;
- allows at most one paper trade per session;
- exits at the 15:00 CT session close;
- writes only local paper JSON/CSV state, then commits that state back to the repo.

This is a **proxy**, not the original UltimateORB strategy. The original entry rules and dated trade CSV are still missing.

The runner still respects the halt flag. An existing position can be marked and closed while halted; new entries are blocked.

### Halt from GitHub

Run **Actions → UltimateORB paper halt switch → Run workflow** and choose `halt` or `resume`.

The local dashboard button and the CLI still write the same `state/control.json`.

## CLI

```bash
python -m ultimateorb_paper.paper_runner status
python -m ultimateorb_paper.paper_runner halt
python -m ultimateorb_paper.paper_runner resume

python -m ultimateorb_paper.paper_runner open --direction long --price 28000
python -m ultimateorb_paper.paper_runner mark --high 28012 --low 27988 --close 28005
python -m ultimateorb_paper.paper_runner close --price 28004
python -m ultimateorb_paper.paper_runner export
```

`open` exits with code 2 and does not create a position when halted.

## What the halt button does

1. Dashboard/CLI writes `state/control.json` with `"halted": true`.
2. `PaperEngine.open_trade()` raises `TradingHalted`.
3. Existing positions can still be marked and closed.
4. `resume` clears the flag.

It does **not** cancel real orders. There is no real order path in this folder.

## What this is not

- Not the original UltimateORB signal generator. The original entry rules and trade CSV are still missing. The scheduled runner is a simple opening-range-breakout proxy, not a replication.
- Not a broker integration.
- Not a GitHub Pages app. A static page cannot persist a server-side halt flag. Run the local server, use the GitHub halt workflow, or deploy the Python app to a private host with authentication.

## Files

- `config.json` — paper settings.
- `paper_engine.py` — ledger, barriers, halt guard, stats.
- `paper_runner.py` — CLI.
- `auto_paper.py` — scheduled 6-minute paper proxy.
- `app.py` — local dashboard server.
- `dashboard.html` — halt button and paper account UI.
- `state/control.json` — halt flag (force-added by the scheduled workflow).
- `state/paper_state.json` — account and trades (force-added by the scheduled workflow).
- `state/paper_trades.csv` — closed-trade export (force-added by the scheduled workflow).
- `.github/workflows/ultimateorb_paper_runner.yml` — 6-minute paper runner.
- `.github/workflows/ultimateorb_halt.yml` — GitHub halt/resume switch.

## Tests

```bash
python -m unittest discover -s ultimateorb_paper/tests -v
```

The tests verify that halt blocks new entries, that halt does not close an existing position, that target/stop fills are conservative, and that the paper-only guard is set.

## Safety rules

- Do not add a broker adapter or API secret to this folder.
- Keep `allow_live_orders` false.
- Keep the server on `127.0.0.1` unless you add authentication.
- Paper only until a dated trade CSV and tick-level replay exist.
