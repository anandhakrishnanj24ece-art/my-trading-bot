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

- Not an automated UltimateORB signal generator. The original entry rules and trade CSV are still missing. Entries are manual or fed by an external signal source.
- Not a broker integration.
- Not a GitHub Pages app. A static page cannot persist a server-side halt flag. Run the local server or deploy the Python app to a private host with authentication.

## Files

- `config.json` — paper settings.
- `paper_engine.py` — ledger, barriers, halt guard, stats.
- `paper_runner.py` — CLI.
- `app.py` — local dashboard server.
- `dashboard.html` — halt button and paper account UI.
- `state/control.json` — halt flag (created at runtime, gitignored).
- `state/paper_state.json` — account and trades (created at runtime, gitignored).
- `state/paper_trades.csv` — closed-trade export (created at runtime, gitignored).

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
