"""Build the static dashboard feed consumed by ``index.html``.

The live runners persist one JSON state file per strategy.  This script merges
them into a single ``dashboard_data.json`` document so the browser only needs
one request and every strategy is rendered from the same snapshot in time.

Run from the repository root (the workflows do this automatically)::

    python build_dashboard.py
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIVE_DIR = ROOT / "nosharashi_1"


def load_state(path: Path) -> dict:
    """Return a state dict, or an empty dict when the file is missing/invalid."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def build_feed() -> dict:
    adaptive = load_state(LIVE_DIR / "paper_state_adaptive.json")
    v3 = load_state(LIVE_DIR / "paper_state_v3.json")
    trend = load_state(LIVE_DIR / "paper_state_trend.json")
    # The AMD engine runs from the repository root.
    amd = load_state(ROOT / "paper_state.json")

    now_utc = datetime.now(timezone.utc)
    now_ist = now_utc.astimezone(timezone(timedelta(hours=5, minutes=30)))

    return {
        "updated_at": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "updated_at_ist": now_ist.strftime("%d %b %Y, %I:%M:%S %p IST"),
        "adaptive": adaptive,
        "v3": v3,
        "trend": trend,
        "amd": amd,
        "history_adaptive": adaptive.get("trade_history", []),
        "history_v3": v3.get("trade_history", []),
        "history_trend": trend.get("trade_history", []),
    }


def main() -> None:
    feed = build_feed()
    output = ROOT / "dashboard_data.json"
    output.write_text(json.dumps(feed, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
