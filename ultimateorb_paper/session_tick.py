"""One 6-minute tick of the UltimateORB paper proxy session loop.

The GitHub workflow calls this in a shell loop from about 08:55 CT until the
paper session closes. Exit code 0 means continue; 2 means the session is over;
1 means an unexpected error (the workflow stops the loop).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, time
from zoneinfo import ZoneInfo

from .auto_paper import SESSION_CLOSE, _decision
from .paper_engine import PaperEngine

CT = ZoneInfo("America/Chicago")
SESSION_START = time(8, 55)
HARD_END = time(15, 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="UltimateORB paper session tick")
    parser.add_argument("--base", default=None)
    parser.add_argument("--now", default=None, help="ISO timestamp, for local testing only")
    args = parser.parse_args(argv)

    now = datetime.fromisoformat(args.now) if args.now else datetime.now(CT)
    if now.tzinfo is None:
        now = now.replace(tzinfo=CT)
    now = now.astimezone(CT)

    if now.weekday() >= 5:
        print(json.dumps({"time_ct": now.isoformat(), "action": "stop", "reason": "weekend"}))
        return 2
    if now.time() < SESSION_START or now.time() > HARD_END:
        print(json.dumps({"time_ct": now.isoformat(), "action": "stop", "reason": "outside session"}))
        return 2

    engine = PaperEngine(base_dir=args.base)
    result = _decision(engine, now, dry_run=False)
    result["loop"] = "continue" if now.time() < SESSION_CLOSE else "stop"
    print(json.dumps(result, indent=2, default=str))
    return 0 if now.time() < SESSION_CLOSE else 2


if __name__ == "__main__":
    raise SystemExit(main())
