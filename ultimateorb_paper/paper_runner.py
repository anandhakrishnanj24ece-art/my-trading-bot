"""CLI for the paper-only UltimateORB ledger.

Examples
--------
    python -m ultimateorb_paper.paper_runner status
    python -m ultimateorb_paper.paper_runner halt
    python -m ultimateorb_paper.paper_runner resume
    python -m ultimateorb_paper.paper_runner open --direction long --price 28000
    python -m ultimateorb_paper.paper_runner mark --high 28012 --low 27988 --close 28005
    python -m ultimateorb_paper.paper_runner close --price 28004
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .paper_engine import PaperEngine, TradingHalted


def build_engine(base: str | None) -> PaperEngine:
    return PaperEngine(base_dir=Path(base) if base else None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="UltimateORB paper-only ledger")
    parser.add_argument("--base", default=None, help="folder containing config.json/state")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="show paper state")
    sub.add_parser("halt", help="block new paper entries")
    sub.add_parser("resume", help="allow new paper entries")

    open_p = sub.add_parser("open", help="open a paper trade")
    open_p.add_argument("--direction", required=True, choices=["long", "short"])
    open_p.add_argument("--price", required=True, type=float)
    open_p.add_argument("--timestamp", default=None)
    open_p.add_argument("--target", type=float, default=None)
    open_p.add_argument("--stop", type=float, default=None)
    open_p.add_argument("--contracts", type=int, default=None)

    mark_p = sub.add_parser("mark", help="mark a bar; closes target/stop if touched")
    mark_p.add_argument("--high", required=True, type=float)
    mark_p.add_argument("--low", required=True, type=float)
    mark_p.add_argument("--close", required=True, type=float)
    mark_p.add_argument("--timestamp", default=None)

    close_p = sub.add_parser("close", help="close the open paper trade at a price")
    close_p.add_argument("--price", required=True, type=float)
    close_p.add_argument("--reason", default="manual")
    close_p.add_argument("--timestamp", default=None)

    export_p = sub.add_parser("export", help="write the closed-trade CSV")
    export_p.add_argument("--path", default=None)

    args = parser.parse_args(argv)
    engine = build_engine(args.base)

    try:
        if args.command == "status":
            print(json.dumps(engine.snapshot(), indent=2, default=str))
        elif args.command == "halt":
            print(json.dumps(engine.halt(), indent=2))
        elif args.command == "resume":
            print(json.dumps(engine.resume(), indent=2))
        elif args.command == "open":
            position = engine.open_trade(
                direction=args.direction,
                entry_price=args.price,
                timestamp=args.timestamp,
                target_points=args.target,
                stop_points=args.stop,
                contracts=args.contracts,
            )
            print(json.dumps(position, indent=2))
        elif args.command == "mark":
            print(json.dumps(engine.mark(args.high, args.low, args.close, args.timestamp), indent=2))
        elif args.command == "close":
            print(json.dumps(engine.close_manual(args.price, args.reason, args.timestamp), indent=2))
        elif args.command == "export":
            print(engine.export_trades(args.path))
    except TradingHalted as exc:
        print(f"HALTED: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
