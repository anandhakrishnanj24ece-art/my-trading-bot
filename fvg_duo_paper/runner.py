"""GitHub Actions paper tick for FVG Duo.

Usage:
    python -m fvg_duo_paper.runner
    python -m fvg_duo_paper.runner --dry-run
    python -m fvg_duo_paper.runner --bars data/NASDAQ_DUKA_5y/bars_5m.parquet \
        --now 2026-09-15T10:00:00-04:00

Live data is the Dukascopy USATECH.IDX/USD Nasdaq-100 CFD 1-minute feed,
resampled to 5 minutes.  Free NQ futures feeds are delayed by roughly ten
minutes, which would make the 09:40/09:50 Opening Drive untradeable.

Paper-only. No broker, no API keys, no live orders.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .paper_engine import PAPER_ONLY, PaperEngine
from .signals import TZ, build_signals, completed_5m, normalize_5m

TICKER = "USATECH.IDX/USD"
FETCH_DAYS = 10


def download() -> pd.DataFrame:
    """Fetch fresh 1m Nasdaq-100 CFD bars and resample to 5m."""
    import dukascopy_python as d
    from dukascopy_python.instruments import I_USATECH_IDX_USD

    logging.getLogger("dukascopy").setLevel(logging.ERROR)
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=FETCH_DAYS)
    rows = list(d.fetch_candles(
        I_USATECH_IDX_USD, 1, d.TimeUnit.MINUTE, d.OfferSide.BID,
        start, now,
    ))
    if not rows:
        raise RuntimeError("Dukascopy returned no USATECH 1m data")

    df = pd.DataFrame([{
        "timestamp": r.timestamp,
        "open": r.open, "high": r.high, "low": r.low,
        "close": r.close, "volume": r.volume,
    } for r in rows])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    bars = df.resample("5min", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna(subset=["open", "high", "low", "close"])
    counts = df["close"].resample(
        "5min", label="left", closed="left"
    ).count()
    bars["n_1m"] = counts.reindex(bars.index).fillna(0).astype(int)
    return bars


def load_bars(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        raw = pd.read_parquet(p)
    else:
        raw = pd.read_csv(p)
    if "timestamp" in raw.columns:
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
        raw = raw.set_index("timestamp")
    return raw


def run_tick(now: datetime | pd.Timestamp | None = None,
             dry_run: bool = False,
             base_dir: str | None = None,
             bars_path: str | None = None) -> dict:
    if not PAPER_ONLY:
        raise RuntimeError("paper-only guard failed")
    if now is None:
        now = datetime.now(ZoneInfo(TZ))
    now_ts = pd.Timestamp(now)
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize(TZ)
    now_utc = now_ts.tz_convert("UTC")

    raw = load_bars(bars_path) if bars_path else download()
    bars = normalize_5m(raw)
    completed = completed_5m(bars, now_utc)
    signals = build_signals(completed)

    engine = PaperEngine(base_dir=base_dir)
    return engine.tick(now_ts, bars, signals=signals, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="FVG Duo paper runner")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--now", default=None,
                        help="ISO timestamp for replay/testing")
    parser.add_argument("--base", default=None,
                        help="state directory (default: fvg_duo_paper/state)")
    parser.add_argument("--bars", default=None,
                        help="optional parquet/CSV file instead of Dukascopy")
    args = parser.parse_args()

    now = None
    if args.now:
        now = datetime.fromisoformat(args.now)
    result = run_tick(now=now, dry_run=args.dry_run,
                      base_dir=args.base, bars_path=args.bars)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
