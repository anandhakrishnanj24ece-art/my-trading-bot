"""GitHub Actions paper tick for FVG Duo.

Usage:
    python -m fvg_duo_paper.runner
    python -m fvg_duo_paper.runner --dry-run
    python -m fvg_duo_paper.runner --heartbeat
    python -m fvg_duo_paper.runner --bars data/NASDAQ_DUKA_5y/bars_5m.parquet \\
        --now 2026-09-15T10:00:00-04:00

``--heartbeat`` writes only the freshness timestamp (``asof``); it never reads
data and never opens, closes, marks, or resizes a position.  Normal post-close
ticks also write one heartbeat so the book can tell a quiet runner from one
that stopped days ago.

Live data is the Dukascopy USATECH.IDX/USD Nasdaq-100 CFD 1-minute feed,
resampled to 5 minutes.  Free NQ futures feeds are delayed by roughly ten
minutes, which would make the 09:40/09:50 Opening Drive untradeable.

Paper-only. No broker, no API keys, no live orders.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .paper_engine import PAPER_ONLY, PaperEngine
from .signals import TZ, build_signals, completed_5m, normalize_5m

TICKER = "USATECH.IDX/USD"
FETCH_DAYS = 10
DOWNLOAD_ATTEMPTS = 3


def download() -> pd.DataFrame:
    """Fetch fresh 1m Nasdaq-100 CFD bars and resample to 5m.

    Dukascopy's HTTP feed occasionally drops a chunk mid-response
    (ChunkedEncodingError / IncompleteRead).  Retry a few times with a
    short back-off so one bad tick does not leave the paper state stale.
    """
    import dukascopy_python as d
    from dukascopy_python.instruments import I_USATECH_IDX_USD

    logging.getLogger("dukascopy").setLevel(logging.ERROR)
    last_exc: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
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
        except Exception as exc:  # noqa: BLE001 - network/transport errors
            last_exc = exc
            if attempt < DOWNLOAD_ATTEMPTS:
                time.sleep(2 ** (attempt - 1) * 2)
    assert last_exc is not None
    raise last_exc


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


def heartbeat(now: datetime | pd.Timestamp | None = None,
              dry_run: bool = False,
              base_dir: str | None = None) -> dict:
    """Write a fresh state timestamp without reading data or trading."""
    if not PAPER_ONLY:
        raise RuntimeError("paper-only guard failed")
    engine = PaperEngine(base_dir=base_dir)
    return engine.heartbeat(now=now, dry_run=dry_run)


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

    try:
        raw = load_bars(bars_path) if bars_path else download()
    except Exception as exc:  # noqa: BLE001
        logging.exception("paper data download failed")
        engine = PaperEngine(base_dir=base_dir)
        result = engine.heartbeat(now=now_ts, dry_run=dry_run)
        engine.state["last_decision"] = "data_unavailable"
        engine.state["last_reason"] = f"Dukascopy download failed: {exc}"
        if not dry_run:
            engine.save()
        result["action"] = "none"
        result["reason"] = engine.state["last_reason"]
        result["stats"] = engine.stats()
        return result

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
    parser.add_argument("--heartbeat", action="store_true",
                        help="write a freshness timestamp only; no data read")
    args = parser.parse_args()

    now = None
    if args.now:
        now = datetime.fromisoformat(args.now)
    try:
        if args.heartbeat:
            result = heartbeat(now=now, dry_run=args.dry_run, base_dir=args.base)
        else:
            result = run_tick(now=now, dry_run=args.dry_run,
                              base_dir=args.base, bars_path=args.bars)
    except Exception as exc:  # noqa: BLE001
        logging.exception("paper runner tick failed")
        result = {"action": "error", "reason": str(exc), "trade": None,
                  "position": None, "stats": None}
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
