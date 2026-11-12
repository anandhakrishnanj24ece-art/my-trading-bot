"""Paper-only FVG Duo signal engine.

Exact rules from the video-derived backtest:

Opening Drive (OD)
  * 10-minute FVGs checked at 09:40 and 09:50 New York.
  * Bull FVG: third low > first high.  Bear FVG: third high < first low.
  * Enter at the next 5-minute open; stop 1.5% from entry; no target.
  * One OD decision per day.  Once OD fires, no Trend entry that day.

Trend FVG
  * 15-minute bullish FVG, EMA50 > EMA100.
  * Middle candle range >= 1.5 x ATR14.
  * Long only; entry at the next 5-minute open, 09:30-15:30.
  * Stop one tick below the first FVG candle low; target 3R.

This module is pure signal logic. It has no broker, order, or network code.
"""
from __future__ import annotations

from datetime import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd

TZ = "America/New_York"
TICK = 0.01  # USATECH CFD tick; NQ research used 0.25
OD_STOP = 0.015
RR = 3.0


def normalize_5m(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a clean 5-minute OHLCV frame indexed in UTC."""
    if raw is None or raw.empty:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz="UTC"),
        )
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    rename = {
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    }
    df = df.rename(columns=rename)
    missing = [c for c in ("open", "high", "low", "close") if c not in df]
    if missing:
        raise ValueError(f"missing OHLC columns: {missing}")
    if "volume" not in df:
        df["volume"] = 0.0
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    cols = ["open", "high", "low", "close", "volume"]
    if "n_1m" in df.columns:
        cols.append("n_1m")
    df = df[cols].copy()

    idx = pd.to_datetime(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    df.index = idx
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def completed_5m(df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Drop the in-progress 5-minute candle (bars are left-labelled).

    When the frame came from the 1-minute Dukascopy feed it carries an
    ``n_1m`` coverage count; a bucket with fewer than five 1-minute bars is
    treated as incomplete even if wall-clock time has passed its close.
    """
    if df.empty:
        return df
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    out = df[df.index + pd.Timedelta(minutes=5) <= now]
    if "n_1m" in out.columns:
        out = out[out["n_1m"] >= 5]
    return out


def resample_complete(df5: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample and keep only buckets fully covered by completed 5m bars."""
    if df5.empty:
        return df5
    x = df5[["open", "high", "low", "close", "volume"]].resample(
        f"{minutes}min", label="left", closed="left"
    ).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    if x.empty:
        return x
    counts = df5["close"].resample(
        f"{minutes}min", label="left", closed="left"
    ).count()
    x["n_bars"] = counts.reindex(x.index).fillna(0).astype(int)
    last_end = df5.index[-1] + pd.Timedelta(minutes=5)
    expected = max(int(minutes // 5), 1)
    x = x[(x.index + pd.Timedelta(minutes=minutes) <= last_end)
          & (x["n_bars"] >= expected)]
    x["ny"] = x.index.tz_convert(TZ)
    x["hm"] = x["ny"].dt.strftime("%H:%M")
    return x


def atr_rma(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def fvg(df: pd.DataFrame, i: int):
    """Return (kind, first_low, first_high, middle_range)."""
    if i < 2:
        return None
    first, mid, third = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
    mid_range = float(mid["high"] - mid["low"])
    if third["low"] > first["high"]:
        return ("bull", float(first["low"]), float(first["high"]), mid_range)
    if third["high"] < first["low"]:
        return ("bear", float(first["low"]), float(first["high"]), mid_range)
    return None


def build_signals(df5: pd.DataFrame) -> Tuple[Dict[pd.Timestamp, int],
                                               Dict[pd.Timestamp, Tuple[int, float]]]:
    """Build OD and Trend entries keyed by the 5m timestamp they fill at."""
    od: Dict[pd.Timestamp, int] = {}
    trend: Dict[pd.Timestamp, Tuple[int, float]] = {}
    m10 = resample_complete(df5, 10)
    m15 = resample_complete(df5, 15)
    if m10.empty or m15.empty:
        return od, trend

    m15 = m15.copy()
    m15["ema50"] = m15["close"].ewm(span=50, adjust=False).mean()
    m15["ema100"] = m15["close"].ewm(span=100, adjust=False).mean()
    m15["atr"] = atr_rma(m15, 14)

    for i in range(2, len(m10)):
        r = m10.iloc[i]
        close_hm = (r["ny"] + pd.Timedelta(minutes=10)).strftime("%H:%M")
        if close_hm not in ("09:40", "09:50"):
            continue
        g = fvg(m10, i)
        if g:
            od[m10.index[i] + pd.Timedelta(minutes=10)] = (
                1 if g[0] == "bull" else -1
            )

    for i in range(2, len(m15)):
        r = m15.iloc[i]
        close_hm = (r["ny"] + pd.Timedelta(minutes=15)).strftime("%H:%M")
        if not ("09:30" <= close_hm <= "15:30"):
            continue
        if not (r["ema50"] > r["ema100"]):
            continue
        g = fvg(m15, i)
        if not g or g[0] != "bull":
            continue
        if not np.isfinite(r["atr"]) or g[3] < 1.5 * r["atr"]:
            continue
        stop = g[1] - TICK
        if stop > 0:
            trend[m15.index[i] + pd.Timedelta(minutes=15)] = (1, stop)

    return od, trend


def signal_day(ts: pd.Timestamp) -> object:
    return pd.Timestamp(ts).tz_convert(TZ).date()


def in_entry_window(now: pd.Timestamp) -> bool:
    hm = now.tz_convert(TZ).time()
    return time(9, 30) <= hm <= time(15, 30)
