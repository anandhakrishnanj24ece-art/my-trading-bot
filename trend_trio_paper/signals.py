"""Trend Trio signal logic (paper-only).

The video uses five free TradingView scripts:

* SuperTrend by Kivanc Ozbilgic - 5m signal, 15m direction filter.
* Ultimate Moving Average by Chris Moody - 4m turn green, 15m Sling Shot
  38/62 EMA filter, 4m Chandelier Exit turn down.
* AlphaTrend by Kivanc Ozbilgic - 2m cross + 2m ADX14 >= 20 (09:30-11:30),
  and 1h cross + 1h EMA200 filter (10:00-15:00).

The video does not print every input, so the open-source defaults are used:
SuperTrend 10/3.0 with RMA ATR, Ultimate MA = SMA20 with 2-bar colour
smoothing, Sling Shot EMA38/EMA62, Chandelier 22/3.0 using closes, and
AlphaTrend coefficient 1.0 with period 14 and MFI.

A signal becomes actionable when its higher-timeframe candle closes.  The
paper engine enters on the next 1-minute observation.  Long/short entries are
long-only for SuperTrend and Ultimate MA; both AlphaTrend setups can short.
This module is pure signal logic: no network, broker, or order code.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Iterable, List

import numpy as np
import pandas as pd

TZ = "America/New_York"
TICK = 0.01

# Video/source defaults.  These are intentionally explicit and frozen.
ST_ATR_PERIOD = 10
ST_MULTIPLIER = 3.0
UMA_LENGTH = 20
UMA_COLOR_SMOOTH = 2
SLING_FAST = 38
SLING_SLOW = 62
CE_LENGTH = 22
CE_MULTIPLIER = 3.0
AT_PERIOD = 14
AT_MULTIPLIER = 1.0
ADX_LENGTH = 14
ADX_MIN = 20.0
EMA_FILTER_LENGTH = 200

MAX_ENTRY_DELAY_SEC = 330
PRIORITY = {"st": 0, "uma": 1, "at2": 2, "at1h": 3}

_WINDOW = {
    "st": (time(9, 30), time(15, 30)),
    "uma": (time(0, 0), time(23, 59)),
    "at2": (time(9, 30), time(11, 30)),
    "at1h": (time(10, 0), time(15, 0)),
}


@dataclass(frozen=True)
class Entry:
    """A higher-timeframe entry event.

    ``signal_time`` is the candle close at which the signal is known.
    ``risk_points`` is zero for setups with no initial stop.
    """

    signal_time: pd.Timestamp
    source: str
    direction: int
    risk_points: float
    detail: str = ""


@dataclass(frozen=True)
class Exit:
    """A higher-timeframe exit event for a specific position direction."""

    signal_time: pd.Timestamp
    source: str
    for_direction: int
    reason: str


def normalize_1m(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a clean 1-minute OHLCV frame indexed in UTC."""
    if raw is None or raw.empty:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz="UTC"),
        )
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
        "timestamp ET": "timestamp",
    })
    missing = [c for c in ("open", "high", "low", "close") if c not in df]
    if missing:
        raise ValueError(f"missing OHLC columns: {missing}")
    if "volume" not in df:
        df["volume"] = 0.0
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    cols = ["open", "high", "low", "close", "volume"]
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


def completed_1m(df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Drop the in-progress 1-minute candle (bars are left-labelled)."""
    if df.empty:
        return df
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    now = now.tz_convert("UTC")
    return df[df.index + pd.Timedelta(minutes=1) <= now]


def resample_1m(df1: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample 1m bars and keep only fully covered buckets."""
    if df1.empty:
        return df1
    x = df1[["open", "high", "low", "close", "volume"]].resample(
        f"{minutes}min", label="left", closed="left"
    ).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    if x.empty:
        return x
    counts = df1["close"].resample(
        f"{minutes}min", label="left", closed="left"
    ).count()
    x["n_1m"] = counts.reindex(x.index).fillna(0).astype(int)
    last_end = df1.index[-1] + pd.Timedelta(minutes=1)
    # Small buckets must be complete.  Longer buckets tolerate a couple of
    # missing CFD minutes so a single gap does not erase every 1h bar.
    min_bars = minutes if minutes <= 5 else max(minutes - 2,
                                                int(minutes * 0.9))
    x = x[(x.index + pd.Timedelta(minutes=minutes) <= last_end)
          & (x["n_1m"] >= min_bars)]
    x["ny"] = x.index.tz_convert(TZ)
    x["hm"] = x["ny"].dt.strftime("%H:%M")
    return x


def _rma_series(series: pd.Series, n: int) -> pd.Series:
    """Pine-style RMA with an SMA seed."""
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    if len(arr) < n:
        return pd.Series(out, index=series.index)
    seed = np.nanmean(arr[:n])
    if not np.isfinite(seed):
        return pd.Series(out, index=series.index)
    out[n - 1] = seed
    for i in range(n, len(arr)):
        prev = out[i - 1]
        val = arr[i]
        if np.isfinite(val) and np.isfinite(prev):
            out[i] = (prev * (n - 1) + val) / n
        else:
            out[i] = prev
    return pd.Series(out, index=series.index)


def true_range(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    return pd.concat(
        [h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1
    ).max(axis=1)


def atr_rma(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return _rma_series(true_range(df), n)


def supertrend(df: pd.DataFrame, period: int = ST_ATR_PERIOD,
               multiplier: float = ST_MULTIPLIER) -> pd.DataFrame:
    """Kivanc-style SuperTrend.

    Returns ``st``, ``trend`` (1 up, -1 down), ``flip_up``, ``flip_down`` and
    ``atr`` columns.
    """
    out = pd.DataFrame(index=df.index)
    if df.empty:
        out["st"] = pd.Series(dtype=float)
        out["trend"] = pd.Series(dtype=int)
        out["flip_up"] = pd.Series(dtype=bool)
        out["flip_down"] = pd.Series(dtype=bool)
        out["atr"] = pd.Series(dtype=float)
        return out

    atr = atr_rma(df, period).to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    hl2 = (high + low) / 2.0
    src = hl2

    up = hl2 - multiplier * atr
    dn = hl2 + multiplier * atr
    up_final = np.full(len(df), np.nan)
    dn_final = np.full(len(df), np.nan)
    trend = np.ones(len(df), dtype=int)
    st = np.full(len(df), np.nan)

    for i in range(len(df)):
        if not np.isfinite(atr[i]):
            continue
        if i == 0 or not np.isfinite(up_final[i - 1]):
            up_final[i] = up[i]
            dn_final[i] = dn[i]
            trend[i] = 1
            st[i] = up[i]
            continue

        up_final[i] = (max(up[i], up_final[i - 1])
                       if src[i - 1] > up_final[i - 1] else up[i])
        dn_final[i] = (min(dn[i], dn_final[i - 1])
                       if src[i - 1] < dn_final[i - 1] else dn[i])

        t = int(trend[i - 1])
        if t == -1 and src[i] > dn_final[i - 1]:
            t = 1
        elif t == 1 and src[i] < up_final[i - 1]:
            t = -1
        trend[i] = t
        st[i] = up_final[i] if t == 1 else dn_final[i]

    out["st"] = st
    out["trend"] = trend
    out["flip_up"] = (out["trend"] == 1) & (out["trend"].shift(1) == -1)
    out["flip_down"] = (out["trend"] == -1) & (out["trend"].shift(1) == 1)
    out["atr"] = atr
    return out


def ultimate_ma(df: pd.DataFrame, length: int = UMA_LENGTH,
                smooth: int = UMA_COLOR_SMOOTH) -> pd.DataFrame:
    """Chris Moody Ultimate MA: SMA20, green when it rises over 2 bars."""
    out = pd.DataFrame(index=df.index)
    if df.empty:
        out["ma"] = pd.Series(dtype=float)
        out["up"] = pd.Series(dtype=bool)
        out["turn_green"] = pd.Series(dtype=bool)
        return out
    ma = df["close"].rolling(length, min_periods=length).mean()
    up = ma >= ma.shift(smooth)
    up = up.fillna(False)
    turn_green = up & ~up.shift(1, fill_value=False)
    out["ma"] = ma
    out["up"] = up
    out["turn_green"] = turn_green
    return out


def sling_shot(df: pd.DataFrame, fast: int = SLING_FAST,
               slow: int = SLING_SLOW) -> pd.DataFrame:
    """Sling Shot filter: fast EMA above slow EMA."""
    out = pd.DataFrame(index=df.index)
    if df.empty:
        out["ema_fast"] = pd.Series(dtype=float)
        out["ema_slow"] = pd.Series(dtype=float)
        out["long"] = pd.Series(dtype=bool)
        return out
    fast_ema = df["close"].ewm(span=fast, adjust=False).mean()
    slow_ema = df["close"].ewm(span=slow, adjust=False).mean()
    out["ema_fast"] = fast_ema
    out["ema_slow"] = slow_ema
    out["long"] = fast_ema > slow_ema
    return out


def chandelier_exit(df: pd.DataFrame, length: int = CE_LENGTH,
                    multiplier: float = CE_MULTIPLIER) -> pd.DataFrame:
    """everget Chandelier Exit, using closes for the extremums."""
    out = pd.DataFrame(index=df.index)
    if df.empty:
        out["dir"] = pd.Series(dtype=int)
        out["turn_down"] = pd.Series(dtype=bool)
        return out
    atr = atr_rma(df, length).to_numpy(dtype=float) * multiplier
    close = df["close"].to_numpy(dtype=float)
    highest = df["close"].rolling(length, min_periods=length).max().to_numpy()
    lowest = df["close"].rolling(length, min_periods=length).min().to_numpy()

    long_stop = highest - atr
    short_stop = lowest + atr
    direction = np.ones(len(df), dtype=int)

    for i in range(len(df)):
        if i == 0 or not np.isfinite(long_stop[i - 1]):
            continue
        lp = long_stop[i - 1]
        sp = short_stop[i - 1]
        if close[i - 1] > lp:
            long_stop[i] = max(long_stop[i], lp)
        if close[i - 1] < sp:
            short_stop[i] = min(short_stop[i], sp)
        if close[i] > sp:
            direction[i] = 1
        elif close[i] < lp:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

    out["dir"] = direction
    out["turn_down"] = (out["dir"] == -1) & (out["dir"].shift(1) == 1)
    return out


def money_flow_index(df: pd.DataFrame, n: int = AT_PERIOD) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    money = tp * df["volume"]
    pos = money.where(tp > tp.shift(1), 0.0)
    neg = money.where(tp < tp.shift(1), 0.0)
    pos_sum = pos.rolling(n, min_periods=n).sum()
    neg_sum = neg.rolling(n, min_periods=n).sum()
    ratio = pos_sum / neg_sum.replace(0.0, np.nan)
    mfi = 100.0 - 100.0 / (1.0 + ratio)
    mfi = mfi.where(neg_sum != 0, 100.0)
    mfi = mfi.where(pos_sum != 0, 0.0)
    return mfi


def alpha_trend(df: pd.DataFrame, period: int = AT_PERIOD,
                multiplier: float = AT_MULTIPLIER) -> pd.DataFrame:
    """Kivanc AlphaTrend with MFI, plus its buy/sell crosses."""
    out = pd.DataFrame(index=df.index)
    if df.empty:
        out["at"] = pd.Series(dtype=float)
        out["atr"] = pd.Series(dtype=float)
        out["buy"] = pd.Series(dtype=bool)
        out["sell"] = pd.Series(dtype=bool)
        return out

    atr = true_range(df).rolling(period, min_periods=period).mean()
    up_t = df["low"] - atr * multiplier
    down_t = df["high"] + atr * multiplier
    mfi = money_flow_index(df, period)

    at = np.zeros(len(df), dtype=float)
    up_arr = up_t.to_numpy(dtype=float)
    down_arr = down_t.to_numpy(dtype=float)
    mfi_arr = mfi.to_numpy(dtype=float)
    for i in range(len(df)):
        prev = at[i - 1] if i > 0 else 0.0
        if not np.isfinite(up_arr[i]) or not np.isfinite(down_arr[i]):
            at[i] = prev
            continue
        if np.isfinite(mfi_arr[i]) and mfi_arr[i] >= 50.0:
            at[i] = prev if up_arr[i] < prev else up_arr[i]
        else:
            at[i] = prev if down_arr[i] > prev else down_arr[i]

    s = pd.Series(at, index=df.index)
    crossed_up = (s > s.shift(2)) & (s.shift(1) <= s.shift(3))
    crossed_down = (s < s.shift(2)) & (s.shift(1) >= s.shift(3))
    out["at"] = s
    out["atr"] = atr
    out["buy"] = crossed_up.fillna(False)
    out["sell"] = crossed_down.fillna(False)
    return out


def adx(df: pd.DataFrame, n: int = ADX_LENGTH) -> pd.Series:
    """Wilder ADX."""
    if df.empty:
        return pd.Series(dtype=float)
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=df.index
    )
    tr = true_range(df)
    atr = _rma_series(tr, n)
    plus_di = 100.0 * _rma_series(plus_dm, n) / atr.replace(0.0, np.nan)
    minus_di = 100.0 * _rma_series(minus_dm, n) / atr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (
        plus_di + minus_di
    ).replace(0.0, np.nan)
    return _rma_series(dx.fillna(0.0), n)


def _asof_values(source: pd.DataFrame, column: str, minutes: int,
                 targets: pd.Series) -> np.ndarray:
    """Return the last completed source value at each target time."""
    if source.empty or len(targets) == 0:
        return np.full(len(targets), np.nan)
    s = source[[column]].copy()
    s["_available_at"] = s.index + pd.Timedelta(minutes=minutes)
    s = s.sort_values("_available_at").drop_duplicates(
        "_available_at", keep="last"
    )
    left = pd.DataFrame({"_target": pd.to_datetime(targets)}).sort_values(
        "_target"
    )
    merged = pd.merge_asof(
        left, s, left_on="_target", right_on="_available_at",
        direction="backward",
    )
    return merged[column].to_numpy(dtype=float)


def _asof_bool(source: pd.DataFrame, column: str, minutes: int,
               targets: pd.Series) -> np.ndarray:
    vals = _asof_values(source, column, minutes, targets)
    return np.array([bool(v) if np.isfinite(v) else False for v in vals])


def _in_window(source: str, ts: pd.Timestamp) -> bool:
    hm = pd.Timestamp(ts).tz_convert(TZ).time()
    lo, hi = _WINDOW[source]
    return lo <= hm <= hi


def build_entries(df1: pd.DataFrame) -> List[Entry]:
    """Build all video-derived entry events from completed 1m bars."""
    if df1.empty:
        return []
    m2 = resample_1m(df1, 2)
    m4 = resample_1m(df1, 4)
    m5 = resample_1m(df1, 5)
    m15 = resample_1m(df1, 15)
    m60 = resample_1m(df1, 60)
    if m2.empty or m4.empty or m5.empty or m15.empty or m60.empty:
        return []

    entries: List[Entry] = []

    # ---- Setup 1: SuperTrend 5m, filtered by 15m SuperTrend.
    st5 = supertrend(m5)
    st15 = supertrend(m15)
    sig = st5[st5["flip_up"]].copy()
    if not sig.empty:
        times = sig.index + pd.Timedelta(minutes=5)
        filt = _asof_values(st15, "trend", 15, pd.Series(times))
        for t, (_, row), f in zip(times, sig.iterrows(), filt):
            if not _in_window("st", t):
                continue
            if not (np.isfinite(f) and f == 1):
                continue
            risk = 2.0 * float(row["atr"])
            if np.isfinite(risk) and risk > 0:
                entries.append(Entry(pd.Timestamp(t), "st", 1, risk,
                                     "SuperTrend 5m flip up / 15m green"))

    # ---- Setup 2: Ultimate MA 4m turn green, Sling Shot 15m filter.
    uma4 = ultimate_ma(m4)
    sling15 = sling_shot(m15)
    sig = uma4[uma4["turn_green"]].copy()
    if not sig.empty:
        times = sig.index + pd.Timedelta(minutes=4)
        filt = _asof_bool(sling15, "long", 15, pd.Series(times))
        for t, f in zip(times, filt):
            if f:
                entries.append(Entry(pd.Timestamp(t), "uma", 1, 0.0,
                                     "Ultimate MA 4m turns green / EMA38>62"))

    # ---- Setup 3: AlphaTrend 2m + ADX14 >= 20, 09:30-11:30.
    at2 = alpha_trend(m2)
    adx2 = adx(m2, ADX_LENGTH)
    for direction, col in ((1, "buy"), (-1, "sell")):
        sig = at2[at2[col]].copy()
        if sig.empty:
            continue
        times = sig.index + pd.Timedelta(minutes=2)
        adx_vals = _asof_values(
            pd.DataFrame({"adx": adx2}), "adx", 2, pd.Series(times)
        )
        for t, av in zip(times, adx_vals):
            if not _in_window("at2", t):
                continue
            if np.isfinite(av) and av >= ADX_MIN:
                entries.append(Entry(
                    pd.Timestamp(t), "at2", direction, 0.0,
                    f"AlphaTrend 2m {'buy' if direction == 1 else 'sell'} "
                    f"/ ADX {av:.1f}",
                ))

    # ---- Setup 4: AlphaTrend 1h + 1h EMA200, 10:00-15:00.
    at60 = alpha_trend(m60)
    ema200 = m60["close"].ewm(
        span=EMA_FILTER_LENGTH, adjust=False,
        min_periods=EMA_FILTER_LENGTH,
    ).mean()
    for direction, col in ((1, "buy"), (-1, "sell")):
        sig = at60[at60[col]].copy()
        if sig.empty:
            continue
        times = sig.index + pd.Timedelta(minutes=60)
        e200 = _asof_values(
            pd.DataFrame({"ema200": ema200}), "ema200", 60,
            pd.Series(times),
        )
        closes = _asof_values(m60, "close", 60, pd.Series(times))
        for t, ev, px, (_, row) in zip(
            times, e200, closes, sig.iterrows()
        ):
            if not _in_window("at1h", t):
                continue
            if not np.isfinite(ev) or not np.isfinite(px):
                continue
            if direction == 1 and not (px > ev):
                continue
            if direction == -1 and not (px < ev):
                continue
            risk = 2.0 * float(row["atr"])
            if np.isfinite(risk) and risk > 0:
                entries.append(Entry(
                    pd.Timestamp(t), "at1h", direction, risk,
                    f"AlphaTrend 1h {'buy' if direction == 1 else 'sell'} "
                    f"/ EMA200 filter",
                ))

    entries.sort(key=lambda e: (e.signal_time, PRIORITY.get(e.source, 9)))
    return entries


def build_exits(df1: pd.DataFrame) -> List[Exit]:
    """Build higher-timeframe exit events from completed 1m bars."""
    if df1.empty:
        return []
    m2 = resample_1m(df1, 2)
    m4 = resample_1m(df1, 4)
    m5 = resample_1m(df1, 5)
    m60 = resample_1m(df1, 60)
    if m2.empty or m4.empty or m5.empty or m60.empty:
        return []

    exits: List[Exit] = []

    st5 = supertrend(m5)
    for ts in st5.index[st5["flip_down"]]:
        exits.append(Exit(ts + pd.Timedelta(minutes=5), "st", 1,
                          "SuperTrend 5m flip down"))

    ce4 = chandelier_exit(m4)
    for ts in ce4.index[ce4["turn_down"]]:
        exits.append(Exit(ts + pd.Timedelta(minutes=4), "uma", 1,
                          "Chandelier 4m turns down"))

    at2 = alpha_trend(m2)
    for ts in at2.index[at2["sell"]]:
        exits.append(Exit(ts + pd.Timedelta(minutes=2), "at2", 1,
                          "AlphaTrend 2m opposite signal"))
    for ts in at2.index[at2["buy"]]:
        exits.append(Exit(ts + pd.Timedelta(minutes=2), "at2", -1,
                          "AlphaTrend 2m opposite signal"))

    at60 = alpha_trend(m60)
    for ts in at60.index[at60["sell"]]:
        exits.append(Exit(ts + pd.Timedelta(minutes=60), "at1h", 1,
                          "AlphaTrend 1h opposite signal"))
    for ts in at60.index[at60["buy"]]:
        exits.append(Exit(ts + pd.Timedelta(minutes=60), "at1h", -1,
                          "AlphaTrend 1h opposite signal"))

    exits.sort(key=lambda e: e.signal_time)
    return exits


def scan_entry(entries: Iterable[Entry], now: pd.Timestamp,
               after: pd.Timestamp | None = None) -> Entry | None:
    """Return the earliest still-fresh signal, or None."""
    now = pd.Timestamp(now)
    candidates = []
    for e in entries:
        if e.signal_time > now:
            continue
        if after is not None and e.signal_time < after:
            continue
        age = (now - e.signal_time).total_seconds()
        if age < 0 or age > MAX_ENTRY_DELAY_SEC:
            continue
        candidates.append(e)
    if not candidates:
        return None
    candidates.sort(key=lambda e: (e.signal_time, PRIORITY.get(e.source, 9)))
    return candidates[0]
