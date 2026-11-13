"""LVN/HVN - FVG Confluence Reversion — research backtester (v1).

Implements the rule set:

  HTF zone: unfilled FVG / IFVG that aligns with a low-volume node (LVN),
  optionally with HVN as a target magnet, BOS/CHoCH, delta divergence at
  formation, a large print inside the gap, absorption and order-flow imbalance.

  LTF entry: price retests the gap/LVN; confirmation requires absorption,
  order-flow imbalance, delta divergence at the retest, CISD, OI and depth
  support. Entry on the close of the confirmation bar (next bar open).

  SL: beyond the gap / order-block low (or your half-candle rule).
  TP: just inside the far edge of the FVG (top for a green gap).

  Divergence is tracked at three stages: formation, retest and reversion.

Paper/research only. No broker adapter, no live orders.

Usage:
  python lvn_fvg_strategy.py --data data/BTCUSDT_2026-10-11_2026-11-09 \
      --tf 15m --profile-window 288
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TF_BASE_POINTS = {
    "1m": 1, "3m": 1.5, "5m": 2, "10m": 2.5,
    "1h": 4, "2h": 5, "4h": 6, "8h": 7,
    "12h": 8, "1d": 10, "2d": 11,
}


@dataclass
class Config:
    tf: str = "15m"
    profile_window: int = 2016      # bars used for the volume profile
    profile_tick: float = 50.0      # price bucket for the rolling profile
    swing_k: int = 3                # bars each side to confirm a swing
    fvg_min_atr: float = 0.15       # minimum gap size in ATR
    fvg_max_age: int = 500          # bars a gap stays tradeable
    lvn_pct: float = 0.20
    hvn_pct: float = 0.80
    align_min: float = 0.50         # min overlap of gap with LVN range
    div_lookback: int = 60          # bars to look for the prior swing
    div_mode: str = "window"        # "window" | "swing"
    div_window: int = 60            # bars used by the window detector
    div_min_sep: int = 5            # min bars between the two price extremes
    abs_body_frac: float = 0.60     # body/range below this = absorption candle
    abs_vol_mult: float = 1.30      # volume vs rolling mean
    imb_thresh: float = 0.20        # |buy-sell|/vol
    big_trade_pct: float = 0.99     # percentile of per-price volume in gap
    min_score: float = 12.0
    require_formation_div: bool = False
    require_retest_div: bool = True
    require_absorb: bool = False    # absorption must be present in the confirm window
    min_div_stages: int = 2         # formation + retest (reversion is post-entry)
    confirm_window: int = 5         # bars around the zone touch allowed to confirm
    entry_mode: str = "after"       # after | anticipate
    entry_proximity_atr: float = 0.5  # "about to retest" distance in ATR
    confirm_from_retest: bool = True  # confirm only from the first gap touch
    retest_window: int = 5            # bars from first touch to confirm
    use_footprint: bool = True        # per-price footprint inside the gap
    fp_imb_min: float = 0.15          # footprint imbalance threshold
    use_heatmap: bool = True
    wall_near_pct: float = 2.5      # wall counts as "at price" within this %
    band_imb_min: float = 0.15      # bid/ask band imbalance to count as support
    depth_pull_thresh: float = 0.20 # 20% depth drop in lookback = pulled
    wall_block_imb: float = 0.20    # opposite band imbalance that blocks target
    wall_drop_thresh: float = 0.15  # wall pulled by this fraction = sweep
    wall_lookback: int = 3          # bars for wall change
    sl_mode: str = "halflow"        # "ob_low" | "halflow"
    sl_buffer_atr: float = 0.25
    tp_mode: str = "next_close"      # "next_close" | "fvg_far" | "r"
    tp_r: float = 2.0                # used when tp_mode == "r"
    min_rr: float = 0.75
    time_stop_bars: int = 96
    risk_pct: float = 0.005
    min_stop_atr: float = 0.50      # minimum stop distance in ATR
    max_notional_mult: float = 5.0  # cap position notional vs equity
    fee_bps: float = 5.0            # per side, 0.05%
    slip_bps: float = 1.0           # per side
    starting_equity: float = 10_000.0
    allow_longs: bool = True
    allow_shorts: bool = True


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_data(root: str, tf: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    bars = pq.read_table(os.path.join(root, f"bars_{tf}.parquet")).to_pandas()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values("timestamp").reset_index(drop=True)

    vp = pq.read_table(os.path.join(root, f"volume_profile_{tf}.parquet")).to_pandas()
    vp["timestamp"] = pd.to_datetime(vp["timestamp"], utc=True)
    return bars, vp


# --------------------------------------------------------------------------
# indicators / structure
# --------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    h, l, c = df["high"], df["low"], df["close"]
    prev_close = c.shift(1)
    tr = pd.concat([h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14, min_periods=5).mean()
    df["vol_ma"] = df["volume"].rolling(50, min_periods=10).mean()
    df["body_frac"] = (c - df["open"]).abs() / (h - l).replace(0, np.nan)
    df["imb"] = (df["buy_vol"] - df["sell_vol"]) / df["volume"].replace(0, np.nan)
    df["ret"] = c.pct_change()
    df["avg_trade"] = df["volume"] / df["trades"].replace(0, np.nan)
    df["avg_trade_q90"] = df["avg_trade"].rolling(500, min_periods=20).quantile(0.90)

    # absorption: high volume, small body, delta opposing the candle direction
    vol_spike = df["volume"] > df["vol_ma"] * cfg.abs_vol_mult
    small_body = df["body_frac"] < cfg.abs_body_frac
    df["absorb_long"] = (vol_spike & small_body &
                         (df["sell_vol"] > df["buy_vol"]))
    df["absorb_short"] = (vol_spike & small_body &
                          (df["buy_vol"] > df["sell_vol"]))

    # simplified CISD / change in state of delivery
    prev_bear = (df["close"].shift(1) < df["open"].shift(1))
    prev_bull = (df["close"].shift(1) > df["open"].shift(1))
    df["cisd_long"] = prev_bear & (c > df["open"].shift(1))
    df["cisd_short"] = prev_bull & (c < df["open"].shift(1))

    # causal swing points (confirmed k bars later)
    k = cfg.swing_k
    sh = pd.Series(np.nan, index=df.index)
    sl = pd.Series(np.nan, index=df.index)
    for i in range(k, len(df) - k):
        window_h = h.iloc[i - k:i + k + 1]
        window_l = l.iloc[i - k:i + k + 1]
        if h.iloc[i] >= window_h.max():
            sh.iloc[i + k] = h.iloc[i]
        if l.iloc[i] <= window_l.min():
            sl.iloc[i + k] = l.iloc[i]
    df["swing_high"] = sh
    df["swing_low"] = sl

    # trend / BOS / CHoCH, using only confirmed swings
    last_sh = np.nan
    last_sl = np.nan
    trend = 0  # 1 up, -1 down
    bos = np.zeros(len(df), dtype=np.int8)
    choch = np.zeros(len(df), dtype=np.int8)
    for i in range(len(df)):
        if not np.isnan(sh.iloc[i]):
            last_sh = sh.iloc[i]
        if not np.isnan(sl.iloc[i]):
            last_sl = sl.iloc[i]
        if not np.isnan(last_sh) and c.iloc[i] > last_sh:
            if trend <= 0:
                choch[i] = 1
            else:
                bos[i] = 1
            trend = 1
        elif not np.isnan(last_sl) and c.iloc[i] < last_sl:
            if trend >= 0:
                choch[i] = -1
            else:
                bos[i] = -1
            trend = -1
    df["bos"] = bos
    df["choch"] = choch
    df["trend"] = trend
    if cfg.use_heatmap:
        df = add_liquidity_features(df, cfg)
    return df


def add_liquidity_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Turn the per-band depth heatmap into wall / sweep / thin-book features."""
    bands = [c for c in df.columns if c.startswith("notional_")]
    if not bands:
        return df
    # Compare bid vs ask depth at the *same* distance. This is robust to the
    # fact that cumulative bands grow with distance and that liquidity
    # naturally decays away from mid.
    ask_map = {abs(float(c.split("_")[1])): c for c in bands
               if float(c.split("_")[1]) > 0}
    for c in bands:
        pct = float(c.split("_")[1])
        if pct >= 0:
            continue
        key = abs(pct)
        if key not in ask_map:
            continue
        b = df[c]
        a = df[ask_map[key]]
        denom = (b + a).replace(0, np.nan)
        df[f"band_imb_{key:g}"] = (b - a) / denom

    for key in ("1", "2"):
        col = f"band_imb_{key}"
        if col not in df:
            df[col] = np.nan

    df["liq_support_long"] = ((df["band_imb_1"] > cfg.band_imb_min) |
                              (df["band_imb_2"] > cfg.band_imb_min))
    df["liq_support_short"] = ((df["band_imb_1"] < -cfg.band_imb_min) |
                               (df["band_imb_2"] < -cfg.band_imb_min))

    # liquidity being pulled: total depth, and each side at the 1% band
    df["depth_total"] = df["bid_usd"] + df["ask_usd"]
    df["depth_chg"] = df["depth_total"].pct_change(cfg.wall_lookback)
    df["depth_pull"] = df["depth_chg"] <= -cfg.depth_pull_thresh
    if "notional_-01.0" in df and "notional_+01.0" in df:
        df["bid_pull"] = df["notional_-01.0"].pct_change(cfg.wall_lookback) <= -cfg.depth_pull_thresh
        df["ask_pull"] = df["notional_+01.0"].pct_change(cfg.wall_lookback) <= -cfg.depth_pull_thresh
    else:
        df["bid_pull"] = False
        df["ask_pull"] = False
    df["sweep_long"] = df["ask_pull"] & (df["close"] > df["open"])
    df["sweep_short"] = df["bid_pull"] & (df["close"] < df["open"])

    base = df["depth_total"].rolling(50, min_periods=10).median()
    df["thin_book"] = df["depth_total"] < base * 0.7
    return df


def add_divergence(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Delta/CVD divergence flags.

    window: price makes a new N-bar extreme while CVD does not confirm.
    swing:  strict confirmed swing-to-swing divergence.
    """
    if cfg.div_mode == "window":
        return add_divergence_window(df, cfg)
    return add_divergence_swing(df, cfg)


def add_divergence_window(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    low = df["low"].to_numpy()
    high = df["high"].to_numpy()
    cvd = df["cvd"].to_numpy()
    n = len(df)
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    w = cfg.div_window
    for i in range(w, n):
        lo_slice = slice(i - w, i - 1)
        j = (i - w) + int(np.argmin(low[lo_slice]))
        if low[i] <= low[j] and cvd[i] > cvd[j] and (i - j) >= cfg.div_min_sep:
            bull[i] = True
        j2 = (i - w) + int(np.argmax(high[lo_slice]))
        if high[i] >= high[j2] and cvd[i] < cvd[j2] and (i - j2) >= cfg.div_min_sep:
            bear[i] = True
    df["div_bull"] = bull
    df["div_bear"] = bear
    return df


def add_divergence_swing(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Regular divergence: price LL + CVD HL (bull) / HH + CVD LL (bear)."""
    n = len(df)
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    sw_l = np.flatnonzero(df["swing_low"].notna().to_numpy())
    sw_h = np.flatnonzero(df["swing_high"].notna().to_numpy())

    for arr, flags, kind in ((sw_l, bull, "low"), (sw_h, bear, "high")):
        for a, b in zip(arr[:-1], arr[1:]):
            if b - a > cfg.div_lookback:
                continue
            if kind == "low":
                p_a, p_b = df["low"].iloc[a], df["low"].iloc[b]
                c_a, c_b = df["cvd"].iloc[a], df["cvd"].iloc[b]
                if p_b < p_a and c_b > c_a:
                    flags[b] = True
            else:
                p_a, p_b = df["high"].iloc[a], df["high"].iloc[b]
                c_a, c_b = df["cvd"].iloc[a], df["cvd"].iloc[b]
                if p_b > p_a and c_b < c_a:
                    flags[b] = True
    df["div_bull"] = bull
    df["div_bear"] = bear
    return df


# --------------------------------------------------------------------------
# rolling volume profile
# --------------------------------------------------------------------------

class _LazyRow:
    """Index rows of a rolling profile on demand (memory-safe for wide ranges)."""

    def __init__(self, fn, key):
        self.fn = fn
        self.key = key

    def __getitem__(self, i):
        return self.fn(int(i))[self.key]


def rolling_profile(bars: pd.DataFrame, vp: pd.DataFrame, cfg: Config) -> dict:
    """Rolling POC / LVN / HVN, computed lazily per requested bar.

    A dense bars x buckets matrix is fine for BTC but not for XRP/DOGE: a year
    at their 0.0001 / 0.00002 tick spans >15k buckets, i.e. several GB. Only
    formation bars are ever scored, so rows are built on demand instead.
    """
    tick = cfg.profile_tick
    vp = vp.copy()
    vp["bucket"] = (vp["price_bucket"] / tick).round() * tick
    vp = vp.sort_values("timestamp")
    vp_ts = pd.to_datetime(vp["timestamp"], utc=True).to_numpy(
        dtype="datetime64[ns]").astype("int64")
    vp_bucket = vp["bucket"].to_numpy(dtype="float64")
    vp_vol = vp["volume"].to_numpy(dtype="float64")
    bars_ts = pd.to_datetime(bars["timestamp"], utc=True).to_numpy(
        dtype="datetime64[ns]").astype("int64")
    buckets = np.unique(vp_bucket)
    nb = len(buckets)

    win = max(10, min(cfg.profile_window, len(bars) - 1))
    minp = max(8, win // 4)
    cache: dict[int, dict] = {}

    def row(i: int) -> dict:
        hit = cache.get(i)
        if hit is not None:
            return hit
        start = bars_ts[max(0, i - win + 1)]
        a = int(np.searchsorted(vp_ts, start, "left"))
        b = int(np.searchsorted(vp_ts, bars_ts[i], "right"))
        sl_b = vp_bucket[a:b]
        sl_v = vp_vol[a:b]
        if len(sl_b):
            ub, inv = np.unique(sl_b, return_inverse=True)
            vols = np.bincount(inv, weights=sl_v)
        else:
            ub = np.empty(0)
            vols = np.empty(0)
        traded = vols[vols > 0]
        if len(traded):
            q_lo = float(np.quantile(traded, cfg.lvn_pct))
            q_hi = float(np.quantile(traded, cfg.hvn_pct))
        else:
            q_lo = q_hi = 0.0
        full = np.zeros(nb, dtype="float32")
        if len(ub):
            full[np.searchsorted(buckets, ub)] = vols
        out = {"lvn": full <= q_lo, "hvn": full >= q_hi,
               "poc": float(ub[int(np.argmax(vols))]) if len(ub) else np.nan,
               "ready": i >= minp - 1}
        cache[i] = out
        return out

    ready = np.arange(len(bars)) >= minp - 1
    return {"buckets": buckets, "lvn": _LazyRow(row, "lvn"),
            "hvn": _LazyRow(row, "hvn"), "poc": _LazyRow(row, "poc"),
            "ready": ready}


def overlap_fraction(buckets: np.ndarray, mask: np.ndarray,
                     lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    inside = (buckets >= lo) & (buckets <= hi)
    if not inside.any():
        return 0.0
    return float(mask[inside].mean())


# --------------------------------------------------------------------------
# FVG / IFVG detection
# --------------------------------------------------------------------------

def detect_gaps(df: pd.DataFrame, cfg: Config) -> list[dict]:
    """All FVGs (and inversions) with causal fill tracking as of each bar.

    Each gap dict: dir(+1 bull/-1 bear), lo, hi, form_idx, kind('fvg'/'ifvg'),
    origin_idx (index of the gap that was inverted).
    """
    gaps: list[dict] = []
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    atr = df["atr"].to_numpy()
    for i in range(2, len(df)):
        # bullish FVG: candle i-2 high < candle i low
        if h[i - 2] < l[i]:
            lo, hi = h[i - 2], l[i]
            if hi - lo >= cfg.fvg_min_atr * atr[i - 1]:
                gaps.append({"dir": 1, "lo": lo, "hi": hi, "form_idx": i,
                             "kind": "fvg", "origin_idx": None,
                             "form_close": c[i],
                             "filled_idx": None, "invert_idx": None})
        # bearish FVG: candle i-2 low > candle i high
        if l[i - 2] > h[i]:
            lo, hi = h[i], l[i - 2]
            if hi - lo >= cfg.fvg_min_atr * atr[i - 1]:
                gaps.append({"dir": -1, "lo": lo, "hi": hi, "form_idx": i,
                             "kind": "fvg", "origin_idx": None,
                             "form_close": c[i],
                             "filled_idx": None, "invert_idx": None})

    # causal fill / inversion tracking (snapshot so appends are not re-processed)
    for g in list(gaps):
        d, lo, hi = g["dir"], g["lo"], g["hi"]
        for j in range(g["form_idx"] + 1, len(df)):
            if d == 1:
                if l[j] <= lo:              # fully filled
                    g["filled_idx"] = j
                    if c[j] < lo and g["kind"] == "fvg":
                        g["invert_idx"] = j
                    break
            else:
                if h[j] >= hi:
                    g["filled_idx"] = j
                    if c[j] > hi and g["kind"] == "fvg":
                        g["invert_idx"] = j
                    break
        if g["invert_idx"] is not None:
            j = g["invert_idx"]
            gaps.append({"dir": -d, "lo": lo, "hi": hi, "form_idx": j,
                         "kind": "ifvg", "origin_idx": g["form_idx"],
                         "form_close": c[j],
                         "filled_idx": None, "invert_idx": None})
    gaps.sort(key=lambda x: x["form_idx"])
    return gaps


# --------------------------------------------------------------------------
# zone scoring
# --------------------------------------------------------------------------

def score_gap(g: dict, idx: int, df: pd.DataFrame, prof: dict,
              cfg: Config) -> dict:
    """Score a gap using information available at bar `idx` (formation bar)."""
    lo, hi = g["lo"], g["hi"]
    buckets = prof["buckets"]
    lvn_ov = overlap_fraction(buckets, prof["lvn"][idx], lo, hi)
    hvn_ov = overlap_fraction(buckets, prof["hvn"][idx], lo, hi)
    poc = prof["poc"][idx]
    poc_target = bool(poc > hi) if g["dir"] == 1 else bool(poc < lo)

    row = df.iloc[idx]
    bos = 0
    look = slice(max(0, idx - 20), idx + 1)
    if (df["bos"].iloc[look] * g["dir"]).max() > 0:
        bos = 3
    elif (df["choch"].iloc[look] * g["dir"]).max() > 0:
        bos = 2

    div_form = False
    fw = slice(max(0, idx - 2), idx + 3)
    if g["dir"] == 1:
        div_form = bool(df["div_bull"].iloc[fw].any())
    else:
        div_form = bool(df["div_bear"].iloc[fw].any())

    # large print inside the gap: formation-bar average trade vs its 90th pct
    big = False
    if np.isfinite(row.get("avg_trade_q90", np.nan)) and np.isfinite(row["avg_trade"]):
        big = bool(row["avg_trade"] > row["avg_trade_q90"])

    imb = float(row["imb"]) if np.isfinite(row["imb"]) else 0.0
    imb_ok = (imb * g["dir"]) > cfg.imb_thresh

    absorb = bool(row["absorb_long"]) if g["dir"] == 1 else bool(row["absorb_short"])
    cisd = bool(row["cisd_long"]) if g["dir"] == 1 else bool(row["cisd_short"])

    oi = float(row["oi_change"]) if "oi_change" in df and np.isfinite(row["oi_change"]) else 0.0
    depth = float(row["depth_imb"]) if "depth_imb" in df and np.isfinite(row["depth_imb"]) else 0.0

    # liquidity heatmap: same-distance band imbalance, sweeps, and an
    # opposite-side imbalance that would block the target
    liq_zone = False
    wall_block = False
    wall_conc = 0.0
    if cfg.use_heatmap and "band_imb_1" in df:
        if g["dir"] == 1:
            liq_zone = bool(row["liq_support_long"] or row["sweep_long"])
            wall_conc = float(row["band_imb_1"]) if np.isfinite(row["band_imb_1"]) else 0.0
            wall_block = bool(np.isfinite(row["band_imb_2"]) and
                              row["band_imb_2"] <= -cfg.wall_block_imb)
        else:
            liq_zone = bool(row["liq_support_short"] or row["sweep_short"])
            wall_conc = float(row["band_imb_1"]) if np.isfinite(row["band_imb_1"]) else 0.0
            wall_block = bool(np.isfinite(row["band_imb_2"]) and
                              row["band_imb_2"] >= cfg.wall_block_imb)

    score = TF_BASE_POINTS.get(cfg.tf, 3.0)
    score += 4.0                                   # unfilled FVG/IFVG exists
    score += 4.0 * lvn_ov if lvn_ov >= cfg.align_min else 2.0 * lvn_ov
    score -= 2.0 if hvn_ov >= cfg.align_min else 0.0
    score += 2.0 if poc_target else 0.0
    score += bos
    score += 3.0 if div_form else 0.0
    score += 2.0 if big else 0.0
    score += 2.0 if imb_ok else 0.0
    score += 2.0 if absorb else 0.0
    score += 2.0 if cisd else 0.0
    score += 1.0 if oi * g["dir"] > 0 else 0.0
    score += 1.0 if depth * g["dir"] > 0 else 0.0
    score += 2.0 if liq_zone else 0.0
    score += 1.0 if abs(wall_conc) >= cfg.band_imb_min else 0.0
    score -= 2.0 if wall_block else 0.0

    return {"score": score, "lvn_ov": lvn_ov, "hvn_ov": hvn_ov,
            "poc_target": poc_target, "div_formation": div_form,
            "big_print": big, "imb_ok": imb_ok, "absorb": absorb,
            "cisd": cisd, "oi": oi, "depth": depth, "bos": bos,
            "liq_zone": liq_zone, "wall_block": wall_block,
            "wall_conc": wall_conc}


# --------------------------------------------------------------------------
# backtest
# --------------------------------------------------------------------------

def backtest(bars: pd.DataFrame, vp: pd.DataFrame, cfg: Config) -> dict:
    df = add_indicators(bars.copy(), cfg)
    df = add_divergence(df, cfg)
    prof = rolling_profile(df, vp, cfg)
    gaps = detect_gaps(df, cfg)

    # score every gap at its formation bar, keep only strong ones
    for g in gaps:
        if not prof["ready"][g["form_idx"]]:
            g["score"] = -1.0
            continue
        s = score_gap(g, g["form_idx"], df, prof, cfg)
        g.update(s)
    scored = [g for g in gaps if g["score"] >= cfg.min_score]
    scored.sort(key=lambda x: x["form_idx"])

    # index gaps by formation bar for the retest loop
    active: list[dict] = []
    by_form: dict[int, list[dict]] = {}
    for g in scored:
        by_form.setdefault(g["form_idx"], []).append(g)

    trades: list[dict] = []
    equity = cfg.starting_equity
    equity_curve = []
    i = 0
    n = len(df)
    last_exit = -1

    while i < n:
        for g in by_form.get(i, []):
            active.append({**g, "touched": False, "inverted_used": False})

        # drop stale / filled gaps
        active = [g for g in active
                  if i - g["form_idx"] <= cfg.fvg_max_age
                  and (g["filled_idx"] is None or g["filled_idx"] >= i)]

        if i == last_exit or i >= n - 2:
            equity_curve.append({"timestamp": df["timestamp"].iloc[i], "equity": equity})
            i += 1
            continue

        row = df.iloc[i]
        entry = None
        for g in active:
            # retest: bar range intersects the gap
            if not (row["low"] <= g["hi"] and row["high"] >= g["lo"]):
                continue
            direction = g["dir"]
            if direction == 1 and not cfg.allow_longs:
                continue
            if direction == -1 and not cfg.allow_shorts:
                continue

            div_retest = bool(row["div_bull"]) if direction == 1 else bool(row["div_bear"])
            absorb = bool(row["absorb_long"]) if direction == 1 else bool(row["absorb_short"])
            cisd = bool(row["cisd_long"]) if direction == 1 else bool(row["cisd_short"])
            imb_ok = (row["imb"] * direction) > cfg.imb_thresh
            depth_ok = (row["depth_imb"] * direction) > 0 if "depth_imb" in df else False
            oi_ok = (row["oi_change"] * direction) > 0 if "oi_change" in df else False
            liq_ok = False
            if cfg.use_heatmap and "band_imb_1" in df:
                if direction == 1:
                    liq_ok = bool(row["liq_support_long"] or row["sweep_long"])
                else:
                    liq_ok = bool(row["liq_support_short"] or row["sweep_short"])

            stages = int(g["div_formation"]) + int(div_retest)
            if cfg.require_formation_div and not g["div_formation"]:
                continue
            if cfg.require_retest_div and not div_retest:
                continue
            if stages < min(cfg.min_div_stages, 2):
                continue
            if not (absorb or cisd or imb_ok or liq_ok):
                continue

            # SL / TP
            if direction == 1:
                if cfg.sl_mode == "halflow":
                    stop = (g["lo"] + min(row["low"], g["lo"])) / 2.0
                else:
                    stop = min(row["low"], g["lo"]) - cfg.sl_buffer_atr * row["atr"]
                if cfg.tp_mode == "next_close":
                    target = g["form_close"] - cfg.sl_buffer_atr * row["atr"] * 0.5
                else:
                    target = g["hi"] - cfg.sl_buffer_atr * row["atr"] * 0.5
                entry_price = df["open"].iloc[i + 1]
                if target <= entry_price or entry_price <= stop:
                    continue
            else:
                if cfg.sl_mode == "halflow":
                    stop = (g["hi"] + max(row["high"], g["hi"])) / 2.0
                else:
                    stop = max(row["high"], g["hi"]) + cfg.sl_buffer_atr * row["atr"]
                if cfg.tp_mode == "next_close":
                    target = g["form_close"] + cfg.sl_buffer_atr * row["atr"] * 0.5
                else:
                    target = g["lo"] + cfg.sl_buffer_atr * row["atr"] * 0.5
                entry_price = df["open"].iloc[i + 1]
                if target >= entry_price or entry_price >= stop:
                    continue

            risk = abs(entry_price - stop)
            reward = abs(target - entry_price)
            if risk <= 0 or reward / risk < cfg.min_rr:
                continue

            entry = {"gap": g, "dir": direction, "entry_price": entry_price,
                     "stop": stop, "target": target, "risk": risk,
                     "entry_idx": i + 1, "stages": stages,
                     "div_retest": div_retest, "absorb": absorb,
                     "cisd": cisd, "imb_ok": imb_ok,
                     "depth_ok": depth_ok, "oi_ok": oi_ok,
                     "score": g["score"]}
            break

        if entry is None:
            equity_curve.append({"timestamp": row["timestamp"], "equity": equity})
            i += 1
            continue

        # size and simulate
        qty = (equity * cfg.risk_pct) / entry["risk"]
        entry_fee = entry["entry_price"] * qty * (cfg.fee_bps + cfg.slip_bps) / 1e4
        exit_idx = None
        exit_price = None
        reason = None
        div_reversion = False
        for j in range(entry["entry_idx"], min(n, entry["entry_idx"] + cfg.time_stop_bars)):
            # stage C: divergence/confirmation observed while the trade is live
            if entry["dir"] == 1 and bool(df["div_bull"].iloc[j]):
                div_reversion = True
            elif entry["dir"] == -1 and bool(df["div_bear"].iloc[j]):
                div_reversion = True
            lo, hi = df["low"].iloc[j], df["high"].iloc[j]
            if entry["dir"] == 1:
                if lo <= entry["stop"]:
                    exit_price, reason, exit_idx = entry["stop"], "stop", j
                    break
                if hi >= entry["target"]:
                    exit_price, reason, exit_idx = entry["target"], "target", j
                    break
            else:
                if hi >= entry["stop"]:
                    exit_price, reason, exit_idx = entry["stop"], "stop", j
                    break
                if lo <= entry["target"]:
                    exit_price, reason, exit_idx = entry["target"], "target", j
                    break
        if exit_idx is None:
            exit_idx = min(n - 1, entry["entry_idx"] + cfg.time_stop_bars - 1)
            exit_price, reason = df["close"].iloc[exit_idx], "time"

        gross = (exit_price - entry["entry_price"]) * entry["dir"] * qty
        exit_fee = exit_price * qty * (cfg.fee_bps + cfg.slip_bps) / 1e4
        pnl = gross - entry_fee - exit_fee
        equity += pnl
        trades.append({
            "entry_time": df["timestamp"].iloc[entry["entry_idx"]],
            "exit_time": df["timestamp"].iloc[exit_idx],
            "direction": "long" if entry["dir"] == 1 else "short",
            "score": round(entry["score"], 2),
            "entry": entry["entry_price"], "stop": entry["stop"],
            "target": entry["target"], "exit": exit_price,
            "reason": reason, "qty": qty, "pnl": pnl,
            "r_multiple": (exit_price - entry["entry_price"]) * entry["dir"] / entry["risk"],
            "div_stages": entry["stages"],
            "div_reversion": div_reversion,
            "div_retest": entry["div_retest"], "absorb": entry["absorb"],
            "cisd": entry["cisd"], "imb_ok": entry["imb_ok"],
            "depth_ok": entry["depth_ok"], "oi_ok": entry["oi_ok"],
        })
        for j in range(i, exit_idx + 1):
            equity_curve.append({"timestamp": df["timestamp"].iloc[j], "equity": equity})
        last_exit = exit_idx
        i = exit_idx + 1

    return summarize(trades, equity_curve, cfg)


def summarize(trades: list[dict], equity_curve: list[dict], cfg: Config) -> dict:
    if not trades:
        return {"trades": 0, "note": "no trades passed filters",
                "config": asdict(cfg)}
    t = pd.DataFrame(trades)
    wins = t[t["pnl"] > 0]
    losses = t[t["pnl"] <= 0]
    gp = wins["pnl"].sum()
    gl = -losses["pnl"].sum()
    eq = pd.DataFrame(equity_curve)
    peak = eq["equity"].cummax()
    dd = (eq["equity"] / peak - 1.0).min()
    per_dir = t.groupby("direction")["pnl"].agg(["count", "sum", "mean"]).to_dict("index")
    ret = eq["equity"].pct_change().dropna()
    bars_per_year = {"1m": 525600, "3m": 175200, "5m": 105120, "10m": 52560,
                     "1h": 8760, "2h": 4380, "4h": 2190, "8h": 1095,
                     "12h": 730, "1d": 365, "2d": 182.5}.get(cfg.tf, 8760)
    sharpe = (float(ret.mean() / ret.std() * np.sqrt(bars_per_year))
              if len(ret) > 2 and ret.std() > 0 else None)

    return {
        "trades": int(len(t)),
        "win_rate": float((t["pnl"] > 0).mean()),
        "profit_factor": float(gp / gl) if gl > 0 else None,
        "net_pnl": float(t["pnl"].sum()),
        "expectancy": float(t["pnl"].mean()),
        "avg_r": float(t["r_multiple"].mean()),
        "max_drawdown_pct": float(dd * 100),
        "by_direction": per_dir,
        "sharpe": sharpe,
        "config": asdict(cfg),
        "trades_detail": trades,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tf", default="15m")
    ap.add_argument("--profile-window", type=int, default=2016)
    ap.add_argument("--profile-tick", type=float, default=50.0)
    ap.add_argument("--min-score", type=float, default=12.0)
    ap.add_argument("--sl-mode", default="halflow", choices=["ob_low", "halflow"])
    ap.add_argument("--tp-mode", default="next_close", choices=["fvg_far", "next_close"])
    ap.add_argument("--min-div-stages", type=int, default=2)
    ap.add_argument("--div-mode", default="window", choices=["window", "swing"])
    ap.add_argument("--div-window", type=int, default=60)
    ap.add_argument("--require-formation-div", action="store_true")
    ap.add_argument("--no-retest-div", action="store_true")
    ap.add_argument("--risk-pct", type=float, default=0.005)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config(tf=args.tf, profile_window=args.profile_window,
                 profile_tick=args.profile_tick, min_score=args.min_score,
                 sl_mode=args.sl_mode, tp_mode=args.tp_mode,
                 min_div_stages=args.min_div_stages,
                 require_formation_div=args.require_formation_div,
                 require_retest_div=not args.no_retest_div,
                 div_mode=args.div_mode, div_window=args.div_window,
                 risk_pct=args.risk_pct)
    bars, vp = load_data(args.data, cfg.tf)
    result = backtest(bars, vp, cfg)
    out = args.out or os.path.join(args.data, f"lvn_fvg_{cfg.tf}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in result.items() if k != "trades_detail"},
                     indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
