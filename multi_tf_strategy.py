"""Multi-timeframe LVN-FVG scanner + backtest (v2).

Zone rule:
  1. Scan the hierarchy from the highest timeframe down:
     3M -> 1M -> 2W -> 1W -> 1D -> 12h -> 8h -> 4h -> 2h -> 1h -> 30m
  2. A zone is valid only if it is an unfilled FVG/IFVG that aligns with a
     low-volume node (LVN) and scores >= min_score.
  3. The same-direction FVG must also exist (at least partially) on the LTF:
     price ranges must overlap and it must be part of the same impulse.
  4. The highest TF with a valid, nested zone wins; if none, wait.

Entry / SL / TP are taken from the nested LTF gap.

Paper/research only.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from lvn_fvg_strategy import (
    Config, add_divergence, add_indicators, detect_gaps, load_data,
    rolling_profile, score_gap,
)

HIERARCHY = ["3M", "1M", "2W", "1W", "1d", "12h", "8h", "4h", "2h", "1h", "30m", "15m"]
DEFAULT_TFS = ["1d", "12h", "8h", "4h", "2h", "1h", "30m"]


def overlap_frac(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> float:
    inter = min(a_hi, b_hi) - max(a_lo, b_lo)
    if inter <= 0:
        return 0.0
    denom = min(a_hi - a_lo, b_hi - b_lo)
    return float(inter / denom) if denom > 0 else 0.0


def tf_available(root: str, tf: str) -> bool:
    return (os.path.exists(os.path.join(root, f"bars_{tf}.parquet")) and
            os.path.exists(os.path.join(root, f"volume_profile_{tf}.parquet")))


def tf_state(root: str, tf: str, cfg: Config):
    bars, vp = load_data(root, tf)
    # Missing source days can otherwise leave NaN bins after a resample.
    bars = bars[bars["close"].notna()].reset_index(drop=True)
    c = replace(cfg, tf=tf)
    df = add_divergence(add_indicators(bars.copy(), c), c)
    prof = rolling_profile(df, vp, c)
    gaps = detect_gaps(df, c)
    for g in gaps:
        if not prof["ready"][g["form_idx"]]:
            g["score"] = -1.0
            continue
        g.update(score_gap(g, g["form_idx"], df, prof, c))
        g["tf"] = tf
        g["form_time"] = df["timestamp"].iloc[g["form_idx"]]
        g["fill_time"] = (None if g["filled_idx"] is None
                          else df["timestamp"].iloc[g["filled_idx"]])
        g["bar_delta"] = (df["timestamp"].iloc[1] - df["timestamp"].iloc[0]
                          if len(df) > 1 else pd.Timedelta("1h"))
    return df, gaps


def attach_nested(zones, htf_df, ltf_gaps, ltf_df, cfg: argparse.Namespace):
    """Keep only zones whose same-direction FVG exists on the LTF."""
    out = []
    for z in zones:
        start = htf_df["timestamp"].iloc[max(0, z["form_idx"] - 2)]
        end = htf_df["timestamp"].iloc[z["form_idx"]] + z["bar_delta"]
        pad = z["bar_delta"] * cfg.nested_time_pad
        matches = []
        for lg in ltf_gaps:
            if lg["dir"] != z["dir"]:
                continue
            if overlap_frac(z["lo"], z["hi"], lg["lo"], lg["hi"]) < cfg.nested_overlap:
                continue
            if cfg.nested_mode == "live":
                matches.append(lg)
                continue
            ft = ltf_df["timestamp"].iloc[lg["form_idx"]]
            if (start - pad) <= ft <= (end + pad):
                matches.append(lg)
        if matches:
            out.append({**z, "nested": matches})
    return out


def build_zones(root, tfs, ltf, cfg, args):
    ltf_df, ltf_gaps = tf_state(root, ltf, cfg)
    zones_by_tf = {}
    for tf in tfs:
        if not tf_available(root, tf):
            print(f"  {tf:>4}: missing, skipped")
            continue
        htf_df, gaps = tf_state(root, tf, cfg)
        valid = [g for g in gaps
                 if g["score"] >= cfg.min_score and g.get("lvn_ov", 0) >= cfg.align_min]
        nested = attach_nested(valid, htf_df, ltf_gaps, ltf_df, args)
        zones_by_tf[tf] = nested
        print(f"  {tf:>4}: gaps={len(gaps):>5}  scored={sum(g['score'] >= cfg.min_score for g in gaps):>4}  "
              f"lvn_ok={len(valid):>3}  nested={len(nested):>3}")
    return ltf_df, zones_by_tf, ltf_gaps


def load_footprint(root: str, ltf: str):
    """timestamp -> (prices, buy_vol, sell_vol) for per-price gap analysis."""
    path = os.path.join(root, f"footprint_{ltf}.parquet")
    if not os.path.exists(path):
        return {}
    fp = pq.read_table(path).to_pandas()
    fp["timestamp"] = pd.to_datetime(fp["timestamp"], utc=True)
    out = {}
    for ts, g in fp.groupby("timestamp", sort=False):
        out[int(pd.Timestamp(ts).value)] = (
            g["price"].to_numpy(dtype=float),
            g["buy_vol"].to_numpy(dtype=float),
            g["sell_vol"].to_numpy(dtype=float))
    return out


def simulate(ltf_df, zones_by_tf, tfs_order, cfg, args, fp_map=None,
             entry_filter=None):
    by_time = {}
    for tf, zones in zones_by_tf.items():
        for z in zones:
            by_time.setdefault(z["form_time"], []).append(z)

    active = []
    trades = []
    equity = cfg.starting_equity
    i, n = 0, len(ltf_df)
    last_exit = -1
    funnel = {"bars": 0, "active_htf": 0, "retest": 0, "nested_live": 0,
              "div_retest": 0, "confirm": 0, "valid_sltp": 0, "entries": 0}

    while i < n:
        t = ltf_df["timestamp"].iloc[i]
        funnel["bars"] += 1
        for z in by_time.get(t, []):
            active.append(z)
        active = [z for z in active
                  if (z["fill_time"] is None or t <= z["fill_time"])
                  and (t - z["form_time"]) <= pd.Timedelta(days=args.max_zone_age_days)]

        chosen_tf = None
        chosen = []
        for tf in tfs_order:            # highest first
            zs = [z for z in active if z["tf"] == tf]
            if zs:
                chosen_tf, chosen = tf, zs
                break
        if not chosen:
            i += 1
            continue
        funnel["active_htf"] += 1

        row = ltf_df.iloc[i]
        entry = None
        for z in chosen:
            touched = row["low"] <= z["hi"] and row["high"] >= z["lo"]
            near = False
            if cfg.entry_mode == "anticipate":
                prox = cfg.entry_proximity_atr * (
                    row["atr"] if np.isfinite(row["atr"]) else 0.0)
                if z["dir"] == 1:
                    near = row["close"] > z["hi"] and (row["close"] - z["hi"]) <= prox
                else:
                    near = row["close"] < z["lo"] and (z["lo"] - row["close"]) <= prox
            if not (touched or near):
                continue
            if near and not touched:
                funnel["approach"] = funnel.get("approach", 0) + 1
            if touched:
                z.setdefault("_retest_start", i)
            funnel["retest"] += 1
            live = []
            for lg in z["nested"]:
                ft = ltf_df["timestamp"].iloc[lg["form_idx"]]
                if ft > t:
                    continue
                if lg["filled_idx"] is not None:
                    ft_fill = ltf_df["timestamp"].iloc[lg["filled_idx"]]
                    if t > ft_fill:
                        continue
                live.append(lg)
            if not live:
                continue
            funnel["nested_live"] += 1
            # deepest/most-overlapping nested gap
            live.sort(key=lambda g: overlap_frac(z["lo"], z["hi"], g["lo"], g["hi"]),
                      reverse=True)
            g = live[0]
            d = g["dir"]
            if d == 1 and not cfg.allow_longs:
                continue
            if d == -1 and not cfg.allow_shorts:
                continue

            cw = max(1, cfg.confirm_window)
            if cfg.confirm_from_retest and "_retest_start" in z:
                rs = z["_retest_start"]
                if i >= rs + max(1, cfg.retest_window):
                    continue        # confirmation must come from the retest start
                rec = slice(rs, i + 1)
            else:
                rec = slice(max(0, i - cw + 1), i + 1)
            div_retest = bool(ltf_df["div_bull"].iloc[rec].any()) if d == 1 \
                else bool(ltf_df["div_bear"].iloc[rec].any())
            absorb = bool(ltf_df["absorb_long"].iloc[rec].any()) if d == 1 \
                else bool(ltf_df["absorb_short"].iloc[rec].any())
            cisd = bool(ltf_df["cisd_long"].iloc[rec].any()) if d == 1 \
                else bool(ltf_df["cisd_short"].iloc[rec].any())
            imb_ok = bool(((ltf_df["imb"].iloc[rec] * d) > cfg.imb_thresh).any())
            # per-price footprint imbalance inside the nested gap over the retest
            fp_ok = False
            fp_imb = 0.0
            if cfg.use_footprint and fp_map:
                bvol = svol = 0.0
                for j in range(rec.start, rec.stop):
                    tsj = ltf_df["timestamp"].iloc[j]
                    pr, bv, sv = fp_map.get(int(pd.Timestamp(tsj).value),
                                            (None, None, None))
                    if pr is None:
                        continue
                    m = (pr >= g["lo"]) & (pr <= g["hi"])
                    bvol += float(bv[m].sum())
                    svol += float(sv[m].sum())
                if bvol + svol > 0:
                    fp_imb = (bvol - svol) / (bvol + svol)
                    fp_ok = fp_imb * d >= cfg.fp_imb_min
            if fp_ok:
                funnel["fp_ok"] = funnel.get("fp_ok", 0) + 1
            liq_ok = False
            if cfg.use_heatmap and "bid_wall_usd" in ltf_df:
                if d == 1:
                    liq_ok = bool((ltf_df["liq_support_long"].iloc[rec] |
                                   ltf_df["sweep_long"].iloc[rec]).any())
                else:
                    liq_ok = bool((ltf_df["liq_support_short"].iloc[rec] |
                                   ltf_df["sweep_short"].iloc[rec]).any())
            if liq_ok:
                funnel["liq_ok"] = funnel.get("liq_ok", 0) + 1
            if div_retest:
                funnel["div_retest"] += 1
            if cfg.require_retest_div and not div_retest:
                continue
            if cfg.require_absorb and not absorb:
                continue
            if not (absorb or cisd or imb_ok or liq_ok or fp_ok):
                continue
            stages = int(z.get("div_formation", False)) + int(div_retest)
            if stages < cfg.min_div_stages:
                continue
            funnel["confirm"] += 1

            if d == 1:
                stop = ((g["lo"] + min(row["low"], g["lo"])) / 2.0
                        if cfg.sl_mode == "halflow"
                        else min(row["low"], g["lo"]) - cfg.sl_buffer_atr * row["atr"])
            else:
                stop = ((g["hi"] + max(row["high"], g["hi"])) / 2.0
                        if cfg.sl_mode == "halflow"
                        else max(row["high"], g["hi"]) + cfg.sl_buffer_atr * row["atr"])
            entry_price = ltf_df["open"].iloc[i + 1]
            funnel["valid_sltp"] += 1
            atr = row["atr"] if np.isfinite(row["atr"]) else 0.0
            min_stop = cfg.min_stop_atr * atr
            if abs(entry_price - stop) < min_stop:
                stop = entry_price - d * min_stop
            risk = abs(entry_price - stop)
            if risk <= 0:
                continue
            if cfg.tp_mode == "r":
                target = entry_price + d * cfg.tp_r * risk
            elif d == 1:
                target = (g["form_close"] - 0.5 * cfg.sl_buffer_atr * row["atr"]
                          if cfg.tp_mode == "next_close"
                          else g["hi"] - 0.5 * cfg.sl_buffer_atr * row["atr"])
            else:
                target = (g["form_close"] + 0.5 * cfg.sl_buffer_atr * row["atr"]
                          if cfg.tp_mode == "next_close"
                          else g["lo"] + 0.5 * cfg.sl_buffer_atr * row["atr"])
            if not ((d == 1 and target > entry_price > stop) or
                    (d == -1 and target < entry_price < stop)):
                continue
            reward = abs(target - entry_price)
            if reward / risk < cfg.min_rr:
                continue
            # heavy opposite-side depth overhead blocks the target
            if cfg.use_heatmap and "band_imb_2" in ltf_df:
                if d == 1 and (np.isfinite(row["band_imb_2"]) and
                               row["band_imb_2"] <= -cfg.wall_block_imb):
                    funnel["wall_block"] = funnel.get("wall_block", 0) + 1
                    continue
                if d == -1 and (np.isfinite(row["band_imb_2"]) and
                                row["band_imb_2"] >= cfg.wall_block_imb):
                    funnel["wall_block"] = funnel.get("wall_block", 0) + 1
                    continue

            entry = {"zone": z, "gap": g, "dir": d, "entry_price": entry_price,
                     "stop": stop, "target": target, "risk": risk,
                     "entry_idx": i + 1, "stages": stages}
            if entry_filter is not None and not entry_filter(
                    ltf_df["timestamp"].iloc[entry["entry_idx"]], d):
                entry = None
                continue
            break

        if entry is None:
            i += 1
            continue

        funnel["entries"] += 1
        risk_budget = equity * cfg.risk_pct
        qty_risk = risk_budget / entry["risk"]
        qty_cap = (equity * cfg.max_notional_mult) / entry["entry_price"]
        qty = min(qty_risk, qty_cap)
        fee_in = entry["entry_price"] * qty * (cfg.fee_bps + cfg.slip_bps) / 1e4
        exit_idx = exit_price = None
        reason = None
        div_rev = False
        for j in range(entry["entry_idx"], min(n, entry["entry_idx"] + cfg.time_stop_bars)):
            if entry["dir"] == 1 and bool(ltf_df["div_bull"].iloc[j]):
                div_rev = True
            elif entry["dir"] == -1 and bool(ltf_df["div_bear"].iloc[j]):
                div_rev = True
            lo, hi = ltf_df["low"].iloc[j], ltf_df["high"].iloc[j]
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
            exit_price, reason = ltf_df["close"].iloc[exit_idx], "time"
        if not np.isfinite(exit_price):
            # Data gap at the exit bar: don't let a NaN poison equity.
            i = entry["entry_idx"] + 1
            last_exit = i - 1
            continue
        gross = (exit_price - entry["entry_price"]) * entry["dir"] * qty
        fee_out = exit_price * qty * (cfg.fee_bps + cfg.slip_bps) / 1e4
        pnl = gross - fee_in - fee_out
        equity += pnl
        trades.append({
            "zone_tf": entry["zone"]["tf"],
            "zone_score": round(entry["zone"]["score"], 2),
            "zone_gap": [round(entry["zone"]["lo"], 2), round(entry["zone"]["hi"], 2)],
            "ltf_gap": [round(entry["gap"]["lo"], 2), round(entry["gap"]["hi"], 2)],
            "entry_time": str(ltf_df["timestamp"].iloc[entry["entry_idx"]]),
            "exit_time": str(ltf_df["timestamp"].iloc[exit_idx]),
            "direction": "long" if entry["dir"] == 1 else "short",
            "entry": entry["entry_price"], "stop": entry["stop"],
            "target": entry["target"], "exit": exit_price, "reason": reason,
            "qty": qty, "pnl": pnl,
            "r_multiple": (exit_price - entry["entry_price"]) * entry["dir"] / entry["risk"],
            "div_stages": entry["stages"], "div_reversion": div_rev,
        })
        last_exit = exit_idx
        i = exit_idx + 1

    return summarize(trades, funnel, cfg)


def summarize(trades, funnel, cfg):
    if not trades:
        return {"trades": 0, "funnel": funnel, "config": asdict(cfg)}
    t = pd.DataFrame(trades)
    wins = t[t["pnl"] > 0]
    losses = t[t["pnl"] <= 0]
    gp, gl = wins["pnl"].sum(), -losses["pnl"].sum()
    return {
        "trades": len(t),
        "win_rate": float((t["pnl"] > 0).mean()),
        "profit_factor": float(gp / gl) if gl > 0 else None,
        "net_pnl": float(t["pnl"].sum()),
        "expectancy": float(t["pnl"].mean()),
        "avg_r": float(t["r_multiple"].mean()),
        "by_zone_tf": t.groupby("zone_tf")["pnl"].agg(["count", "sum"]).to_dict("index"),
        "funnel": funnel,
        "config": asdict(cfg),
        "trades_detail": trades,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ltf", default="10m")
    ap.add_argument("--tfs", default=",".join(DEFAULT_TFS))
    ap.add_argument("--profile-window", type=int, default=720)
    ap.add_argument("--profile-tick", type=float, default=50.0)
    ap.add_argument("--min-score", type=float, default=12.0)
    ap.add_argument("--nested-overlap", type=float, default=0.5)
    ap.add_argument("--nested-mode", default="impulse", choices=["impulse", "live"],
                    help="impulse: LTF gap from the same HTF gap; live: any overlapping LTF gap")
    ap.add_argument("--nested-time-pad", type=float, default=1.0,
                    help="HTF bar-durations of tolerance around the gap")
    ap.add_argument("--max-zone-age-days", type=float, default=30)
    ap.add_argument("--min-div-stages", type=int, default=2)
    ap.add_argument("--no-retest-div", action="store_true")
    ap.add_argument("--risk-pct", type=float, default=0.005)
    ap.add_argument("--tp-mode", default="next_close",
                    choices=["next_close", "fvg_far", "r"])
    ap.add_argument("--tp-r", type=float, default=2.0)
    ap.add_argument("--min-rr", type=float, default=0.75)
    ap.add_argument("--entry-mode", default="after",
                    choices=["after", "anticipate"])
    ap.add_argument("--entry-proximity-atr", type=float, default=0.5)
    ap.add_argument("--no-footprint", action="store_true")
    ap.add_argument("--fp-imb-min", type=float, default=0.15)
    ap.add_argument("--retest-window", type=int, default=5)
    ap.add_argument("--confirm-from-approach", action="store_true")
    ap.add_argument("--min-stop-atr", type=float, default=0.5)
    ap.add_argument("--fee-bps", type=float, default=5.0)
    ap.add_argument("--slip-bps", type=float, default=1.0)
    ap.add_argument("--time-stop-bars", type=int, default=96)
    ap.add_argument("--confirm-window", type=int, default=5)
    ap.add_argument("--fvg-max-age", type=int, default=500)
    ap.add_argument("--require-absorb", action="store_true",
                    help="only enter if absorption printed in the confirm window")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config(tf=args.ltf, profile_window=args.profile_window,
                 profile_tick=args.profile_tick, min_score=args.min_score,
                 min_div_stages=args.min_div_stages,
                 require_retest_div=not args.no_retest_div,
                 risk_pct=args.risk_pct,
                 tp_mode=args.tp_mode, tp_r=args.tp_r, min_rr=args.min_rr,
                 entry_mode=args.entry_mode,
                 entry_proximity_atr=args.entry_proximity_atr,
                 use_footprint=not args.no_footprint,
                 fp_imb_min=args.fp_imb_min,
                 retest_window=args.retest_window,
                 min_stop_atr=args.min_stop_atr,
                 fee_bps=args.fee_bps,
                 slip_bps=args.slip_bps,
                 time_stop_bars=args.time_stop_bars,
                 confirm_window=args.confirm_window,
                 fvg_max_age=args.fvg_max_age,
                 require_absorb=args.require_absorb,
                 confirm_from_retest=not args.confirm_from_approach)
    tfs = [x.strip() for x in args.tfs.split(",") if x.strip()]
    tfs = [t for t in HIERARCHY if t in tfs]   # enforce highest-first order

    print("scanning hierarchy:", " -> ".join(tfs), " | LTF:", args.ltf)
    ltf_df, zones_by_tf, _ = build_zones(args.data, tfs, args.ltf, cfg, args)
    fp_map = {} if args.no_footprint else load_footprint(args.data, args.ltf)
    if fp_map:
        print(f"footprint: {len(fp_map)} bars loaded")
    result = simulate(ltf_df, zones_by_tf, tfs, cfg, args, fp_map)
    out = args.out or os.path.join(args.data, f"lvn_fvg_mtf_{args.ltf}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in result.items() if k != "trades_detail"},
                     indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
