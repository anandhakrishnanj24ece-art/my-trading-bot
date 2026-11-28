"""Paper-only ledger and one-tick execution for FVG Duo.

There is no broker adapter, API key, or live-order code in this package.  The
runner reads public NQ futures candles, writes a local/committed paper ledger,
and never sends an order.
"""
from __future__ import annotations

import json
from datetime import datetime, time, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .signals import (
    OD_STOP, RR, TICK, TZ, build_signals, completed_5m, normalize_5m,
    signal_day,
)

PAPER_ONLY = True
STARTING_CAPITAL = 50_000.0
POINT_VALUE = 1.0          # USATECH Nasdaq-100 CFD, $1 per index point
FEE_PER_SIDE = 1.0         # all-in spread + slippage, USD per lot per side
SLIPPAGE_TICKS = 0         # fee already includes execution cost
MAX_ENTRY_DELAY_SEC = 330  # one 5m bar + scheduler jitter

#: Sizing rule: 2% of current equity risked per trade, in 0.01-lot steps.
#: Contract size is derived from the stop distance, so every trade risks the
#: same fraction of the account regardless of stop width.  On the deployed
#: NDX CFD feed this is 93.7% pass / 5.8% fail / ~16 bd median for 0.625R
#: (4% target, 10% static floor, $2 round-trip).  The 88.4% figure is the
#: stacked NDX+NQ backtest, not a deployable stream.
RISK_PCT = 0.02
LOT_STEP = 0.01
MAX_LOTS = 50.0
MIN_LOTS = 0.01

DEFAULT_STATE_DIR = Path(__file__).resolve().parent / "state"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def _load_json(path: Path, default: dict) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return dict(default)


class PaperEngine:
    def __init__(self, base_dir: str | Path | None = None,
                 starting_capital: float = STARTING_CAPITAL) -> None:
        if not PAPER_ONLY:
            raise RuntimeError("paper-only guard failed")
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_STATE_DIR
        self.state_dir = self.base_dir
        self.state_path = self.state_dir / "paper_state.json"
        self.control_path = self.state_dir / "control.json"
        self.trades_csv_path = self.state_dir / "paper_trades.csv"
        self.state = _load_json(
            self.state_path,
            {
                "capital": starting_capital,
                "cash": starting_capital,
                "equity": starting_capital,
                "realized_pnl": 0.0,
                "positions": [],
                "trades": [],
                "equity_curve": [],
                "asof": None,
                "last_update": None,
                "last_price": None,
                "last_decision": None,
                "last_reason": None,
                "od_days": [],
            },
        )
        self.control = _load_json(
            self.control_path,
            {"halted": False, "reason": "", "updated_at": None},
        )
        self.state.setdefault("capital", starting_capital)
        self.state.setdefault("cash", starting_capital)
        self.state.setdefault("equity", starting_capital)
        self.state.setdefault("realized_pnl", 0.0)
        self.state.setdefault("positions", [])
        self.state.setdefault("trades", [])
        self.state.setdefault("equity_curve", [])
        self.state.setdefault("od_days", [])
        self.state["config"] = {"rr": RR, "risk_pct": RISK_PCT * 100,
                                "max_lots": MAX_LOTS, "point_value": POINT_VALUE}
        self._dry_run = False
        if not self.state["equity_curve"]:
            self.state["equity_curve"].append(
                {"timestamp": _now_iso(),
                 "equity": float(self.state["equity"])}
            )

    def save(self) -> None:
        if self._dry_run:
            return
        _save_json(self.state_path, self.state)
        _save_json(self.control_path, self.control)
        if self.state["trades"]:
            pd.DataFrame(self.state["trades"]).to_csv(
                self.trades_csv_path, index=False
            )

    def _touch(self, when: datetime | pd.Timestamp) -> None:
        """Record that the paper runner was alive, without changing a signal."""
        self.state["asof"] = self._utc(when).isoformat()

    def heartbeat(self, now: datetime | pd.Timestamp | None = None,
                  dry_run: bool = False) -> dict:
        """Write a freshness timestamp without creating a trading decision.

        This is for the book layer, which refuses to allocate to a sleeve with
        no ``asof``.  It never opens, closes, marks, or resizes a position.
        """
        if not PAPER_ONLY:
            raise RuntimeError("paper-only guard failed")
        self._dry_run = bool(dry_run)
        when_utc = self._utc(now or datetime.now(timezone.utc))
        self._touch(when_utc)
        self.state["last_update"] = _now_iso()
        self.state["last_decision"] = "heartbeat"
        self.state["last_reason"] = "runner alive; no trading decision"
        self.save()
        return {"action": "heartbeat", "asof": self.state["asof"],
                "stats": self.stats()}

    def is_halted(self) -> bool:
        return bool(self.control.get("halted", False))

    def stats(self) -> dict:
        trades = self.state["trades"]
        wins = [t for t in trades if float(t.get("net_usd", 0)) > 0]
        losses = [t for t in trades if float(t.get("net_usd", 0)) < 0]
        gp = sum(float(t.get("net_usd", 0)) for t in wins)
        gl = abs(sum(float(t.get("net_usd", 0)) for t in losses))
        peak = -np.inf
        max_dd = 0.0
        for p in self.state.get("equity_curve", []):
            e = float(p.get("equity", 0))
            peak = max(peak, e)
            if peak > 0:
                max_dd = max(max_dd, (peak - e) / peak)
        positions = self.state.get("positions", [])
        unrealized = sum(float(p.get("unrealized_usd", 0))
                         for p in positions)
        return {
            "equity": float(self.state.get("equity", STARTING_CAPITAL)),
            "cash": float(self.state.get("cash", STARTING_CAPITAL)),
            "realized_pnl": float(self.state.get("realized_pnl", 0.0)),
            "unrealized_pnl": unrealized,
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": (len(wins) / len(trades) * 100) if trades else 0.0,
            "gross_profit_usd": gp,
            "gross_loss_usd": gl,
            "profit_factor": (gp / gl) if gl > 0 else None,
            "max_drawdown_pct": max_dd * 100,
            "open_positions": len(positions),
            "halted": self.is_halted(),
            "clamped_trades": sum(1 for t in trades
                                  if t.get("sizing_clamped")),
            "risk_pct_target": RISK_PCT * 100,
        }

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _fill_bar(when: pd.Timestamp) -> pd.Timestamp:
        """Label of the 5-minute bar that was forming when the fill happened.

        Entry is filled at ``now`` from the live close, so part of this bar
        traded before the entry price existed.  The exit scan starts after it.
        """
        t = pd.Timestamp(when)
        bar = t.floor("5min")
        if bar == t:
            bar = bar - pd.Timedelta(minutes=5)
        return bar

    @staticmethod
    def _utc(now: datetime | pd.Timestamp) -> pd.Timestamp:
        t = pd.Timestamp(now)
        if t.tzinfo is None:
            t = t.tz_localize(TZ)
        return t.tz_convert("UTC")

    def _latest_close(self, bars: pd.DataFrame,
                      now: pd.Timestamp) -> float | None:
        usable = bars[bars.index <= now]
        if usable.empty:
            return None
        return float(usable.iloc[-1]["close"])

    def _previous_close(self, bars: pd.DataFrame,
                        ts: pd.Timestamp) -> float | None:
        earlier = bars[bars.index < ts]
        if earlier.empty:
            return None
        return float(earlier.iloc[-1]["close"])

    def _append_equity(self, when: pd.Timestamp) -> None:
        eq = float(self.state.get("equity", 0.0))
        curve = self.state.setdefault("equity_curve", [])
        if curve and abs(float(curve[-1].get("equity", 0)) - eq) < 1e-9:
            return
        curve.append({"timestamp": when.isoformat(), "equity": eq})
        if len(curve) > 5000:
            del curve[: len(curve) - 5000]

    def _size(self, equity: float, risk_points: float) -> dict:
        """Lots risking RISK_PCT of ``equity`` over ``risk_points`` of stop.

        Returns the sized lots together with the uncapped request and whether
        MAX_LOTS or MIN_LOTS bound it, so a capped trade is never silent.
        """
        if equity <= 0 or risk_points <= 0:
            return {"lots": MIN_LOTS, "raw": MIN_LOTS, "clamped": "no_equity",
                    "risk_pct": 0.0}
        budget = float(equity) * RISK_PCT
        raw = budget / (float(risk_points) * POINT_VALUE)
        raw = float(np.floor(raw / LOT_STEP + 1e-9) * LOT_STEP)
        lots = float(min(max(raw, MIN_LOTS), MAX_LOTS))
        reason = None
        if raw > MAX_LOTS + 1e-9:
            reason = "max_lots"
        elif raw < MIN_LOTS - 1e-9:
            reason = "min_lots"
        return {"lots": lots, "raw": raw, "clamped": reason,
                "risk_pct": round(100 * risk_points * POINT_VALUE * lots / float(equity), 3)}

    def _resize(self, pos: dict) -> None:
        """Re-derive lots after the stop distance changes."""
        sz = self._size(float(self.state.get("equity", 0.0)),
                        float(pos.get("risk_points", 0.0)))
        pos["contracts"] = sz["lots"]
        pos["contracts_raw"] = sz["raw"]
        pos["sizing_clamped"] = sz["clamped"]
        pos["risk_pct_target"] = RISK_PCT * 100
        pos["risk_pct_actual"] = sz["risk_pct"]

    def _flag_sizing(self, result: dict) -> None:
        """Surface a clamped lot size in the tick output and the last reason."""
        if not self.state.get("positions"):
            return
        pos = self.state["positions"][0]
        if not pos.get("sizing_clamped"):
            return
        note = (f"LOT CAP {pos.get('sizing_clamped')}: sized "
                f"{float(pos.get('contracts', 0)):g} of "
                f"{float(pos.get('contracts_raw', 0)):g} lots; risk "
                f"{pos.get('risk_pct_actual')}% not "
                f"{pos.get('risk_pct_target')}%")
        result["sizing_warning"] = note
        prior = self.state.get("last_reason") or ""
        self.state["last_reason"] = f"{prior} | {note}" if prior else note

    def _close(self, price: float, when: pd.Timestamp, reason: str) -> dict:
        pos = self.state["positions"][0]
        d = int(pos["direction_sign"])
        lots = float(pos.get("contracts", 1.0)) or 1.0
        pts = (price - float(pos["entry_price"])) * d
        net_pts = pts - (2 * FEE_PER_SIDE / POINT_VALUE)
        gross_usd = pts * POINT_VALUE * lots
        net_usd = gross_usd - 2 * FEE_PER_SIDE * lots
        risk_pts = max(float(pos.get("risk_points", 0.0)), 1e-9)
        trade = {
            "strategy": "FVG Duo",
            "source": pos.get("source", ""),
            "direction": pos["direction"],
            "entry_time": pos["entry_time"],
            "entry_bar": pos.get("entry_bar"),
            "signal_time": pos.get("signal_time", pos["entry_time"]),
            "closed_at": when.isoformat(),
            "entry_price": round(float(pos["entry_price"]), 4),
            "exit_price": round(float(price), 4),
            "stop_price": round(float(pos["stop_price"]), 4),
            "target_price": (round(float(pos["target_price"]), 4)
                             if pos.get("target_price") is not None else None),
            "contracts": round(lots, 4),
            "contracts_raw": round(float(pos.get("contracts_raw", lots)), 4),
            "sizing_clamped": pos.get("sizing_clamped"),
            "risk_usd": round(risk_pts * POINT_VALUE * lots, 2),
            "risk_pct_actual": pos.get("risk_pct_actual"),
            "gross_points": round(pts, 4),
            "net_points": round(net_pts, 4),
            "gross_usd": round(gross_usd, 2),
            "net_usd": round(net_usd, 2),
            "r_gross": round(pts / risk_pts, 4),
            "r_net": round(net_pts / risk_pts, 4),
            "reason": reason,
        }
        self.state["positions"] = []
        self.state["trades"].append(trade)
        self.state["cash"] = round(float(self.state["cash"]) + net_usd, 2)
        self.state["realized_pnl"] = round(
            float(self.state["realized_pnl"]) + net_usd, 2
        )
        self.state["equity"] = round(float(self.state["cash"]), 2)
        self.state["last_price"] = round(float(price), 4)
        self._append_equity(when)
        return trade

    def _open(self, direction: int, entry: float, stop: float,
              target: float | None, source: str, when: pd.Timestamp,
              risk_points: float, signal_time: pd.Timestamp | None = None) -> dict:
        sz = self._size(float(self.state.get("equity")), risk_points)
        pos = {
            "direction": "long" if direction == 1 else "short",
            "direction_sign": direction,
            "entry_price": round(float(entry), 4),
            "stop_price": round(float(stop), 4),
            "target_price": (round(float(target), 4)
                             if target is not None else None),
            "risk_points": round(float(risk_points), 4),
            "contracts": sz["lots"],
            "contracts_raw": sz["raw"],
            "sizing_clamped": sz["clamped"],
            "risk_pct_target": RISK_PCT * 100,
            "risk_pct_actual": sz["risk_pct"],
            "contract": "USATECH.IDX/USD",
            "source": source,
            "entry_time": when.isoformat(),
            "entry_bar": self._fill_bar(self._utc(when)).isoformat(),
            "signal_time": (pd.Timestamp(signal_time).isoformat()
                            if signal_time is not None else when.isoformat()),
            "last_price": round(float(entry), 4),
            "unrealized_points": 0.0,
            "unrealized_usd": 0.0,
        }
        self.state["positions"] = [pos]
        self.state["last_price"] = round(float(entry), 4)
        return pos

    def _mark(self, price: float, when: pd.Timestamp) -> None:
        pos = self.state["positions"][0]
        d = int(pos["direction_sign"])
        pts = (price - float(pos["entry_price"])) * d
        lots = float(pos.get("contracts", 1.0)) or 1.0
        pos["last_price"] = round(float(price), 4)
        pos["unrealized_points"] = round(pts, 4)
        pos["unrealized_usd"] = round(pts * POINT_VALUE * lots, 2)
        self.state["equity"] = round(
            float(self.state["cash"]) + float(pos["unrealized_usd"]), 2
        )
        self.state["last_price"] = round(float(price), 4)
        self._append_equity(when)

    # ---------------------------------------------------------------- tick
    def tick(self, now: datetime | pd.Timestamp, bars: pd.DataFrame,
             signals=None, dry_run: bool = False) -> dict:
        now_utc = self._utc(now)
        now_et = now_utc.tz_convert(TZ)
        bars = normalize_5m(bars)
        if signals is None:
            signals = build_signals(completed_5m(bars, now_utc))
        od, trend = signals

        result = {
            "time_et": now_et.isoformat(),
            "action": "none",
            "reason": "",
            "trade": None,
            "position": None,
            "stats": self.stats(),
        }
        self._dry_run = bool(dry_run)

        # Always persist initial files so GitHub has a state directory.
        self.save()

        if bars.empty:
            result["reason"] = "no data"
            result["stats"] = self.stats()
            self.save()
            return result

        latest = self._latest_close(bars, now_utc)
        if latest is None:
            result["reason"] = "no completed price"
            self.save()
            return result

        # ---- manage an existing position
        completed = completed_5m(bars, now_utc)
        if self.state["positions"]:
            pos = self.state["positions"][0]
            # Start the exit scan after the bar that was forming at the fill:
            # that bar's extremes can predate the entry price, which would book
            # a stop or target the market had already printed.
            if pos.get("entry_bar"):
                entry_bar = pd.Timestamp(pos["entry_bar"])
                if entry_bar.tzinfo is None:
                    entry_bar = entry_bar.tz_localize(TZ)
            else:
                entry_bar = self._fill_bar(pd.Timestamp(pos["entry_time"]))
            entry_bar = entry_bar.tz_convert("UTC")
            after = completed[(completed.index > entry_bar)
                              & (completed.index <= now_utc)]
            exit_price = None
            reason = ""
            exit_ts = now_utc
            for ts, bar in after.iterrows():
                if pos["direction"] == "long":
                    if float(bar["low"]) <= float(pos["stop_price"]):
                        exit_price, reason, exit_ts = (
                            float(pos["stop_price"]), "stop", ts
                        )
                        break
                    if (pos.get("target_price") is not None
                            and float(bar["high"]) >= float(pos["target_price"])):
                        exit_price, reason, exit_ts = (
                            float(pos["target_price"]), "target", ts
                        )
                        break
                else:
                    if float(bar["high"]) >= float(pos["stop_price"]):
                        exit_price, reason, exit_ts = (
                            float(pos["stop_price"]), "stop", ts
                        )
                        break
                    if (pos.get("target_price") is not None
                            and float(bar["low"]) <= float(pos["target_price"])):
                        exit_price, reason, exit_ts = (
                            float(pos["target_price"]), "target", ts
                        )
                        break

            if exit_price is None and now_et.time() >= time(15, 55):
                exit_price, reason, exit_ts = latest, "eod", now_utc

            if exit_price is not None:
                trade = self._close(exit_price, exit_ts, reason)
                self._touch(now_utc)
                self.state["last_update"] = _now_iso()
                self.state["last_decision"] = "exit"
                self.state["last_reason"] = reason
                self.save()
                result.update(action="exit", reason=reason, trade=trade,
                              stats=self.stats())
                return result

            # OD interaction: a Trend long can be switched to OD rules, or
            # reversed if OD is short.
            today = now_et.date()
            od_days = set(str(x) for x in self.state.get("od_days", []))
            if (pos.get("source") == "trend"
                    and str(today) not in od_days
                    and time(9, 40) <= now_et.time() <= time(10, 0)):
                cands = sorted(
                    t for t in od
                    if signal_day(t) == today and t <= now_utc
                )
                if cands:
                    sig_ts = cands[0]
                    age = (now_utc - sig_ts).total_seconds()
                    if age <= MAX_ENTRY_DELAY_SEC:
                        d = int(od[sig_ts])
                        if d == 1:
                            ref = self._previous_close(completed, sig_ts)
                            if ref is None:
                                ref = latest
                            stop = round(ref * (1 - OD_STOP), 4)
                            pos["stop_price"] = stop
                            pos["target_price"] = None
                            # The stop just moved, so the old risk distance and
                            # lot size no longer honour the 2% rule.  Re-derive
                            # both from entry to the new stop.
                            pos["risk_points"] = round(
                                abs(float(pos["entry_price"]) - stop), 4
                            )
                            self._resize(pos)
                            pos["source"] = "od"
                            pos["switched"] = True
                            self.state["od_days"].append(str(today))
                            self.state["od_days"] = self.state["od_days"][-60:]
                            self._mark(latest, now_utc)
                            self._touch(now_utc)
                            self.state["last_update"] = _now_iso()
                            self.state["last_decision"] = "switch_od"
                            self.state["last_reason"] = (
                                f"Trend long switched to OD rules, stop "
                                f"{stop:,.2f}, {pos['risk_points']:g} pts risk"
                            )
                            self._flag_sizing(result)
                            self.save()
                            result.update(action="switch",
                                          reason=self.state["last_reason"],
                                          position=self.state["positions"][0],
                                          stats=self.stats())
                            return result
                        else:
                            trade = self._close(latest, now_utc,
                                                "od_reverse")
                            entry = latest - SLIPPAGE_TICKS * TICK
                            stop = entry * (1 + OD_STOP)
                            self._open(-1, entry, stop, None, "od",
                                       now_utc, abs(entry - stop),
                                       signal_time=sig_ts)
                            self.state["od_days"].append(str(today))
                            self.state["od_days"] = self.state["od_days"][-60:]
                            self._touch(now_utc)
                            self.state["last_update"] = _now_iso()
                            self.state["last_decision"] = "reverse_od"
                            self.state["last_reason"] = (
                                "Trend long reversed to OD short"
                            )
                            self._flag_sizing(result)
                            self.save()
                            result.update(action="reverse",
                                          reason=self.state["last_reason"],
                                          trade=trade,
                                          position=self.state["positions"][0],
                                          stats=self.stats())
                            return result
                    else:
                        self.state["od_days"].append(str(today))
                        self.state["od_days"] = self.state["od_days"][-60:]

            self._mark(latest, now_utc)
            self._touch(now_utc)
            self.state["last_update"] = _now_iso()
            self.state["last_decision"] = "mark"
            self.state["last_reason"] = "open position"
            self.save()
            result.update(action="mark", position=self.state["positions"][0],
                          stats=self.stats())
            return result

        # ---- flat: entries
        if self.is_halted():
            result["reason"] = "halted: new paper entries blocked"
            self._touch(now_utc)
            self.state["last_update"] = _now_iso()
            self.state["last_decision"] = "halted"
            self.state["last_reason"] = result["reason"]
            self.save()
            result["stats"] = self.stats()
            return result

        if not (time(9, 30) <= now_et.time() <= time(15, 35)):
            # No position, no entry window.  Stay quiet during the session so
            # the GitHub runner does not commit every five minutes, but write
            # one post-close freshness stamp so the book layer can distinguish
            # "runner alive" from "runner stopped days ago".
            result["reason"] = "outside entry window"
            asof = self.state.get("asof")
            asof_date = self._utc(asof).tz_convert(TZ).date() if asof else None
            if asof_date != now_et.date() and now_et.time() >= time(15, 55):
                self._touch(now_utc)
                self.state["last_update"] = _now_iso()
                self.state["last_decision"] = "heartbeat"
                self.state["last_reason"] = result["reason"]
                self.save()
            result["stats"] = self.stats()
            return result

        today = now_et.date()
        od_days = set(str(x) for x in self.state.get("od_days", []))

        # ---- Opening Drive: at most one decision per day
        if str(today) not in od_days:
            cands = sorted(
                t for t in od
                if signal_day(t) == today and t <= now_utc
            )
            if cands:
                sig_ts = cands[0]
                age = (now_utc - sig_ts).total_seconds()
                if age > MAX_ENTRY_DELAY_SEC:
                    self.state["od_days"].append(str(today))
                    self.state["od_days"] = self.state["od_days"][-60:]
                    result["reason"] = (
                        f"OD signal {sig_ts.tz_convert(TZ).strftime('%H:%M')} "
                        f"missed by {age / 60:.1f} min"
                    )
                    self._touch(now_utc)
                    self.state["last_update"] = _now_iso()
                    self.state["last_decision"] = "skip"
                    self.state["last_reason"] = result["reason"]
                    self.save()
                    result["stats"] = self.stats()
                    return result

                d = int(od[sig_ts])
                entry = latest + d * SLIPPAGE_TICKS * TICK
                stop = entry * (1 - d * OD_STOP)
                self._open(d, entry, stop, None, "od", now_utc,
                           abs(entry - stop), signal_time=sig_ts)
                self.state["od_days"].append(str(today))
                self.state["od_days"] = self.state["od_days"][-60:]
                self._touch(now_utc)
                self.state["last_update"] = _now_iso()
                self.state["last_decision"] = "open_od"
                self.state["last_reason"] = (
                    f"OD {'long' if d == 1 else 'short'} signal "
                    f"{sig_ts.tz_convert(TZ).strftime('%H:%M')}"
                )
                self._flag_sizing(result)
                self.save()
                result.update(action="open", reason=self.state["last_reason"],
                              position=self.state["positions"][0],
                              stats=self.stats())
                return result

        # ---- Trend FVG: latest unexpired signal, never on an OD day
        if str(today) not in set(str(x) for x in self.state.get("od_days", [])):
            cands = [
                t for t in trend
                if signal_day(t) == today and t <= now_utc
                and (now_utc - t).total_seconds() <= MAX_ENTRY_DELAY_SEC
            ]
            if cands:
                sig_ts = max(cands)
                _, signal_stop = trend[sig_ts]
                entry = latest + SLIPPAGE_TICKS * TICK
                risk = entry - signal_stop
                if risk > 0:
                    target = entry + RR * risk
                    self._open(1, entry, signal_stop, target, "trend",
                               now_utc, risk, signal_time=sig_ts)
                    self._touch(now_utc)
                    self.state["last_update"] = _now_iso()
                    self.state["last_decision"] = "open_trend"
                    self.state["last_reason"] = (
                        "Trend bullish FVG "
                        f"{sig_ts.tz_convert(TZ).strftime('%H:%M')}"
                    )
                    self._flag_sizing(result)
                    self.save()
                    result.update(action="open",
                                  reason=self.state["last_reason"],
                                  position=self.state["positions"][0],
                                  stats=self.stats())
                    return result

        result["reason"] = "no entry signal"
        self._touch(now_utc)
        self.state["last_update"] = _now_iso()
        self.state["last_decision"] = "idle"
        self.state["last_reason"] = result["reason"]
        self.save()
        result["stats"] = self.stats()
        return result
