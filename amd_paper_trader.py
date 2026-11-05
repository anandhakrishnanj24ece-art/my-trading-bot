"""
================================================================================
AMD PAPER TRADER — Smart Money Concepts (Accumulation / Manipulation / Distribution)
Paper-trading engine with state persistence for GitHub Actions cron execution.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from amd_futures_backtest import (
    FUTURES_ASSETS, INITIAL_CAPITAL, RISK_PER_TRADE_PCT,
    TAKER_FEE, SLIPPAGE_BPS, FUNDING_RATE_8H,
    fetch_history, amd_signals, detect_regime,
)

STATE_FILE = Path("paper_state.json")
EQUITY_FILE = Path("equity_history.json")


def load_state(symbol: str, initial_capital: float) -> dict:
    """Load persisted state for a symbol, or create default."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                all_state = json.load(f)
            if symbol in all_state:
                return all_state[symbol]
        except Exception:
            pass
    return {
        "symbol": symbol,
        "equity": initial_capital,
        "peak_equity": initial_capital,
        "position_shares": 0.0,
        "position_side": 0,
        "position_entry": 0.0,
        "position_notional": 0.0,
        "sl": 0.0,
        "tp": 0.0,
        "partial_taken": False,
        "trading_halted": False,
        "daily_pnl": 0.0,
        "_last_date": None,
        "trade_log": [],
        "equity_history": [],
    }


def save_state(all_state: dict) -> None:
    """Persist state to disk."""
    with open(STATE_FILE, "w") as f:
        json.dump(all_state, f, indent=2, default=str)


def append_equity_history(symbol: str, history: list) -> None:
    """Append equity history points."""
    existing = []
    if EQUITY_FILE.exists():
        try:
            with open(EQUITY_FILE, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.extend(history)
    with open(EQUITY_FILE, "w") as f:
        json.dump(existing, f, indent=2, default=str)


class PaperTrader:
    def __init__(
        self,
        symbol: str,
        initial_capital: float = INITIAL_CAPITAL,
        risk_per_trade_pct: float = RISK_PER_TRADE_PCT,
        rr_target: float = 3.0,
        max_drawdown_pct: float = 0.20,
        dd_reduce_threshold: float = 0.15,
        dd_reduce_factor: float = 0.5,
        live: bool = False,
    ):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.risk_per_trade_pct = risk_per_trade_pct
        self.rr_target = rr_target
        self.max_drawdown_pct = max_drawdown_pct
        self.dd_reduce_threshold = dd_reduce_threshold
        self.dd_reduce_factor = dd_reduce_factor
        self.live = live

        # Load persisted state
        st = load_state(symbol, initial_capital)
        self.equity = st["equity"]
        self.peak_equity = st["peak_equity"]
        self.position_shares = st["position_shares"]
        self.position_side = st["position_side"]
        self.position_entry = st["position_entry"]
        self.position_notional = st["position_notional"]
        self.sl = st["sl"]
        self.tp = st["tp"]
        self.partial_taken = st["partial_taken"]
        self.trading_halted = st["trading_halted"]
        self.daily_pnl = st.get("daily_pnl", 0.0)
        self._last_date = st.get("_last_date")
        self.trade_log = st.get("trade_log", [])
        self.equity_history = st.get("equity_history", [])

    def run_paper(self, lookback_days: int = 30) -> dict:
        df = fetch_history(self.symbol, period="60d", interval="1h")
        regime_feat = detect_regime(df)
        df = df.join(regime_feat[["regime", "atr_percentile", "price_vs_sma200"]])
        df = df.dropna(subset=["regime"])

        sig = amd_signals(df, rr_target=self.rr_target)

        cutoff = df.index[-1] - timedelta(days=lookback_days)
        paper_df = df[df.index >= cutoff].copy()
        paper_sig = sig[sig.index >= cutoff]

        print(f"\n{'='*60}")
        print(f" PAPER TRADING — {self.symbol} {'(LIVE MODE)' if self.live else ''}")
        print(f"{'='*60}")
        print(f" Paper period : {paper_df.index[0]} to {paper_df.index[-1]}")
        print(f" Bars         : {len(paper_df)}")
        print(f" Initial capital : ${self.initial_capital:,.2f}")
        print(f" Current equity  : ${self.equity:,.2f}")
        print(f" Risk per trade  : {self.risk_per_trade_pct*100:.1f}%")
        print(f" Max DD halt     : {self.max_drawdown_pct*100:.0f}%")
        print(f" DD reduce       : {self.dd_reduce_threshold*100:.0f}% -> {self.dd_reduce_factor*100:.0f}% size")
        print(f" Open position   : {self.position_side:+d} @ {self.position_entry:.2f} ({self.position_shares:.4f} shares)")
        print(f"{'='*60}\n")

        close = paper_df["close"]
        new_equity_points = []

        for i in range(1, len(paper_df)):
            ts = paper_df.index[i]
            px = float(close.iloc[i])

            ts_date = ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()
            if self._last_date is None:
                self._last_date = ts_date
                self.daily_pnl = 0.0
                self.partial_taken = False
            elif ts_date != self._last_date:
                self._last_date = ts_date
                self.daily_pnl = 0.0
                self.partial_taken = False

            position_value = self.position_shares * px if self.position_shares != 0 else 0.0
            total_equity = self.equity + position_value
            if total_equity > self.peak_equity:
                self.peak_equity = total_equity

            current_dd = (self.peak_equity - total_equity) / self.peak_equity

            if current_dd >= self.max_drawdown_pct:
                self.trading_halted = True

            if self.trading_halted:
                effective_risk_pct = 0.0
            elif current_dd >= self.dd_reduce_threshold:
                effective_risk_pct = self.risk_per_trade_pct * self.dd_reduce_factor
            else:
                effective_risk_pct = self.risk_per_trade_pct

            exit_reason = None
            if self.position_shares != 0:
                if self.position_side > 0:
                    if px <= self.sl:
                        exit_reason = "STOP_LOSS"
                    elif px >= self.tp and not self.partial_taken:
                        exit_reason = "TAKE_PROFIT"

                    if not self.partial_taken:
                        partial_px = self.position_entry + abs(self.position_entry - self.sl) * 1.5
                        if px >= partial_px:
                            close_qty = self.position_shares * 0.5
                            exit_notional = close_qty * px
                            cost = exit_notional * (TAKER_FEE + SLIPPAGE_BPS / 10000)
                            realized = exit_notional - cost - (close_qty * self.position_entry)
                            self.equity += realized
                            self.daily_pnl += realized
                            self.trade_log.append({
                                "timestamp": str(ts), "symbol": self.symbol, "side": "PARTIAL_TP",
                                "price": px, "notional": exit_notional, "cost": cost,
                                "qty": close_qty, "equity_after": self.equity,
                                "reason": "PARTIAL_TP_1.5R"
                            })
                            self.position_shares -= close_qty
                            self.partial_taken = True
                            self.sl = self.position_entry

                    s_now = float(paper_sig["signal"].iloc[i])
                    if s_now != 0 and np.sign(s_now) != self.position_side:
                        exit_reason = "SIGNAL_FLIP"

                else:
                    if px >= self.sl:
                        exit_reason = "STOP_LOSS"
                    elif px <= self.tp and not self.partial_taken:
                        exit_reason = "TAKE_PROFIT"

                    if not self.partial_taken:
                        partial_px = self.position_entry - abs(self.position_entry - self.sl) * 1.5
                        if px <= partial_px:
                            close_qty = self.position_shares * 0.5
                            exit_notional = close_qty * px
                            cost = exit_notional * (TAKER_FEE + SLIPPAGE_BPS / 10000)
                            realized = (close_qty * self.position_entry) - (exit_notional + cost)
                            self.equity += realized
                            self.daily_pnl += realized
                            self.trade_log.append({
                                "timestamp": str(ts), "symbol": self.symbol, "side": "PARTIAL_TP",
                                "price": px, "notional": exit_notional, "cost": cost,
                                "qty": close_qty, "equity_after": self.equity,
                                "reason": "PARTIAL_TP_1.5R"
                            })
                            self.position_shares -= close_qty
                            self.partial_taken = True
                            self.sl = self.position_entry

                    s_now = float(paper_sig["signal"].iloc[i])
                    if s_now != 0 and np.sign(s_now) != self.position_side:
                        exit_reason = "SIGNAL_FLIP"

                if exit_reason is not None:
                    exit_notional = self.position_shares * px
                    cost = exit_notional * (TAKER_FEE + SLIPPAGE_BPS / 10000)
                    if self.position_side > 0:
                        realized = exit_notional - cost - (self.position_shares * self.position_entry)
                    else:
                        realized = (self.position_shares * self.position_entry) - (exit_notional + cost)
                    self.equity += realized
                    self.daily_pnl += realized
                    self.trade_log.append({
                        "timestamp": str(ts), "symbol": self.symbol, "side": "EXIT",
                        "price": px, "notional": exit_notional, "cost": cost,
                        "qty": self.position_shares, "equity_after": self.equity,
                        "reason": exit_reason
                    })
                    print(f"  {ts} | EXIT {self.position_side:+d} @ {px:.2f} | "
                          f"Reason: {exit_reason} | Equity: ${self.equity:,.2f}")
                    self.position_shares = 0.0
                    self.position_side = 0
                    self.position_entry = 0.0
                    self.sl = 0.0
                    self.tp = 0.0
                    self.partial_taken = False

            s = float(paper_sig["signal"].iloc[i])
            if s != 0 and self.position_shares == 0 and not self.trading_halted:
                entry_px = px
                entry_sl = float(paper_sig["stop_loss"].iloc[i]) if not np.isnan(paper_sig["stop_loss"].iloc[i]) else px
                entry_tp = float(paper_sig["take_profit"].iloc[i]) if not np.isnan(paper_sig["take_profit"].iloc[i]) else px

                if (s > 0 and entry_sl >= entry_px) or (s < 0 and entry_sl <= entry_px):
                    continue

                risk_amount = total_equity * effective_risk_pct
                price_risk = abs(entry_px - entry_sl)
                if price_risk <= 0:
                    continue

                notional = risk_amount / price_risk * entry_px
                cap_notional = total_equity * 0.20
                notional = min(notional, cap_notional)
                if notional <= 0:
                    continue

                cost = notional * (TAKER_FEE + SLIPPAGE_BPS / 10000)
                self.equity -= cost

                shares = notional / entry_px
                self.position_shares = shares
                self.position_side = int(s)
                self.position_entry = entry_px
                self.position_notional = notional
                self.sl = entry_sl
                self.tp = entry_tp
                self.partial_taken = False

                self.trade_log.append({
                    "timestamp": str(ts), "symbol": self.symbol,
                    "side": "BUY" if s > 0 else "SELL",
                    "price": entry_px, "notional": notional, "cost": cost,
                    "qty": shares, "sl": entry_sl, "tp": entry_tp,
                    "confluence": float(paper_sig["confluence"].iloc[i]),
                    "equity_after": self.equity
                })
                print(f"  {ts} | ENTRY {'LONG' if s > 0 else 'SHORT'} @ {entry_px:.2f} | "
                      f"SL: {entry_sl:.2f} TP: {entry_tp:.2f} | Notional: ${notional:,.2f}")

            if self.position_shares != 0:
                funding = abs(self.position_notional) * FUNDING_RATE_8H / 3
                self.equity -= funding
                self.daily_pnl -= funding

            total_equity = self.equity + (self.position_shares * px if self.position_shares != 0 else 0.0)
            new_equity_points.append({"timestamp": str(ts), "symbol": self.symbol, "equity": total_equity})

        # Save state
        all_state = {}
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r") as f:
                    all_state = json.load(f)
            except Exception:
                all_state = {}
        all_state[self.symbol] = {
            "symbol": self.symbol,
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "position_shares": self.position_shares,
            "position_side": self.position_side,
            "position_entry": self.position_entry,
            "position_notional": self.position_notional,
            "sl": self.sl,
            "tp": self.tp,
            "partial_taken": self.partial_taken,
            "trading_halted": self.trading_halted,
            "daily_pnl": self.daily_pnl,
            "_last_date": str(self._last_date) if self._last_date else None,
            "trade_log": self.trade_log,
            "equity_history": self.equity_history,
        }
        save_state(all_state)
        append_equity_history(self.symbol, new_equity_points)

        self._print_report(paper_df)

        return {
            "symbol": self.symbol,
            "final_equity": self.equity,
            "total_return": (self.equity / self.initial_capital - 1) * 100,
            "n_trades": len(self.trade_log),
            "trade_log": self.trade_log,
            "equity_history": self.equity_history,
        }

    def _print_report(self, paper_df: pd.DataFrame):
        print(f"\n{'='*60}")
        print(f" PAPER TRADE RESULTS — {self.symbol}")
        print(f"{'='*60}")
        print(f" Initial Capital : ${self.initial_capital:,.2f}")
        print(f" Final Equity    : ${self.equity:,.2f}")
        print(f" Total Return    : {(self.equity / self.initial_capital - 1) * 100:+.2f}%")
        print(f" N Trades        : {len(self.trade_log)}")
        print(f" Trading Halted  : {self.trading_halted}")
        print(f" Peak Equity     : ${self.peak_equity:,.2f}")
        print(f" Max DD          : {(self.peak_equity - self.equity) / self.peak_equity * 100:.2f}%")
        print(f"{'='*60}\n")


def run_paper_all(symbols: list[str], lookback_days: int = 30,
                  initial_capital: float = INITIAL_CAPITAL) -> None:
    print("\n" + "=" * 60)
    print(" AMD PAPER TRADER — ALL ASSETS")
    print(" Smart Money Concepts | Paper Mode | Fees + Slippage + Funding")
    print("=" * 60)

    for symbol in symbols:
        try:
            trader = PaperTrader(
                symbol,
                initial_capital=initial_capital,
                risk_per_trade_pct=RISK_PER_TRADE_PCT,
                rr_target=3.0,
                max_drawdown_pct=0.20,
                dd_reduce_threshold=0.15,
                dd_reduce_factor=0.5,
                live=True,
            )
            trader.run_paper(lookback_days=lookback_days)
        except Exception as e:
            print(f"  [skip] {symbol}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMD Paper Trader")
    parser.add_argument("--live", action="store_true", help="Live/paper trading mode")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL, help="Initial capital per symbol")
    parser.add_argument("--symbols", nargs="+", default=FUTURES_ASSETS, help="Symbols to trade")
    parser.add_argument("--lookback", type=int, default=30, help="Lookback days")
    args = parser.parse_args()

    run_paper_all(args.symbols, lookback_days=args.lookback, initial_capital=args.capital)
