import numpy as np
import pandas as pd

class VolatilityTarget:
    """
    Continuous Volatility Targeting.
    Aims for a constant risk contribution by adjusting leverage inversely to realized vol.
    """
    def __init__(self, target_vol=0.15, max_cap=4.0):
        self.target_vol = target_vol
        self.max_cap = max_cap

    def get_leverage(self, returns: pd.Series) -> float:
        if len(returns) < 20:
            return 1.0

        # Realized 20-day volatility (annualized)
        realized_vol = returns.tail(20).std() * np.sqrt(252)

        if realized_vol == 0:
            return self.max_cap

        lev = self.target_vol / realized_vol
        return min(lev, self.max_cap)

class VolatilitySwitch:
    """
    Volatility Switch.
    Switches between base leverage and cautious leverage based on a volatility threshold.
    """
    def __init__(self, base_leverage=4.0, threshold=0.40):
        self.base_leverage = base_leverage
        self.threshold = threshold

    def get_leverage(self, returns: pd.Series) -> float:
        if len(returns) < 20:
            return 1.0

        realized_vol = returns.tail(20).std() * np.sqrt(252)

        # Switch leverage based on volatility threshold
        if realized_vol > self.threshold:
            return 1.0
        return self.base_leverage

class KellySizer:
    """
    Fractional Kelly Sizing.
    Protects against model overconfidence by scaling optimal f.
    """
    def __init__(self, fraction=0.25):
        self.fraction = fraction

    def calculate_f(self, returns: pd.Series) -> float:
        if len(returns) < 20:
            return 1.0

        wins = returns[returns > 0]
        losses = returns[returns < 0]

        if len(wins) == 0 or len(losses) == 0:
            return 1.0

        p = len(wins) / len(returns)
        q = 1 - p
        b = wins.mean()
        a = np.abs(losses.mean())

        if a == 0: return 1.0

        # Full Kelly: f = (p/a) - (q/b) is simplified; we use (p*b - q*a) / (b*a)
        # More robust: f = (expected_return) / (variance)
        full_kelly = (p * b - q * a) / (b * a) if (b * a) != 0 else 1.0

        return max(0.1, full_kelly * self.fraction)

class DrawdownGuard:
    """
    Linear Drawdown Scaling (The Circuit Breaker).
    Reduces leverage as the portfolio drifts from its High-Water Mark.
    """
    def __init__(self, caution_threshold=0.05, halt_threshold=0.15, floor=0.2):
        self.caution = caution_threshold
        self.halt = halt_threshold
        self.floor = floor
        self.peak_equity = 1.0

    def get_multiplier(self, current_equity: float) -> float:
        self.peak_equity = max(self.peak_equity, current_equity)
        dd = (self.peak_equity - current_equity) / self.peak_equity

        if dd < self.caution:
            return 1.0
        elif dd <= self.halt:
            # Linear decay from 1.0 to floor
            return 1.0 - ((dd - self.caution) / (self.halt - self.caution)) * (1.0 - self.floor)
        else:
            return self.floor

class StopLoss:
    """
    Per-Asset Stop Loss.
    Force closes a position if the price drops below a certain percentage from entry.
    """
    def __init__(self, stop_pct=0.10):
        self.stop_pct = stop_pct

    def should_exit(self, entry_price, current_price, side='long'):
        if side == 'long':
            return current_price < entry_price * (1 - self.stop_pct)
        else: # short
            return current_price > entry_price * (1 + self.stop_pct)

