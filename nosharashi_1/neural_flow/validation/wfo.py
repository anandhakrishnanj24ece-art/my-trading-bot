import pandas as pd
import numpy as np
from neural_flow.backtest import StrategyRunner

class WalkForwardOptimizer:
    """
    Implements Walk-Forward Optimization (WFO) to detect overfitting.
    """
    def __init__(self, runner, is_months=12, oos_months=3):
        self.runner = runner
        self.is_months = is_months
        self.oos_months = oos_months

    def run(self, close_prices, volumes):
        """
        Runs rolling WFO windows and calculates the WFO Ratio.
        """
        # Use a rough 21 trading days per month
        is_days = self.is_months * 21
        oos_days = self.oos_months * 21

        all_oos_returns = []
        all_is_returns = []

        # Total data available
        total_days = len(close_prices)

        # Start the first window
        current_start = 0
        while (current_start + is_days + oos_days) <= total_days:
            # 1. In-Sample Period
            is_close = close_prices.iloc[current_start : current_start + is_days]
            is_vol = volumes.iloc[current_start : current_start + is_days]

            # In a real WFO, we would optimize parameters here.
            # For this strategy, we run the base model to get the baseline IS performance.
            is_rets = self.runner.run(is_close, is_vol)
            all_is_returns.append(is_rets.mean() * 252)

            # 2. Out-of-Sample Period
            oos_close = close_prices.iloc[current_start + is_days : current_start + is_days + oos_days]
            oos_vol = volumes.iloc[current_start + is_days : current_start + is_days + oos_days]

            oos_rets = self.runner.run(oos_close, oos_vol)
            all_oos_returns.append(oos_rets.mean() * 252)

            # Shift window forward by OOS period (Rolling Window)
            current_start += oos_days

        # Calculate WFO Ratio
        avg_is_ret = np.mean(all_is_returns)
        avg_oos_ret = np.mean(all_oos_returns)

        wfo_ratio = avg_oos_ret / avg_is_ret if avg_is_ret != 0 else np.nan

        return {
            "is_annualized_ret": avg_is_ret,
            "oos_annualized_ret": avg_oos_ret,
            "wfo_ratio": wfo_ratio,
            "oos_series": pd.concat([pd.Series(all_oos_returns)]) # simplified
        }
