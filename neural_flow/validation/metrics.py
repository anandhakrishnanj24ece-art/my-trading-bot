import numpy as np
import pandas as pd

def calculate_sortino_ratio(returns, risk_free_rate=0.0):
    """
    Calculate the Sortino Ratio, which only penalizes downside volatility.
    """
    if len(returns) == 0: return np.nan

    expected_return = returns.mean() * 252
    downside_rets = returns[returns < 0]

    if len(downside_rets) == 0:
        return np.nan if expected_return <= 0 else np.inf

    downside_std = downside_rets.std() * np.sqrt(252)
    return (expected_return - risk_free_rate) / downside_std

def calculate_kelly_criterion(returns):
    """
    Calculate the Kelly Criterion optimal fraction f*.
    f* = (p/a) - (q/b)
    p: win probability, q: loss probability
    a: avg loss, b: avg win
    """
    if len(returns) == 0: return np.nan

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    p = len(wins) / len(returns)
    q = 1 - p

    if len(wins) == 0 or len(losses) == 0:
        return np.nan

    b = wins.mean()
    a = np.abs(losses.mean())

    return (p / a) - (q / b) if a != 0 else np.nan

def calculate_p_value(strategy_rets, benchmark_rets=None):
    """
    Hypothesis testing: Is the strategy return significantly different from a random walk?
    """
    if benchmark_rets is None:
        # Compare against 10k random portfolios
        # This is usually handled by the Simulation engine, but we provide a wrapper.
        return np.nan

    # Simple t-test p-value against benchmark
    from scipy import stats
    t_stat, p_val = stats.ttest_ind(strategy_rets, benchmark_rets)
    return p_val

def apply_holm_sidak_correction(p_values):
    """
    Corrects p-values for multiple testing to avoid p-hacking.
    """
    p_values = np.array(p_values)
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    corrected_p = np.zeros(n)
    for i in range(n):
        # Holm-Sidak: p_i * (n - i + 1)
        corrected_p[sorted_indices[i]] = min(1.0, sorted_p[i] * (n - i + 1))

    return corrected_p
