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
        return np.nan

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
        corrected_p[sorted_indices[i]] = min(1.0, sorted_p[i] * (n - i + 1))

    return corrected_p

def calculate_calmar_ratio(returns):
    """
    Calmar Ratio = Annualized Return / Max Drawdown.
    """
    if len(returns) == 0: return np.nan

    ann_return = returns.mean() * 252

    # Calculate Max Drawdown
    cum_rets = (1 + returns).cumprod()
    peak = cum_rets.cummax()
    drawdown = (peak - cum_rets) / peak
    max_dd = drawdown.max()

    if max_dd == 0:
        return np.inf if ann_return > 0 else -np.inf

    return ann_return / max_dd

def calculate_ulcer_index(returns):
    """
    Ulcer Index measures the depth and duration of drawdowns.
    UI = sqrt(mean(drawdown^2))
    """
    if len(returns) == 0: return np.nan

    cum_rets = (1 + returns).cumprod()
    peak = cum_rets.cummax()
    drawdown = (peak - cum_rets) / peak

    return np.sqrt(np.mean(drawdown**2))

def calculate_expected_shortfall(outcomes, percentile=0.05):
    """
    Conditional Value at Risk (CVaR) / Expected Shortfall.
    The average loss given that the loss is beyond the VaR threshold.
    """
    if len(outcomes) == 0: return np.nan

    var = np.percentile(outcomes, percentile * 100)
    tail_losses = outcomes[outcomes <= var]

    return np.mean(tail_losses) if len(tail_losses) > 0 else var

def calculate_profit_factor(returns):
    """
    Profit Factor = Sum of Gains / Sum of Losses.
    """
    if len(returns) == 0: return np.nan

    gains = returns[returns > 0].sum()
    losses = np.abs(returns[returns < 0].sum())

    if losses == 0:
        return np.inf if gains > 0 else 0.0

    return gains / losses
