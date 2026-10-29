import numpy as np
import pandas as pd

def circular_block_bootstrap(returns, block_size=5, n_sims=10000):
    """
    Implements Circular Block Bootstrapping to preserve volatility clustering.
    Returns a list of simulated return series.
    """
    n_days = len(returns)
    simulations = []

    # Convert to numpy for speed
    rets_array = returns.values if isinstance(returns, pd.Series) else returns

    for _ in range(n_sims):
        # Pick random starting points for blocks
        n_blocks = int(np.ceil(n_days / block_size))
        start_indices = np.random.randint(0, n_days, size=n_blocks)

        sim_series = []
        for start in start_indices:
            # Circular indexing
            indices = np.arange(start, start + block_size) % n_days
            sim_series.extend(rets_array[indices])

        # Trim to original length
        simulations.append(np.array(sim_series[:n_days]))

    return np.array(simulations)

def calculate_simulation_metrics(sim_returns):
    """
    Calculate VaR and MaxDD from a set of simulated return series.
    sim_returns: array of shape (n_sims, n_days)
    """
    # Convert daily returns to final portfolio values
    # Equity = Prod(1 + r)
    final_values = np.prod(1 + sim_returns, axis=1)

    # 5% VaR: the 5th percentile of final outcomes
    var_5 = np.percentile(final_values, 5)

    # Max Drawdown across simulations
    # For each sim, find the max drawdown
    equity_curves = np.cumprod(1 + sim_returns, axis=1)
    running_max = np.maximum.accumulate(equity_curves, axis=1)
    drawdowns = (running_max - equity_curves) / running_max

    if drawdowns.size == 0:
        return {"var_5": np.nan, "avg_max_dd": np.nan, "p5_max_dd": np.nan}

    max_drawdowns = np.max(drawdowns, axis=1)

    return {
        "var_5": var_5,
        "avg_max_dd": np.mean(max_drawdowns),
        "p5_max_dd": np.percentile(max_drawdowns, 5)
    }
