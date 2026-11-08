import yfinance as yf
import pandas as pd
import numpy as np
from neural_flow.backtest import StrategyRunner
from neural_flow.validation.metrics import (
    calculate_sortino_ratio,
    calculate_kelly_criterion,
    calculate_p_value,
    apply_holm_sidak_correction
)
from neural_flow.validation.simulation import circular_block_bootstrap, calculate_simulation_metrics
from neural_flow.validation.wfo import WalkForwardOptimizer

class ValidationSuite:
    """
    Professional Quantitative Audit Suite for Neural-Flow.
    """
    def __init__(self, tickers=None):
        self.tickers = tickers or ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOT-USD', 'POL-USD', 'LINK-USD', 'AVAX-USD']
        self.runner = StrategyRunner()

    def run_full_audit(self, years=5):
        """
        Executes the full validation pipeline: Multi-year analysis, WFO, MC Simulation, and Statistical Audit.
        """
        print(f"🚀 STARTING FULL QUANTITATIVE AUDIT (Past {years} Years)")
        print("="*60)

        # 1. Data Ingestion
        # Using a more stable set of tickers for 5y history
        tickers = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD']
        data_raw = yf.download(tickers, period=f'{years}y', interval='1d')

        # Use dropna(axis=1) to remove columns (tickers) that have any NaNs after ffill
        # This ensures a perfectly aligned matrix for the StrategyRunner
        close_prices_filled = data_raw['Close'].ffill()
        volumes_filled = data_raw['Volume'].ffill()

        # Find columns that have no NaNs across the whole period
        valid_tickers = close_prices_filled.columns[close_prices_filled.notna().all()].tolist()
        valid_tickers = [t for t in valid_tickers if t in volumes_filled.columns and volumes_filled[t].notna().all()]

        if not valid_tickers:
            print("❌ Error: No tickers have full 5y history. Try reducing 'years' or adjusting tickers.")
            return

        close_prices = close_prices_filled[valid_tickers]
        volumes = volumes_filled[valid_tickers]

        if close_prices.empty:
            print("❌ Error: No overlapping data found for tickers.")
            return

        # 2. Multi-Year Regime Analysis
        yearly_results = {}
        all_p_values = []

        for year in range(years):
            # Slice data for the specific year
            start_idx = (len(close_prices) - (years - year) * 252) # approximate
            # Better slice by date if possible, but for now using index slices
            # Correct slicing:
            end_idx = len(close_prices) - (years - 1 - year) * 252
            start_idx = max(0, end_idx - 252)

            y_close = close_prices.iloc[start_idx:end_idx]
            y_vol = volumes.iloc[start_idx:end_idx]

            if len(y_close) < 100: continue # Skip incomplete years

            # Run strategy
            rets = self.runner.run(y_close, y_vol)

            # Sortino & Kelly
            sortino = calculate_sortino_ratio(rets)
            kelly = calculate_kelly_criterion(rets)

            # Hypothesis Testing (vs Random Walk)
            # Generate 1000 random portfolios for this year to get a p-value
            random_rets = np.random.normal(rets.mean(), rets.std(), (1000, len(rets)))
            random_means = np.prod(1 + random_rets, axis=1)
            strategy_final = np.prod(1 + rets)
            p_val = np.sum(random_means >= strategy_final) / 1000

            all_p_values.append(p_val)

            yearly_results[f"Year {year+1}"] = {
                "return": (np.prod(1 + rets) - 1) * 100,
                "sortino": sortino,
                "kelly": kelly,
                "p_value": p_val
            }

        # Apply Multiple Testing Correction
        corrected_p = apply_holm_sidak_correction(all_p_values)

        for i, (year, res) in enumerate(yearly_results.items()):
            res["corrected_p"] = corrected_p[i]

        # 3. Walk-Forward Optimization (WFO)
        wfo = WalkForwardOptimizer(self.runner)
        wfo_results = wfo.run(close_prices, volumes)

        # 4. Volatility-Aware Monte Carlo (10k Sims)
        # Use the full strategy returns for the bootstrap
        full_rets = self.runner.run(close_prices, volumes)
        sim_returns = circular_block_bootstrap(full_rets, block_size=7, n_sims=10000)
        mc_metrics = calculate_simulation_metrics(sim_returns)

        # 5. Final Report Synthesis
        self._print_report(yearly_results, wfo_results, mc_metrics)

    def _print_report(self, yearly, wfo, mc):
        print("\n--- 📊 REGIME REPORT (Past 5 Years) ---")
        print(f"{'Year':<10} | {'Return':<10} | {'Sortino':<10} | {'Kelly':<10} | {'P-Val (Corr)':<12}")
        print("-" * 60)
        for year, res in yearly.items():
            print(f"{year:<10} | {res['return']:>9.2f}% | {res['sortino']:>9.4f} | {res['kelly']:>9.4f} | {res['corrected_p']:>11.4f}")

        print("\n--- 📈 WALK-FORWARD ANALYSIS ---")
        print(f"In-Sample Annualized:  {wfo['is_annualized_ret']*100:.2f}%")
        print(f"Out-of-Sample Annualized: {wfo['oos_annualized_ret']*100:.2f}%")
        print(f"WFO Ratio (Robustness):  {wfo['wfo_ratio']:.4f}")
        if wfo['wfo_ratio'] < 0.5:
            print("⚠️ WARNING: Severe Overfitting Detected!")
        elif wfo['wfo_ratio'] > 0.8:
            print("✅ Strategy is Robust.")

        print("\n--- 🎲 MONTE CARLO STRESS TEST (10k Sims / Vol-Clustering) ---")
        print(f"5% Value at Risk (1y): {mc['var_5']:.4f}")
        print(f"Avg Max Drawdown:      {mc['avg_max_dd']:.4f}")
        print(f"95% Max Drawdown:     {mc['p5_max_dd']:.4f}")
        print("="*60)

if __name__ == '__main__':
    suite = ValidationSuite()
    suite.run_full_audit()
