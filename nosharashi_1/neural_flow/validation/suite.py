import numpy as np
import pandas as pd
import yfinance as yf
from neural_flow.backtest import StrategyRunner
from neural_flow.execution.model import StealthExecution
from neural_flow.validation.simulation import circular_block_bootstrap
from neural_flow.validation import metrics

class RobustnessSuite:
    """
    Professional-grade validation suite for the Neural-Flow strategy.
    Orchestrates Monte Carlo simulations and Friction Stress-Testing.
    """
    def __init__(self, tickers=None, initial_capital=10000):
        self.tickers = tickers or ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD']
        self.initial_capital = initial_capital

    def fetch_data(self, period='2y'):
        print(f"Downloading data for {len(self.tickers)} tickers...")
        data_raw = yf.download(self.tickers, period=period, interval='1d')
        close_prices = data_raw['Close'].ffill().dropna()
        volumes = data_raw['Volume'].ffill().dropna()

        # Filter tickers that actually have data
        valid_tickers = close_prices.columns.tolist()
        return close_prices, volumes, valid_tickers

    def run_monte_carlo(self, close_prices, volumes, n_sims=1000, block_size=5):
        print(f"\n🚀 Running Monte Carlo Simulation ({n_sims} paths)...")

        # Baseline run to get the strategy's actual return series
        baseline_runner = StrategyRunner()
        baseline_rets = baseline_runner.run(close_prices, volumes)

        if baseline_rets is None or len(baseline_rets) == 0:
            print("❌ Baseline returns are empty. Monte Carlo cannot proceed.")
            return np.array([])

        # Bootstrap the strategy's realized return series
        sim_rets = circular_block_bootstrap(baseline_rets, block_size=block_size, n_sims=n_sims)

        if sim_rets is None or sim_rets.size == 0:
            print("❌ Simulation returns are empty.")
            return np.array([])

        # Convert simulated returns to equity curves
        sim_equity_curves = []
        for path in sim_rets:
            # Add 1 to returns and cumulative product to get the equity curve
            sim_equity_curves.append((1 + path).cumprod())

        return np.array(sim_equity_curves)

    def run_friction_stress_test(self, close_prices, volumes):
        print(f"\n⚡ Running Friction Stress-Test...")

        fee_grid = [0.0001, 0.0005, 0.001, 0.005]
        slip_grid = [0.0001, 0.0002, 0.0005, 0.001]

        results = []
        for fee in fee_grid:
            for slip in slip_grid:
                # Custom execution engine for this friction level
                exec_engine = StealthExecution(fee=fee, slippage=slip)
                runner = StrategyRunner(execution_engine=exec_engine)
                rets = runner.run(close_prices, volumes)

                sharpe = (rets.mean() * 252) / (rets.std() * np.sqrt(252)) if rets.std() != 0 else 0
                results.append({
                    "fee": fee,
                    "slip": slip,
                    "sharpe": sharpe,
                    "return": rets.sum()
                })

        return pd.DataFrame(results)

    def generate_report(self, baseline_rets, mc_curves, friction_df):
        print(f"\n" + "="*50)
        print(f"🛡️  NEURAL-FLOW ROBUSTNESS REPORT")
        print("="*50)

        # 1. Executive Summary
        final_values = mc_curves[:, -1]
        mean_return = (np.mean(final_values) - 1) * 100
        profitable_paths = (final_values > 1.0).mean() * 100

        print(f"\n[ EXECUTIVE SUMMARY ]")
        print(f"Mean MC Return: {mean_return:.2f}%")
        print(f"Confidence Score: {profitable_paths:.2f}% of paths profitable")

        # 2. Risk Distribution
        var_95 = np.percentile(final_values, 5)
        cvar_95 = np.mean(final_values[final_values <= var_95])

        print(f"\n[ RISK DISTRIBUTION ]")
        print(f"VaR (95%): {(var_95 - 1)*100:.2f}%")
        print(f"CVaR (95%): {(cvar_95 - 1)*100:.2f}%")

        # 3. Quant Metrics
        print(f"\n[ QUANT METRICS ]")
        print(f"Sharpe Ratio:   { (baseline_rets.mean()*252)/(baseline_rets.std()*np.sqrt(252)):.4f}")
        print(f"Sortino Ratio:  {metrics.calculate_sortino_ratio(baseline_rets):.4f}")
        print(f"Calmar Ratio:   {metrics.calculate_calmar_ratio(baseline_rets):.4f}")
        print(f"Ulcer Index:    {metrics.calculate_ulcer_index(baseline_rets):.4f}")
        print(f"Profit Factor:  {metrics.calculate_profit_factor(baseline_rets):.4f}")

        # 4. Friction Breaking Point
        breaking_point = friction_df[friction_df['sharpe'] > 0.5].iloc[-1] if not friction_df[friction_df['sharpe'] > 0.5].empty else None

        print(f"\n[ FRICTION ANALYSIS ]")
        if breaking_point is not None:
            print(f"Breaking Point: Fee {breaking_point['fee']:.5f}, Slip {breaking_point['slip']:.5f}")
        else:
            print("Breaking Point: Strategy unprofitable at all tested friction levels.")

        print("="*50)

    def run_full_suite(self):
        close_prices, volumes, valid_tickers = self.fetch_data()

        # 1. Baseline
        baseline_runner = StrategyRunner()
        baseline_rets = baseline_runner.run(close_prices, volumes)

        # 2. Monte Carlo
        mc_curves = self.run_monte_carlo(close_prices, volumes)

        # 3. Friction
        friction_df = self.run_friction_stress_test(close_prices, volumes)

        # 4. Report
        self.generate_report(baseline_rets, mc_curves, friction_df)

if __name__ == '__main__':
    suite = RobustnessSuite()
    suite.run_full_suite()
