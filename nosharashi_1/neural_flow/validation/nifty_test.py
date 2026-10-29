import numpy as np
import pandas as pd
import yfinance as yf
from neural_flow.backtest import StrategyRunner
from neural_flow.execution.model import StealthExecution
from neural_flow.validation.simulation import circular_block_bootstrap
from neural_flow.validation import metrics

class NiftyRobustnessSuite:
    """
    Specialized validation suite for the Nifty 50 index.
    """
    def __init__(self, ticker='^NSEI', initial_capital=10000):
        self.ticker = ticker
        self.initial_capital = initial_capital

    def fetch_data(self, period='2y'):
        print(f"Downloading data for {self.ticker}...")
        df = yf.download(self.ticker, period=period, interval='1d')

        if df.empty:
            print(f"❌ No data found for {self.ticker}")
            return pd.DataFrame(), pd.DataFrame(), []

        if isinstance(df.columns, pd.MultiIndex):
            close = df['Close'][self.ticker]
            volume = df['Volume'][self.ticker]
        else:
            close = df['Close']
            volume = df['Volume']

        close = close.ffill().dropna()
        volume = volume.ffill().dropna()

        close_df = pd.DataFrame({self.ticker: close})
        volume_df = pd.DataFrame({self.ticker: volume})

        return close_df, volume_df, [self.ticker]

    def run_monte_carlo(self, close_prices, volumes, n_sims=1000, block_size=5):
        print(f"\n🚀 Running Monte Carlo Simulation on NIFTY ({n_sims} paths)...")
        baseline_runner = StrategyRunner()
        baseline_rets = baseline_runner.run(close_prices, volumes)

        if baseline_rets is None or len(baseline_rets) == 0:
            print("❌ Baseline returns are empty. Monte Carlo cannot proceed.")
            return np.array([])

        sim_rets = circular_block_bootstrap(baseline_rets, block_size=block_size, n_sims=n_sims)

        if sim_rets is None or sim_rets.size == 0:
            print("❌ Simulation returns are empty.")
            return np.array([])

        sim_equity_curves = []
        for path in sim_rets:
            sim_equity_curves.append((1 + path).cumprod())

        return np.array(sim_equity_curves)

    def run_friction_stress_test(self, close_prices, volumes):
        print(f"\n⚡ Running Friction Stress-Test on NIFTY...")
        fee_grid = [0.0001, 0.0005, 0.001, 0.005]
        slip_grid = [0.0001, 0.0002, 0.0005, 0.001]
        results = []
        for fee in fee_grid:
            for slip in slip_grid:
                exec_engine = StealthExecution(fee=fee, slippage=slip)
                runner = StrategyRunner(execution_engine=exec_engine)
                rets = runner.run(close_prices, volumes)
                sharpe = (rets.mean() * 252) / (rets.std() * np.sqrt(252)) if (rets is not None and rets.std() != 0) else 0
                results.append({"fee": fee, "slip": slip, "sharpe": sharpe, "return": rets.sum() if rets is not None else 0})
        return pd.DataFrame(results)

    def generate_report(self, baseline_rets, mc_curves, friction_df):
        print(f"\n" + "="*50)
        print(f"🛡️  NIFTY-50 ROBUSTNESS REPORT")
        print("="*50)

        if mc_curves is None or mc_curves.size == 0:
            print("\n❌ No Monte Carlo data available to generate report.")
            return

        final_values = mc_curves[:, -1] if mc_curves.ndim > 1 else mc_curves
        mean_return = (np.mean(final_values) - 1) * 100
        profitable_paths = (final_values > 1.0).mean() * 100

        print(f"\n[ EXECUTIVE SUMMARY ]")
        print(f"Mean MC Return: {mean_return:.2f}%")
        print(f"Confidence Score: {profitable_paths:.2f}% of paths profitable")

        var_95 = np.percentile(final_values, 5)
        cvar_95 = np.mean(final_values[final_values <= var_95]) if len(final_values[final_values <= var_95]) > 0 else var_95

        print(f"\n[ RISK DISTRIBUTION ]")
        print(f"VaR (95%): {(var_95 - 1)*100:.2f}%")
        print(f"CVaR (95%): {(cvar_95 - 1)*100:.2f}%")

        print(f"\n[ QUANT METRICS ]")
        std = baseline_rets.std()
        sharpe = (baseline_rets.mean() * 252) / (std * np.sqrt(252)) if std != 0 else 0
        print(f"Sharpe Ratio:   {sharpe:.4f}")
        print(f"Sortino Ratio:  {metrics.calculate_sortino_ratio(baseline_rets):.4f}")
        print(f"Calmar Ratio:   {metrics.calculate_calmar_ratio(baseline_rets):.4f}")
        print(f"Ulcer Index:    {metrics.calculate_ulcer_index(baseline_rets):.4f}")
        print(f"Profit Factor:  {metrics.calculate_profit_factor(baseline_rets):.4f}")

        breaking_point = friction_df[friction_df['sharpe'] > 0.5].iloc[-1] if not friction_df[friction_df['sharpe'] > 0.5].empty else None

        print(f"\n[ FRICTION ANALYSIS ]")
        if breaking_point is not None:
            print(f"Breaking Point: Fee {breaking_point['fee']:.5f}, Slip {breaking_point['slip']:.5f}")
        else:
            print("Breaking Point: Strategy unprofitable at all tested friction levels.")
        print("="*50)

    def run_full_suite(self):
        close_prices, volumes, valid_tickers = self.fetch_data()
        if close_prices.empty:
            print("❌ No data fetched. Aborting.")
            return

        baseline_runner = StrategyRunner()
        baseline_rets = baseline_runner.run(close_prices, volumes)

        if baseline_rets is None or len(baseline_rets) == 0:
            print("❌ No returns generated. Aborting.")
            return

        mc_curves = self.run_monte_carlo(close_prices, volumes)
        friction_df = self.run_friction_stress_test(close_prices, volumes)
        self.generate_report(baseline_rets, mc_curves, friction_df)

if __name__ == '__main__':
    suite = NiftyRobustnessSuite()
    suite.run_full_suite()
