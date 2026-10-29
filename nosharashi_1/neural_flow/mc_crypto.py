import yfinance as yf
import pandas as pd
import numpy as np
from neural_flow.alpha.model import LiquidityAlpha
from neural_flow.execution.model import StealthExecution
from neural_flow.risk.model import VolatilityTarget, KellySizer, DrawdownGuard
from neural_flow.validation.simulation import circular_block_bootstrap, calculate_simulation_metrics

class StrategyRunner:
    def __init__(self, target_vol=0.15, kelly_fraction=0.25, execution_engine=None):
        self.alpha_engine = LiquidityAlpha()
        self.execution_engine = execution_engine if execution_engine else StealthExecution()
        self.vol_target = VolatilityTarget(target_vol=target_vol)
        self.kelly_sizer = KellySizer(fraction=kelly_fraction)
        self.dd_guard = DrawdownGuard()

    def run(self, close_prices, volumes):
        tickers = close_prices.columns.tolist()
        returns = close_prices.pct_change().dropna()

        common_index = returns.index.intersection(close_prices.index).intersection(volumes.index)
        returns = returns.loc[common_index]
        close_prices = close_prices.loc[common_index]
        volumes = volumes.loc[common_index]

        portfolio_value = [1.0]
        current_weights = np.zeros(len(tickers))

        for i in range(len(returns)):
            asset_signals = []
            for ticker in tickers:
                asset_data = pd.DataFrame({
                    'Close': close_prices[ticker],
                    'Volume': volumes[ticker]
                })
                sig_df = self.alpha_engine.generate_signal(asset_data)
                sig = sig_df['signal'].iloc[i] if not sig_df.empty else 0
                asset_signals.append(sig)

            signals = np.array(asset_signals)
            if np.abs(signals).sum() > 0:
                weights = np.sign(signals) * (np.abs(signals) / np.abs(signals).sum())
            else:
                weights = np.zeros(len(tickers))

            vol_lev = self.vol_target.get_leverage(returns.iloc[:i+1].mean(axis=1))
            kelly_lev = self.kelly_sizer.calculate_f(pd.Series(portfolio_value).pct_change().fillna(0))
            dd_mult = self.dd_guard.get_multiplier(portfolio_value[-1])

            final_lev = min(vol_lev, kelly_lev) * dd_mult
            final_weights = weights * final_lev
            period_return = (returns.iloc[i] @ final_weights)

            turnover = np.sum(np.abs(final_weights - current_weights))
            costs = self.execution_engine.calculate_cost(turnover)

            portfolio_value.append(portfolio_value[-1] * (1 + period_return - costs))
            current_weights = final_weights

        return pd.Series(portfolio_value).pct_change().dropna()

def run_crypto_analysis(n_sims=10000):
    tickers = ['BTC-USD', 'ETH-USD']
    print(f"Downloading data for Crypto Bundle: {tickers}...")

    close_list = {}
    vol_list = {}

    for t in tickers:
        try:
            df = yf.download(t, period='2y', interval='1d', progress=False)
            if not df.empty:
                # yfinance v0.2+ returns a DataFrame where columns might be MultiIndex
                # We flatten them to ensure we get a Series
                if isinstance(df.columns, pd.MultiIndex):
                    close_list[t] = df['Close'][t] if t in df['Close'].columns else df['Close'].iloc[:,0]
                    vol_list[t] = df['Volume'][t] if t in df['Volume'].columns else df['Volume'].iloc[:,0]
                else:
                    close_list[t] = df['Close']
                    vol_list[t] = df['Volume']
        except Exception as e:
            print(f"Failed to download {t}: {e}")

    if not close_list:
        print("Error: No data downloaded.")
        return

    close_prices = pd.DataFrame(close_list).ffill().dropna()
    volumes = pd.DataFrame(vol_list).ffill().dropna()

    runner = StrategyRunner()
    strategy_rets = runner.run(close_prices, volumes)

    if strategy_rets.empty or np.isnan(strategy_rets).all():
        print("Error: Strategy generated no returns.")
        return

    print(f"Running Monte Carlo simulation ({n_sims} iterations)...")
    sim_rets = circular_block_bootstrap(strategy_rets, n_sims=n_sims)
    metrics = calculate_simulation_metrics(sim_rets)

    print(f"\n--- CRYPTO MC RESULTS ---")
    print(f"Expected Final Value (Avg): {np.mean(np.prod(1 + sim_rets, axis=1)):.4f}")
    print(f"5% VaR (Final Value): {metrics['var_5']:.4f}")
    print(f"Avg Max Drawdown: {metrics['avg_max_dd']:.4f}")
    print(f"P5 Max Drawdown: {metrics['p5_max_dd']:.4f}")

if __name__ == '__main__':
    run_crypto_analysis()
