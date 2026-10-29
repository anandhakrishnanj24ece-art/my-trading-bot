import yfinance as yf
import pandas as pd
import numpy as np
from neural_flow.alpha.model import LiquidityAlpha
from neural_flow.execution.model import StealthExecution
from neural_flow.risk.model import VolatilityTarget, KellySizer, DrawdownGuard

class StrategyRunner:
    """
    Professional-grade engine to run a Neural-Flow backtest.
    Integrates Vol-Targeting, Fractional Kelly, and Drawdown Guard.
    """
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
            # 1. Alpha Generation
            asset_signals = []
            for ticker in tickers:
                asset_data = pd.DataFrame({
                    'Close': close_prices[ticker],
                    'Volume': volumes[ticker]
                })
                sig_df = self.alpha_engine.generate_signal(asset_data)
                sig = sig_df['signal'].iloc[i]
                asset_signals.append(sig)

            signals = np.array(asset_signals)
            if np.abs(signals).sum() > 0:
                weights = np.sign(signals) * (np.abs(signals) / np.abs(signals).sum())
            else:
                weights = np.zeros(len(tickers))

            # 2. Risk Layer 1: Volatility Targeting
            # Calculate leverage based on target vol / realized vol
            vol_lev = self.vol_target.get_leverage(returns.iloc[:i+1].mean(axis=1))

            # 3. Risk Layer 2: Fractional Kelly
            # Adjust leverage based on historical edge
            kelly_lev = self.kelly_sizer.calculate_f(pd.Series(portfolio_value).pct_change().fillna(0))

            # 4. Risk Layer 3: Drawdown Guard
            # Scale leverage based on distance from peak equity
            dd_mult = self.dd_guard.get_multiplier(portfolio_value[-1])

            # Combine: Use the most conservative of the three
            final_lev = min(vol_lev, kelly_lev) * dd_mult

            # Final Weights
            final_weights = weights * final_lev

            # 5. PnL Calculation
            period_return = (returns.iloc[i] @ final_weights)

            # 6. Execution Costs
            turnover = np.sum(np.abs(final_weights - current_weights))
            costs = self.execution_engine.calculate_cost(turnover)

            portfolio_value.append(portfolio_value[-1] * (1 + period_return - costs))
            current_weights = final_weights

        return pd.Series(portfolio_value).pct_change().dropna()

def run_neural_flow_backtest():
    """
    Maintain legacy compatibility.
    """
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOT-USD', 'POL-USD', 'LINK-USD', 'AVAX-USD']
    data_raw = yf.download(tickers, period='2y', interval='1d')
    close_prices = data_raw['Close'].ffill().dropna()
    volumes = data_raw['Volume'].ffill().dropna()

    runner = StrategyRunner()
    strategy_rets = runner.run(close_prices, volumes)

    portfolio_value = (1 + strategy_rets).cumprod()
    portfolio_value = np.insert(portfolio_value.values, 0, 1.0)

    print(f"--- NEURAL-FLOW STABILIZED RESULTS ---")
    print(f"Final Portfolio Value: {portfolio_value[-1]:.4f}")
    print(f"Total Return: {(portfolio_value[-1]-1)*100:.2f}%")
    print(f"Max Drawdown: {(1 - pd.Series(portfolio_value).cummax()/pd.Series(portfolio_value)).max():.4f}")

if __name__ == '__main__':
    run_neural_flow_backtest()
