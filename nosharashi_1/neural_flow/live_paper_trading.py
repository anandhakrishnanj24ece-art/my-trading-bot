import yfinance as yf
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.covariance import LedoitWolf
from neural_flow.alpha.model import LiquidityAlpha
from neural_flow.execution.model import StealthExecution

# Re-import Pro v3 Components for the Live System
class AdaptiveWeightedV3:
    def __init__(self, lambda_base=0.005, tau_base=0.01, critical_threshold=3.0):
        self.lambda_base = lambda_base
        self.tau_base = tau_base
        self.critical_threshold = critical_threshold
        self.prev_weights = None

    def _sigmoid_scaling(self, p):
        alpha = 10.0
        return 0.5 * (1 / (1 + np.exp(-alpha * (p - 0.5))))

    def optimize(self, signals, confidences, market_vol=0.15):
        snr = np.mean(np.abs(signals)) / (np.std(signals) + 1e-6)
        lambda_adj = self.lambda_base / (1.0 + snr)
        abs_sigs = np.abs(signals)
        if abs_sigs.sum() == 0: return np.zeros(len(signals))
        weights = np.sign(signals) * (abs_sigs / abs_sigs.sum())
        weights = np.where(np.abs(weights) > lambda_adj, weights, 0)
        if np.abs(weights).sum() > 0: weights = weights / np.abs(weights).sum()
        else: weights = np.zeros(len(signals))
        final_weights = []
        for w, conf in zip(weights, confidences):
            p = np.clip(conf / 3.0, 0, 1)
            scaling = self._sigmoid_scaling(p)
            cap = 0.05 + scaling * (0.50 - 0.05)
            final_weights.append(np.clip(w, -cap, cap))
        weights = np.array(final_weights)
        total_exp = np.sum(np.abs(weights))
        if total_exp > 1.0: weights = weights / total_exp
        if self.prev_weights is not None:
            new_weights = np.copy(self.prev_weights)
            vol_scale = 1.0 / (1.0 + market_vol * 10)
            tau_adj = self.tau_base * vol_scale
            for i in range(len(weights)):
                if abs(signals[i]) > self.critical_threshold:
                    new_weights[i] = weights[i]
                    continue
                diff = weights[i] - self.prev_weights[i]
                if diff < 0 and abs(diff) > (0.5 * tau_adj):
                    new_weights[i] = weights[i]
                elif diff > 0 and abs(diff) > (1.5 * tau_adj):
                    new_weights[i] = weights[i]
                else:
                    new_weights[i] = self.prev_weights[i]
            weights = new_weights
        self.prev_weights = weights
        return weights

class ProRiskManagerV3:
    def __init__(self, target_vol=0.15, base_leverage=4.0):
        self.target_vol = target_vol
        self.base_leverage = base_leverage

    def get_portfolio_leverage(self, weights, returns_df):
        if len(returns_df) < 20: return 1.0
        recent_rets = returns_df.tail(20)
        try:
            lw = LedoitWolf().fit(recent_rets)
            cov_matrix = lw.covariance_ * 252
        except:
            cov_matrix = recent_rets.cov() * 252
        port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        port_vol = np.sqrt(max(0, port_var))
        if port_vol == 0: return self.base_leverage
        return min(self.target_vol / port_vol, self.base_leverage)

class PaperTrader:
    def __init__(self, initial_balance=10000):
        self.balance = initial_balance
        self.portfolio_value = initial_balance
        self.current_weights = None
        self.tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD', 'AVAX-USD']
        self.alpha_engine = LiquidityAlpha()
        self.execution_engine = StealthExecution()
        self.optimizer = AdaptiveWeightedV3()
        self.risk_manager = ProRiskManagerV3()
        self.log_file = "paper_trade_log.csv"

    def update_portfolio(self):
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Updating Paper Portfolio...")

        # 1. Fetch Data
        data_raw = yf.download(self.tickers, period='1y', interval='1d')
        if data_raw.empty:
            print("Error: No data fetched.")
            return

        close_prices = data_raw['Close'].ffill().dropna()
        volumes = data_raw['Volume'].ffill().dropna()
        tickers_list = close_prices.columns.tolist()
        returns = close_prices.pct_change().dropna()

        # Align Data
        common_index = returns.index.intersection(close_prices.index).intersection(volumes.index)
        returns = returns.loc[common_index]
        close_prices = close_prices.loc[common_index]
        volumes = volumes.loc[common_index]

        # 2. Generate signals and optimize
        asset_signals, asset_confidences = [], []
        for ticker in tickers_list:
            asset_df = pd.DataFrame({'Close': close_prices[ticker], 'Volume': volumes[ticker]})
            res_df = self.alpha_engine.generate_signal(asset_df)
            if not res_df.empty:
                res = res_df.iloc[-1]
                asset_signals.append(res['signal'])
                asset_confidences.append(res['confidence'])
            else:
                asset_signals.append(0.0)
                asset_confidences.append(0.0)

        # Calculate Market Vol for the optimizer
        mkt_vol = returns.iloc[:, 0].std() * np.sqrt(252) # BTC as proxy

        weights = self.optimizer.optimize(np.array(asset_signals), np.array(asset_confidences), market_vol=mkt_vol)
        lev = self.risk_manager.get_portfolio_leverage(weights, returns)
        final_weights = weights * lev

        # 3. Calculate Paper PnL
        # Use today's actual return to update the balance
        current_returns = returns.iloc[-1].values
        day_return = np.sum(final_weights * current_returns)

        turnover = 0
        if self.current_weights is not None:
            turnover = np.sum(np.abs(final_weights - self.current_weights))

        costs = self.execution_engine.calculate_cost(turnover)
        net_return = day_return - costs

        self.portfolio_value *= (1 + net_return)
        self.current_weights = final_weights

        # 4. Log the trade
        log_entry = pd.DataFrame([{
            'timestamp': datetime.now(),
            'portfolio_value': self.portfolio_value,
            'net_return': net_return,
            'leverage': lev,
            'turnover': turnover,
            'costs': costs
        }])

        log_entry.to_csv(self.log_file, mode='a', header=not pd.io.common.file_exists(self.log_file), index=False)

        print(f"Portfolio Value: ${self.portfolio_value:.2f}")
        print(f"Daily Net Return: {net_return*100:.4f}%")
        print(f"Current Leverage: {lev:.4f}")
        print(f"Turnover: {turnover*100:.2f}%")

if __name__ == '__main__':
    trader = PaperTrader()
    # Run once for initial check, then you can wrap this in a while loop/cron
    trader.update_portfolio()
