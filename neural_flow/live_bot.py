import ccxt
import pandas as pd
import numpy as np
import time
import json
import os
import sys
from datetime import datetime
from neural_flow.alpha.model import LiquidityAlpha
from neural_flow.execution.model import StealthExecution
from neural_flow.risk.model import VolatilityTarget, StopLoss

class NeuralFlowLive:
    def __init__(self, tickers, initial_balance=10000, account_file='paper_account.json'):
        self.exchange = ccxt.binance()
        self.tickers = tickers
        self.balance = initial_balance
        self.portfolio = {ticker: 0.0 for ticker in tickers}
        self.account_file = account_file
        self.load_account()

        self.alpha_engine = LiquidityAlpha()
        self.execution_engine = StealthExecution()
        self.risk_engine = VolatilityTarget(max_cap=4.0)
        self.stop_loss_engine = StopLoss(stop_pct=0.10)

        print(f"🚀 NEURAL-FLOW LIVE STARTUP")
        print(f"Initial Balance: ${self.balance:.2f}")
        print(f"Monitoring: {tickers}")
        print("-" * 30)

    def load_account(self):
        if os.path.exists(self.account_file):
            with open(self.account_file, 'r') as f:
                data = json.load(f)
                self.balance = data.get('balance', 10000)
                self.portfolio = data.get('portfolio', {ticker: 0.0 for ticker in self.tickers})
                if self.balance < 0:
                    self.balance = 10000
                    self.portfolio = {ticker: 0.0 for ticker in self.tickers}

    def save_account(self):
        with open(self.account_file, 'w') as f:
            json.dump({'balance': self.balance, 'portfolio': self.portfolio}, f)
        print("💾 Account state saved successfully.")

    def fetch_data(self, ticker):
        try:
            ohlcv = self.exchange.fetch_ohlcv(ticker, timeframe='1h', limit=100)
            if not ohlcv:
                return pd.DataFrame()
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            return df
        except Exception as e:
            print(f"❌ API Error fetching {ticker}: {e}")
            return pd.DataFrame()

    def run(self):
        last_trade_date = None
        try:
            while True:
                now = datetime.now()
                today = now.date()
                print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] Scanning Market...")

                if last_trade_date == today:
                    current_prices = {}
                    for ticker in self.tickers:
                        try:
                            df = self.fetch_data(ticker)
                            current_prices[ticker] = df['Close'].iloc[-1]
                        except: current_prices[ticker] = 0
                    
                    unrealized_pnl = sum(self.portfolio[t] * current_prices.get(t, 0) for t in self.tickers)
                    total_equity = self.balance + unrealized_pnl
                    print(f"Holding positions. Current Equity: ${total_equity:.2f} (Next trade tomorrow)")
                    time.sleep(3600)
                    continue

                current_signals, current_confidences, current_prices = {}, {}, {}
                for ticker in self.tickers:
                    try:
                        df = self.fetch_data(ticker)
                        price = df['Close'].iloc[-1]
                        current_prices[ticker] = price
                        alpha_res = self.alpha_engine.generate_signal(df)
                        current_signals[ticker] = alpha_res['signal'].iloc[-1]
                        current_confidences[ticker] = alpha_res['confidence'].iloc[-1]
                        print(f" {ticker}: Price ${price:.4f} | Signal: {current_signals[ticker]:.4f} | Conf: {current_confidences[ticker]:.4f}")
                    except Exception as e:
                        print(f"❌ Error fetching {ticker}: {e}")
                        current_signals[ticker], current_confidences[ticker], current_prices[ticker] = 0.0, 0.0, 0.0

                # 1. Check Stop Losses first
                for ticker in self.tickers:
                    units = self.portfolio.get(ticker, 0.0)
                    if units == 0: continue

                    price = current_prices.get(ticker, 0)
                    if price == 0: continue

                    # Simplification: we assume entry price is roughly the average if we don't track it
                    # In a real bot, you'd save entry_price in paper_account.json
                    # For now, we use the 20-period average as a proxy for entry
                    df = self.fetch_data(ticker)
                    entry_proxy = df['Close'].rolling(20).mean().iloc[-1]

                    side = 'long' if units > 0 else 'short'
                    if self.stop_loss_engine.should_exit(entry_proxy, price, side):
                        print(f"⚠️ STOP LOSS TRIGGERED: {ticker} at ${price:.4f}")
                        # Zero out position
                        diff_units = -units
                        trade_value = diff_units * price
                        self.balance -= trade_value
                        self.portfolio[ticker] = 0.0
                        trades_made = True

                # 2. Normal signal processing
                signals_arr = np.array(list(current_signals.values()))
                conf_arr = np.array(list(current_confidences.values()))

                if np.abs(signals_arr).sum() > 0:
                    weights = np.sign(signals_arr) * (np.abs(signals_arr) / np.abs(signals_arr).sum())
                else:
                    weights = np.zeros(len(self.tickers))

                final_weights = []
                for i in range(len(weights)):
                    conf = conf_arr[i]
                    cap = 0.05 if conf < 1.0 else (0.20 if conf < 2.0 else 0.50)
                    final_weights.append(np.clip(weights[i], -cap, cap))

                final_weights = np.array(final_weights)
                total_exp = np.sum(np.abs(final_weights))
                if total_exp > 1.0: final_weights = final_weights / total_exp

                btc_df = self.fetch_data('BTC/USDT')
                if btc_df is None or btc_df.empty or 'Close' not in btc_df.columns:
                    print("❌ Could not fetch BTC data or 'Close' column missing for risk calculation. Skipping cycle.")
                    time.sleep(60)
                    continue
                btc_rets = btc_df['Close'].pct_change()
                lev = self.risk_engine.get_leverage(btc_rets)
                final_weights = final_weights * lev

                unrealized_pnl = sum(self.portfolio.get(t, 0.0) * current_prices.get(t, 0.0) for t in self.tickers)
                total_equity = self.balance + unrealized_pnl

                trades_made = False
                for i, ticker in enumerate(self.tickers):
                    target_weight = final_weights[i]
                    current_weight = (self.portfolio.get(ticker, 0.0) * current_prices.get(ticker, 0)) / total_equity if total_equity != 0 else 0
                    diff_weight = target_weight - current_weight
                    if abs(diff_weight) > 0.01:
                        turnover_dollars = abs(diff_weight * total_equity)
                        cost = self.execution_engine.calculate_cost(turnover_dollars / total_equity if total_equity != 0 else 0)
                        trade_units = (diff_weight * total_equity) / current_prices.get(ticker, 1)
                        trade_value = trade_units * current_prices.get(ticker, 1)
                        self.balance -= trade_value
                        self.balance -= cost
                        self.portfolio[ticker] = self.portfolio.get(ticker, 0.0) + trade_units
                        print(f" ⚡ TRADE: {ticker} | Weight: {target_weight:.2%} | Units: {trade_units:.4f} | Cost: ${cost:.4f}")
                        trades_made = True

                if trades_made:
                    last_trade_date = today
                
                self.save_account()
                total_eq_final = self.balance + sum(self.portfolio.get(t, 0.0) * current_prices.get(t, 0.0) for t in self.tickers)
                print(f"Current Balance: ${self.balance:.2f} | Total Equity: ${total_eq_final:.2f}")

                if total_eq_final < 0:
                    print("❌ MARGIN CALL: Portfolio negative. Stopping bot.")
                    break

                time.sleep(60)

        except KeyboardInterrupt:
            print("\n\n🛑 Ctrl+C detected! Shutting down gracefully...")
        except Exception as e:
            import traceback
            print(f"❌ Unexpected Error: {e}")
            traceback.print_exc()
        finally:
            self.save_account()
            print("✅ Final account state saved. You can now safely close the program.")
            sys.exit(0)

if __name__ == '__main__':
    TICKERS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT', 'DOT/USDT', 'POL/USDT', 'LINK/USDT', 'AVAX/USDT']
    # To run a different wallet, change 'paper_account.json' to something else (e.g., 'paper_account_1.json')
    bot = NeuralFlowLive(tickers=TICKERS, account_file='paper_account.json')
    bot.run()
