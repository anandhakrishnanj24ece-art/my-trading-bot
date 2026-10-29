import yfinance as yf
import pandas as pd
import numpy as np
from camp2.alpha.model import ConditionalFactorModel, CrossSectionalAlpha, MomentumAlpha, QualityAlpha, ConvexityAlpha, VolatilityScaler
from camp2.meta.model import TripleBarrierLabeler, MetaLabelingModel
from camp2.risk.model import DCCGARCHRiskModel, CostAwareOptimizer

def run_camp2_backtest():
    # 1. Setup Data
    tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD',
               'DOT-USD', 'LINK-USD', 'LTC-USD', 'TRX-USD', 'BCH-USD']
    data = yf.download(tickers, period='2y', interval='1d')['Close']
    # Drop columns with too many NaNs first, then ffill, then drop remaining
    data = data.dropna(axis=1, thresh=int(len(data)*0.8))
    data = data.ffill().dropna()
    returns = data.pct_change().dropna()

    # Update tickers list based on what actually survived
    tickers = data.columns.tolist()

    # Create synthetic factors for demonstration (In real usage, these are On-Chain/Derivs)
    # Factor 1: Momentum (20d), Factor 2: Volatility (20d), Factor 3: Relative Strength
    factors = pd.DataFrame(index=returns.index)
    for t in tickers:
        factors[f'{t}_mom'] = returns[t].rolling(20).mean()
        factors[f'{t}_vol'] = returns[t].rolling(20).std()
    factors = factors.fillna(0)

    # Split Train/Test
    split = int(len(returns) * 0.7)
    train_ret = returns.iloc[:split]
    test_ret = returns.iloc[split:]
    train_fact = factors.iloc[:split]
    test_fact = factors.iloc[split:]

    # 2. Model Fitting
    # Primary Model: Bayesian Conditional Factor Model
    cf_model = ConditionalFactorModel()
    cf_model.fit(train_ret.mean(axis=1), train_fact.mean(axis=1).to_frame('avg_fact'))

    # Pure Quant Alpha Stack
    cs_alpha = CrossSectionalAlpha(window=20)
    ts_alpha = MomentumAlpha(window=20)
    q_alpha = QualityAlpha(window=20)
    cx_alpha = ConvexityAlpha(window=60)

    # Generate signals for the whole period
    cs_signals = cs_alpha.generate_signal(returns)
    ts_signals = ts_alpha.generate_signal(returns)
    q_signals = q_alpha.generate_signal(returns)
    cx_signals = cx_alpha.generate_signal(returns)

    # Meta-Labeling: Triple Barrier
    labeler = TripleBarrierLabeler(pt=0.001, sl=0.30, t_barrier=120)
    primary_signal = (train_ret['BTC-USD'] > 0).astype(int).map({1:1, 0:-1})
    labels = labeler.label_data(data['BTC-USD'].iloc[:split], primary_signal)
    labels = labels.fillna(0).astype(int)

    meta_model = MetaLabelingModel()
    meta_features = train_fact.mean(axis=1).to_frame('avg_fact')
    meta_model.fit(meta_features, labels)

    meta_model = MetaLabelingModel()
    meta_features = train_fact.mean(axis=1).to_frame('avg_fact')
    meta_model.fit(meta_features, labels)

    # 3. Backtest Simulation
    portfolio_value = [1.0]
    current_weights = np.zeros(len(tickers))
    LEVERAGE = 3.0 # Apply institutional leverage to amplify returns

    # Costs
    FEE = 0.0005 # 5bps (Maker/Taker avg)
    SLIPPAGE = 0.0002 # 2bps

    risk_model = DCCGARCHRiskModel()
    risk_model.fit_univariate(train_ret) # Initial fit

    for i in range(len(test_ret)):
        # A. Signal Generation
        curr_x_val = test_fact.iloc[i].mean()
        curr_x = pd.DataFrame({'avg_fact': [curr_x_val]}, index=[test_fact.index[i]])

        mu_pred = cf_model.predict(curr_x).values[0]

        # Pure Quant Alpha Combination (Equal Weighted Diversification)
        # We combine Mean Reversion, Momentum, Quality, and Convexity
        combined_signal = (cs_signals.iloc[i] + ts_signals.iloc[i] + q_signals.iloc[i] + cx_signals.iloc[i]) / 4
        combined_mu = mu_pred + combined_signal.mean()

        # B. Meta-Sizing
        prob = meta_model.predict_prob(curr_x).values[0]
        size_scalar = max(0, 2*prob - 1)

        # C. Risk Optimization
        # mu: predicted return, sigma: dynamic covariance
        mu = np.full(len(tickers), combined_mu) * size_scalar * LEVERAGE
        sigma = risk_model.estimate_covariance(train_ret).values

        optimizer = CostAwareOptimizer()
        new_weights = optimizer.optimize(mu, sigma, current_weights)

        # D. PnL Calculation with Costs
        period_return = (test_ret.iloc[i] @ new_weights)
        turnover = np.sum(np.abs(new_weights - current_weights))
        costs = turnover * (FEE + SLIPPAGE)

        portfolio_value.append(portfolio_value[-1] * (1 + period_return - costs))
        current_weights = new_weights

    # 4. Monte Carlo Simulation (on strategy returns)
    strategy_rets = pd.Series(portfolio_value).pct_change().dropna()
    mc_results = []
    for _ in range(1000):
        sim_rets = np.random.choice(strategy_rets, size=len(strategy_rets), replace=True)
        mc_results.append(np.prod(1 + sim_rets))

    print(f"Final Portfolio Value: {portfolio_value[-1]:.4f}")
    print(f"Sharpe Ratio: { (strategy_rets.mean()/strategy_rets.std()) * np.sqrt(252):.4f}")
    print(f"Max Drawdown: { (1 - pd.Series(portfolio_value).cummax()/pd.Series(portfolio_value)).max():.4f}")
    print(f"MC 5% VaR (1y): {np.percentile(mc_results, 5):.4f}")

if __name__ == '__main__':
    run_camp2_backtest()
