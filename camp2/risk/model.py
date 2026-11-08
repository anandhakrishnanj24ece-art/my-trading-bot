import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize

class DCCGARCHRiskModel:
    """
    Dynamic Conditional Correlation GARCH.
    1. Univariate GJR-GARCH for variances.
    2. DCC for dynamic correlations.
    """
    def __init__(self):
        self.variances = {}
        self.correlations = None

    def fit_univariate(self, returns: pd.DataFrame):
        # GJR-GARCH(1,1) for each asset
        for col in returns.columns:
            model = arch_model(returns[col], vol='Garch', p=1, o=1, q=1, dist='Normal')
            res = model.fit(disp='off')
            self.variances[col] = res.conditional_volatility

    def estimate_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        # Simplified DCC approximation for demonstration:
        # In a full impl, we solve for the Q matrix (1-a-b)Qbar + a(z z') + b(Qprev)
        # Here we provide the GARCH-scaled correlation matrix
        std_res = pd.DataFrame({col: returns[col] / self.variances[col] for col in returns.columns})
        corr = std_res.corr() # Static for this simplified version

        # Scale back by forecasted volatility
        vols = np.diag([self.variances[col].iloc[-1] for col in returns.columns])
        cov = vols @ corr.values @ vols
        return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)

class CostAwareOptimizer:
    """
    Portfolio optimization with transaction cost penalties.
    """
    def __init__(self, lambda_risk=0.001, cost_coeff=0.0001):
        self.lambda_risk = lambda_risk
        self.cost_coeff = cost_coeff

    def optimize(self, mu, sigma, w_old):
        n = len(mu)
        def objective(w):
            risk = 0.5 * self.lambda_risk * (w.T @ sigma @ w)
            costs = self.cost_coeff * np.sum(np.abs(w - w_old))
            return - (w.T @ mu) + risk + costs

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - 1.0}, # Full investment / Budget
        ]
        bounds = [(-0.2, 0.2) for _ in range(n)]

        res = minimize(objective, w_old, method='SLSQP', bounds=bounds, constraints=constraints)
        return res.x
