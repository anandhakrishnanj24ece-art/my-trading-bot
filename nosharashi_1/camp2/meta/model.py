import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

class TripleBarrierLabeler:
    """
    Implementation of the Triple-Barrier Method for Meta-Labeling.
    """
    def __init__(self, pt: float, sl: float, t_barrier: int):
        self.pt = pt  # Profit Target
        self.sl = sl  # Stop Loss
        self.t_barrier = t_barrier # Time limit

    def label_data(self, prices: pd.Series, signal: pd.Series) -> pd.Series:
        labels = pd.Series(index=prices.index, dtype=int)

        for i in range(len(prices) - self.t_barrier):
            if signal.iloc[i] == 0:
                labels.iloc[i] = 0
                continue

            # Slicing the window for barrier check
            window = prices.iloc[i+1 : i+1 + self.t_barrier]
            ret = (window / prices.iloc[i]) - 1
            if signal.iloc[i] == 1: # Long
                ret = ret
            else: # Short
                ret = -ret

            # Check which barrier is hit first
            hit_pt = np.where(ret >= self.pt)[0]
            hit_sl = np.where(ret <= -self.sl)[0]

            first_pt = hit_pt[0] if len(hit_pt) > 0 else float('inf')
            first_sl = hit_sl[0] if len(hit_sl) > 0 else float('inf')

            if first_pt < first_sl and first_pt < self.t_barrier:
                labels.iloc[i] = 1
            elif first_sl < first_pt and first_sl < self.t_barrier:
                labels.iloc[i] = -1
            else:
                labels.iloc[i] = 0

        return labels

class MetaLabelingModel:
    """
    Secondary model to predict P(profitable | signal, features)
    """
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=100, max_depth=3)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Ensure X and y are aligned
        common_index = X.index.intersection(y.index)
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]

        mask = y_aligned != 0
        self.model.fit(X_aligned[mask], y_aligned[mask])
        return self

    def predict_prob(self, X: pd.DataFrame) -> pd.Series:
        # Returns probability of the '1' class (profitable)
        return pd.Series(self.model.predict_proba(X)[:, 1], index=X.index)
