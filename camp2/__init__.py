import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class AlphaSleeve(ABC):
    """Abstract base class for alpha sources."""
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> pd.Series:
        pass

class RiskModel(ABC):
    """Abstract base class for risk/covariance estimation."""
    @abstractmethod
    def estimate_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        pass

class MetaModel(ABC):
    """Abstract base class for the meta-labeling layer."""
    @abstractmethod
    def predict_probability(self, features: pd.DataFrame) -> pd.Series:
        pass

class ExecutionEngine(ABC):
    """Abstract base class for trade execution."""
    @abstractmethod
    def execute(self, target_weights: pd.Series, current_weights: pd.Series):
        pass
