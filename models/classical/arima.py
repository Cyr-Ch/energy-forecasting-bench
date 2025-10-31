"""ARIMA model implementation."""

from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel:
    """ARIMA (AutoRegressive Integrated Moving Average) model."""
    
    def __init__(self, config):
        self.config = config
        # Implementation here
    
    def fit(self, data):
        """Fit ARIMA model."""
        pass
    
    def predict(self, steps):
        """Make predictions."""
        pass

