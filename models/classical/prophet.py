"""Prophet model implementation."""

from prophet import Prophet


class ProphetModel:
    """Facebook Prophet model."""
    
    def __init__(self, config):
        self.config = config
        # Implementation here
    
    def fit(self, data):
        """Fit Prophet model."""
        pass
    
    def predict(self, steps):
        """Make predictions."""
        pass

