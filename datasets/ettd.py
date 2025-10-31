"""ETTh and ETTm dataset loaders."""

from .base import BaseDataset
from .registry import register_dataset


@register_dataset("etth")
class ETThDataset(BaseDataset):
    """ETTh (Electricity Transformer Temperature Hourly) dataset."""
    
    def __init__(self, config):
        self.config = config
        self.data_path = config.get("data_path", "data/raw/etth")
    
    def load_data(self):
        """Load ETTh dataset."""
        # Implementation here
        pass


@register_dataset("ettm")
class ETTmDataset(BaseDataset):
    """ETTm (Electricity Transformer Temperature Minute-level) dataset."""
    
    def __init__(self, config):
        self.config = config
        self.data_path = config.get("data_path", "data/raw/ettm")
    
    def load_data(self):
        """Load ETTm dataset."""
        # Implementation here
        pass

