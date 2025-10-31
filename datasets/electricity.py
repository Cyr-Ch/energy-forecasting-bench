"""Electricity dataset loader."""

from .base import BaseDataset
from .registry import register_dataset


@register_dataset("electricity")
class ElectricityDataset(BaseDataset):
    """Electricity consumption dataset."""
    
    def __init__(self, config):
        self.config = config
        self.data_path = config.get("data_path", "data/raw/electricity")
    
    def load_data(self):
        """Load electricity dataset."""
        # Implementation here
        pass

