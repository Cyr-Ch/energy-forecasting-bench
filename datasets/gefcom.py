"""GEFCOM 2014 Solar dataset loader."""

from .base import BaseDataset
from .registry import register_dataset


@register_dataset("gefcom2014_solar")
class GEFCOMDataset(BaseDataset):
    """GEFCOM 2014 Solar dataset."""
    
    def __init__(self, config):
        self.config = config
        self.data_path = config.get("data_path", "data/raw/gefcom2014_solar")
    
    def load_data(self):
        """Load GEFCOM dataset."""
        # Implementation here
        pass

