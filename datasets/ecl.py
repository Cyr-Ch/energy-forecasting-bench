"""ECL dataset loader."""

from .base import BaseDataset
from .registry import register_dataset


@register_dataset("ecl")
class ECLDataset(BaseDataset):
    """ECL (Electricity Consuming Load) dataset."""
    
    def __init__(self, config):
        self.config = config
        self.data_path = config.get("data_path", "data/raw/ecl")
    
    def load_data(self):
        """Load ECL dataset."""
        # Implementation here
        pass

