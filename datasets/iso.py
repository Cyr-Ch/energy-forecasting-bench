"""ISO PJM dataset loader."""

from .base import BaseDataset
from .registry import register_dataset


@register_dataset("iso_pjm")
class ISOPJMDataset(BaseDataset):
    """ISO PJM dataset."""
    
    def __init__(self, config):
        self.config = config
        self.data_path = config.get("data_path", "data/raw/iso_pjm")
    
    def load_data(self):
        """Load ISO PJM dataset."""
        # Implementation here
        pass

