"""Utilities for data scaling."""

import numpy as np


class StandardScaler:
    def fit(self, x):
        self.mean = np.nanmean(x, axis=0)
        self.std = np.nanstd(x, axis=0) + 1e-8
        return self
    
    def transform(self, x):
        return (x - self.mean) / self.std
    
    def inverse_transform(self, x):
        return x * self.std + self.mean
