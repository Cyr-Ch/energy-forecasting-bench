import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class EnergyDataset:
    def __init__(self, df: pd.DataFrame, freq: str):
        assert df.index.is_monotonic_increasing
        self.df = df
        self.freq = freq

    def split(self, train=0.7, val=0.1):
        n = len(self.df)
        n_train = int(n * train)
        n_val = int(n * val)
        return Split(
            train=self.df.iloc[:n_train],
            val=self.df.iloc[n_train:n_train+n_val],
            test=self.df.iloc[n_train+n_val:],
        )
