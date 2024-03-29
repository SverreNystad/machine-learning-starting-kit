""" This module contains functions to load the data, directly from the source. """

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

import src.config as config


class DataLoader(ABC):

    @abstractmethod
    def load_raw_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load the data and return the features and target variable."""
        pass

    @abstractmethod
    def load_raw_test_data(self) -> pd.DataFrame:
        """Load the test data."""
        pass


class LocalDataLoader(DataLoader):
    def load_raw_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        data = pd.read_csv(config.TRAINING_DATA_FILE)
        target_column = config.TARGET_FEATURE
        x = data.drop(target_column, axis=1)
        y = data[target_column]
        # Make sure the x and y are DataFrames
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        return x, y

    def load_raw_test_data(self) -> pd.DataFrame:
        data = pd.read_csv(config.TEST_DATA_FILE)
        return data


class MockDataLoader(DataLoader):

    def __init__(self, n_train: int, n_test: int) -> None:
        self.n_train = n_train
        self.n_test = n_test

    def load_raw_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        X_train, y_train, _, _ = self.get_data(self.n_train, self.n_test)
        return pd.DataFrame(X_train), pd.DataFrame(y_train)

    def load_raw_test_data(self) -> pd.DataFrame:
        _, _, X_test, _ = self.get_data(self.n_train, self.n_test)
        return X_test

    def get_data(
        self, n_train: int, n_test: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generating training and test data for
        """
        X_train = np.random.rand(n_train, 2) * 2 - 1
        y_train = self._noisy_func(X_train)
        X_test = np.random.rand(n_test, 2) * 2 - 1
        y_test = self._noisy_func(X_test)

        return X_train, y_train, X_test, y_test

    def _noisy_func(self, X: np.ndarray, epsilon: float = 0.075) -> np.ndarray:
        """
        Add Gaussian noise to the data generating function.
        """
        return self._func(X) + np.random.randn(len(X)) * epsilon

    def _func(self, X: np.ndarray) -> np.ndarray:
        """
        The data generating function.
        """
        return 0.3 * X[:, 0] + 0.6 * X[:, 1] ** 2


def create_data_loader(loader: str = config.LOADER_NAME) -> DataLoader:
    match loader.lower():
        case "local_data_loader":
            return LocalDataLoader()
        # TODO: If your dataset is to large to fit on the machine, add a custom data loader here that queries the data from a database.
        case _:
            raise ValueError(f"The DataLoader: '{loader}' not found.")
