""" This module contains functions to load the data, directly from the source. """

import pandas as pd
import src.config as config 
from abc import ABC, abstractmethod

class DataLoader(ABC):
    
    @abstractmethod
    def load_raw_data() -> tuple[pd.DataFrame, pd.Series]:
        pass

    @abstractmethod
    def load_raw_test_data() -> pd.DataFrame:
        pass



class LocalDataLoader(DataLoader):
    def load_raw_data() -> tuple[pd.DataFrame, pd.Series]:
        """Load the data and return the features and target variable."""
        data = pd.read_csv(config.TRAINING_DATA_FILE)
        target_column = config.TARGET_FEATURES 
        x = data.drop(target_column, axis=1)
        y = data[target_column]
        return x, y


    def load_raw_test_data() -> pd.DataFrame:
        """Load the test data."""
        data = pd.read_csv(config.TEST_DATA_FILE)
        return data
