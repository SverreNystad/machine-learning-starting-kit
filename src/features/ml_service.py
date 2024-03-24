""" This module contains the functions to prepare the data for the machine learning model. """

import pandas as pd
from sklearn.model_selection import train_test_split
from src.data.data_loader import load_raw_data, load_raw_test_data
from src.features.feature_engineering import engineer_features


def prepare_data(test_size: float = 0.2) -> tuple:
    """
    Prepare the data for the machine learning model.
    This includes loading the data, feature engineering, and splitting the data into
    training, validation, and testing sets.

    The test_size parameter represents the proportion of the dataset to include in the test split,
    and the validation_size parameter represents the proportion of the training dataset to include
    in the validation split.

    Returns:
        A tuple containing the training, validation, and test data.
        x_train, y_train, x_test, y_test.

    Example:
        >>> X_train, X_validate, X_test, y_train, y_validate, y_test = prepare_data()
        >>> model.fit(X_train, y_train)
        >>> valid_score = model.score(X_validate, y_validate)
        >>> test_score = model.score(X_test, y_test)
    """
    # Load the data
    x, y = load_raw_data()

    # Feature engineering
    engineered_features = engineer_features(x)

    # Split the data into training and test sets
    RANDOM_STATE: int = 42
    x_train, x_test, y_train, y_test = train_test_split(
        engineered_features, y, test_size=test_size, random_state=RANDOM_STATE
    )

    return x_train, x_test, y_train, y_test

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """ Remove duplicates"""

def prepare_test_data() -> pd.DataFrame:
    """
    Prepare the test data for the machine learning model.

    Returns
        The test data.

    Example:
        >>> test_data = prepare_test_data()
        >>> predictions = model.predict(test_data)
    """
    # Load the data
    test_data = load_raw_test_data()

    # Feature engineering
    engineered_features = engineer_features(test_data)

    return engineered_features
