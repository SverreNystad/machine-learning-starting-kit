""" This module contains the functions to prepare the data for the machine learning model. """

import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
from src.config import PREDICTIONS_PATH
from src.data.data_loader import DataLoader, create_data_loader
from src.features.feature_engineering import engineer_features
from src.features.post_processor import post_process


def prepare_data(
    validation_size: float = 0.1,
    test_size: float = 0.1,
    loader: DataLoader = create_data_loader(),
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Prepare the data for the machine learning model.
    This includes loading the data, feature engineering, and splitting the data into
    training, validation, and testing sets.

    Args:
        validation_size: The size of the validation set. Should be between 0.0 and 1.0.
        test_size: The size of the test set. Should be between 0.0 and 1.0.
        loader: An instance of the DataLoader class to load the data.

    Returns:
        A tuple containing the training, validation, and test data.
        x_train, x_validate, x_test, y_train, y_validate, y_test.

    Example:
        >>> X_train, X_validate, X_test, y_train, y_validate, y_test = prepare_data()
        >>> model.fit(X_train, y_train)
        >>> valid_score = model.score(X_validate, y_validate)
        >>> test_score = model.score(X_test, y_test)
    """
    if not 0.0 <= validation_size <= 1.0:
        raise ValueError("validation_size should be between 0.0 and 1.0.")
    if not 0.0 <= test_size <= 1.0:
        raise ValueError("test_size should be between 0.0 and 1.0.")

    # Prepare the data
    x, y = loader.load_raw_data()

    engineered_features = engineer_features(x)

    empty_x = pd.DataFrame(columns=engineered_features.columns)
    empty_y = pd.DataFrame(columns=y.columns)

    # Determine the total size of the dataset to be allocated to training
    holdout_size = validation_size + test_size
    if holdout_size > 1:
        raise ValueError(
            f"The sum of validation_size and test_size should be less than 1. Got {holdout_size}."
        )

    # Define a random state for consistency in splits
    RANDOM_STATE = 42

    # Split the data into training, validation, and test sets
    if holdout_size == 0:
        # All data goes into training
        return engineered_features, empty_x, empty_x, y, empty_y, empty_y
    elif test_size == 0:
        # No test data
        x_train, x_validate, y_train, y_validate = train_test_split(
            engineered_features, y, test_size=validation_size, random_state=RANDOM_STATE
        )
        return x_train, x_validate, empty_x, y_train, y_validate, empty_y
    elif validation_size == 0:
        # No validation data
        x_train, x_test, y_train, y_test = train_test_split(
            engineered_features, y, test_size=test_size, random_state=RANDOM_STATE
        )
        return x_train, empty_x, x_test, y_train, empty_y, y_test

    # Calculate test size as a proportion of the holdout set
    temp_test_size = test_size / holdout_size

    # Split the data into training and a temporary set
    x_train, x_temp, y_train, y_temp = train_test_split(
        engineered_features, y, test_size=holdout_size, random_state=RANDOM_STATE
    )

    # Further split the temporary set into validation and test sets
    x_validate, x_test, y_validate, y_test = train_test_split(
        x_temp, y_temp, test_size=temp_test_size, random_state=RANDOM_STATE
    )

    return x_train, x_validate, x_test, y_train, y_validate, y_test


def prepare_test_data() -> pd.DataFrame:
    """
    Prepare the test data for the machine learning model.

    Returns
        The test data.

    Example:
        >>> test_data = prepare_test_data()
        >>> predictions = model.predict(test_data)
    """
    # Prepare the Test data
    loader: DataLoader = create_data_loader()
    test_data = loader.load_raw_test_data()

    # Feature engineering
    engineered_features = engineer_features(test_data)

    return engineered_features


def save_predictions(
    predictions: pd.DataFrame, file_name: str, shallHaveIndexColum: bool = True
) -> None:
    """
    Save the predictions to a file and post process predictions.

    Args:
        predictions: The predictions to save.
        file_name: The name of the file to save the predictions to.
    """

    # Post-process the predictions
    predictions = post_process(predictions)

    # Save the predictions to a CSV file
    predictions_filename = PREDICTIONS_PATH + file_name
    if not predictions_filename.endswith(".csv"):
        predictions_filename += ".csv"

    # Add time to the filename
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions_filename = predictions_filename.replace(".csv", f"_{current_time}.csv")

    predictions.to_csv(predictions_filename, index=shallHaveIndexColum)
