""" This module contains functions to post-process the data."""

import pandas as pd
from src.config import PREDICTIONS_PATH


def save_predictions(predictions: pd.DataFrame, file_name: str) -> None:
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

    predictions.to_csv(predictions_filename, index=False)


def post_process(data: pd.DataFrame) -> pd.DataFrame:
    """Post-process the data."""
    return data
