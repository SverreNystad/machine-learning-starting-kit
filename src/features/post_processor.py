""" This module contains functions to post-process the data."""

from datetime import datetime
import pandas as pd
from src.config import PREDICTIONS_PATH


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


def post_process(predictions: pd.DataFrame) -> pd.DataFrame:
    """Post-process the predictions."""
    return predictions
