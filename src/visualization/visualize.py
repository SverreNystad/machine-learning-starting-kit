""" Visualization functions to create exploratory and result oriented plots."""

from matplotlib import pyplot as plt
import pandas as pd

from src.config import PREDICTIONS_PATH, FIGURE_PATH


def plot_residual_predictions(submission_1_name: str, submission_2_name: str) -> None:
    """
    Plot the residuals of two submissions. The residuals are the difference between the two submissions.
    """

    prediction_1_path = PREDICTIONS_PATH + submission_1_name
    prediction_2_path = PREDICTIONS_PATH + submission_2_name
    if not prediction_1_path.endswith(".csv"):
        prediction_1_path += ".csv"
    if not prediction_2_path.endswith(".csv"):
        prediction_2_path += ".csv"

    prediction_1 = pd.read_csv(prediction_1_path)
    prediction_2 = pd.read_csv(prediction_2_path)
    residual = prediction_1 - prediction_2

    # Plot the residuals
    plt.figure(figsize=(12, 6))
    # Line plot of Residuals
    plt.plot(
        residual,
        label="Residuals",
        linestyle="-",
        marker="o",
        markersize=5,
        alpha=0.7,
        color="blue",
    )
    # Titles and labels
    plt.title(f"Residuals: {submission_1_name} - {submission_2_name}", fontsize=16)
    plt.ylabel("Pv Measurement ")
    plt.xlabel("Time")
    plt.legend()

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        f"{FIGURE_PATH}residuals/{submission_1_name}_vs_{submission_2_name}.png"
    )
    plt.show()
