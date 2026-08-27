"""Evaluation metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


def evaluate_predictions(y_true, probabilities, threshold=0.5):
    """Return standard binary prediction metrics."""
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(int)

    return {
        "log_loss": log_loss(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
        "accuracy": accuracy_score(y_true, predictions),
    }


def calibration_table(y_true, probabilities, n_bins=10):
    """Return a quantile-binned calibration table."""
    
    data = pd.DataFrame({"actual": y_true,"predicted": probabilities})
    data["bin"] = pd.qcut(data["predicted"], q=n_bins, duplicates="drop")
    
    return (data.groupby("bin", observed=True).agg(
            mean_predicted=("predicted", "mean"),
            observed_rate=("actual", "mean"),
            count=("actual", "size")).reset_index())


def add_model_edge(df, model_prob_col="model_prob", market_prob_col="market_prob"):
    """Add model probability minus market probability."""
    result = df.copy()
    result["edge"] = result[model_prob_col] - result[market_prob_col]
    return result
