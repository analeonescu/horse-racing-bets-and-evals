"""Model pipelines and prediction utilities."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_logistic_pipeline(numeric_features, categorical_features):
    """Build the baseline logistic-regression pipeline."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric, numeric_features),
        ("categorical", categorical, categorical_features),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=2000)),
    ])


def fit_model(model, X_train, y_train):
    """Fit a scikit-learn compatible model."""
    return model.fit(X_train, y_train)


def predict_prob(model, X):
    """Return positive-class probabilities."""
    return model.predict_proba(X)[:, 1]


def normalise_within_race(probabilities, race_ids):
    """Normalise probabilities so they sum to one within each race."""
    probabilities = np.asarray(probabilities, dtype=float)
    race_ids = np.asarray(race_ids)
    output = np.zeros_like(probabilities)

    for race in np.unique(race_ids):
        mask = race_ids == race
        total = probabilities[mask].sum()
        if total > 0:
            output[mask] = probabilities[mask] / total

    return output
