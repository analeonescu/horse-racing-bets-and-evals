from pathlib import Path

import pandas as pd

from src.data import load_data, sort_races_chronologically
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, PointInTimeFeatures, build_features
from src.models import build_logistic_pipeline, fit_model, normalise_within_race, predict_prob
from src.evaluation import add_model_edge, calibration_table, evaluate_predictions
from src.betting import backtest_ev_threshold, devig_within_race, expected_value, implied_prob


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "races.csv"
RESULTS_DIR = ROOT / "results"

TEST_SIZE = 0.2
EV_THRESHOLDS = (0.0, 0.05, 0.10, 0.20)


def split_by_race(df, test_size=0.2):
    """Split races chronologically so that complete races stay together."""
    races = (df[["race_id", "race_date"]]
        .drop_duplicates("race_id")
        .sort_values("race_date")
        )

    split = int(len(races) * (1 - test_size))
    train_ids = set(races.iloc[:split]["race_id"])
    test_ids = set(races.iloc[split:]["race_id"])

    train = df[df["race_id"].isin(train_ids)].copy()
    test = df[df["race_id"].isin(test_ids)].copy()

    return train, test


def top1_accuracy(df, prob_col):
    """Fraction of races where the highest-probability horse wins."""
    
    winners = df.loc[df.groupby("race_id")[prob_col].idxmax(),"won"]
    
    return winners.mean()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    
    df = load_data(DATA_PATH)
    df = sort_races_chronologically(df, "race_date")

    train_raw, test_raw = split_by_race(df, TEST_SIZE)

    print(f"Train: {train_raw['race_id'].nunique()} races, ",
          f"{len(train_raw)} runners", 
          f"\nTest:  {test_raw['race_id'].nunique()} races, ",
          f"{len(test_raw)} runners")

    # Build the static features first.
    train, categorical_features, train_gear_cols = build_features(train_raw)
    test, _, test_gear_cols = build_features(test_raw)

    gear_cols = sorted(set(train_gear_cols) | set(test_gear_cols))
    for col in gear_cols:
        if col not in train:
            train[col] = 0
        if col not in test:
            test[col] = 0

    # Historical features must only use information available before
    # each race. Fit the fill values on the training data.
    
    point_in_time = PointInTimeFeatures()
    point_in_time.fit(train)

    combined = pd.concat([train, test], ignore_index=True)
    combined = sort_races_chronologically(combined, "race_date")
    combined = point_in_time.transform(combined)

    train = combined[combined["race_id"].isin(train_raw["race_id"])].copy()
    test = combined[combined["race_id"].isin(test_raw["race_id"])].copy()

    numeric_features = [col for col in NUMERIC_FEATURES + gear_cols if col in train.columns]
    categorical_features = [col for col in CATEGORICAL_FEATURES if col in train.columns]
    feature_columns = numeric_features + categorical_features

    X_train = train[feature_columns]
    y_train = train["won"]
    X_test = test[feature_columns]
    y_test = test["won"]

    print("Training logistic regression...")
    
    model = build_logistic_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features
        )
    
    model = fit_model(model, X_train, y_train)

    test["model_prob"] = normalise_within_race(
        predict_prob(model, X_test),
        test["race_id"]
        )

    # Market probabilities
    test["implied_prob"] = implied_prob(test["win_odds"])
    test["market_prob"] = devig_within_race(test,
                                            race_col="race_id",
                                            implied_col="implied_prob"
                                            )

    # Predictive evaluation
    model_metrics = evaluate_predictions(y_test, test["model_prob"])
    market_metrics = evaluate_predictions(y_test, test["market_prob"])

    model_top1 = top1_accuracy(test, "model_prob")
    market_top1 = top1_accuracy(test, "market_prob")

    print("\nModel")
    print(f"Log loss:      {model_metrics['log_loss']:.4f}")
    print(f"Brier score:   {model_metrics['brier_score']:.4f}")
    print(f"Top-1 accuracy:{model_top1:.4f}")

    print("\nMarket")
    print(f"Log loss:      {market_metrics['log_loss']:.4f}")
    print(f"Brier score:   {market_metrics['brier_score']:.4f}")
    print(f"Top-1 accuracy:{market_top1:.4f}")

    # Compare model probabilities with the market.
    test = add_model_edge(test,
                          model_prob_col="model_prob",
                          market_prob_col="market_prob"
                          )
    test["expected_value"] = expected_value(test["model_prob"],
                                            test["win_odds"]
                                            )

    betting_results = backtest_ev_threshold(
        test,
        ev_col="expected_value",
        odds_col="win_odds",
        won_col="won",
        thresholds=EV_THRESHOLDS,
    )

    print("\nBetting results")
    print(betting_results.to_string(index=False))

    calibration = calibration_table(y_test,
                                    test["model_prob"],
                                    n_bins=10)

    predictions = test[
        ["race_id",
         "horse_id",
         "won",
         "win_odds",
         "model_prob",
         "market_prob",
         "edge",
         "expected_value"
         ]
        ]

    predictions.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    betting_results.to_csv(RESULTS_DIR / "betting_results.csv", index=False)
    calibration.to_csv(RESULTS_DIR / "calibration.csv", index=False)

    metrics = pd.DataFrame(
        {"model": [model_metrics["log_loss"],
                   model_metrics["brier_score"],
                   model_top1
                   ],
         "market": [market_metrics["log_loss"],
                    market_metrics["brier_score"],
                    market_top1
                    ]
         },
        index=["log_loss", "brier_score", "top1_accuracy"]
        )
    
    metrics.to_csv(RESULTS_DIR / "metrics.csv")

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()