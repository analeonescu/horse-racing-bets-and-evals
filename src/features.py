"""Point-in-time feature engineering.

The one rule everything here must respect: a feature for a given race may only use
information that would have been known BEFORE that race happened. Every rolling/expanding
stat is built with .shift() before .expanding() so a row never sees its own outcome.
"""
import numpy as np
import pandas as pd

# horse_type is a real-data quality issue in this dataset: a handful of rows have coat
# colours ('Brown', 'Roan', 'Grey') where a sex should be. We bucket those into 'Other'
# rather than silently misclassifying them as a sex.
_MALE_TYPES = {"Gelding", "Colt", "Horse", "Rig"}
_FEMALE_TYPES = {"Mare", "Filly"}


def add_horse_sex(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["horse_sex"] = np.select(
        [df["horse_type"].isin(_MALE_TYPES), df["horse_type"].isin(_FEMALE_TYPES)],
        ["Male", "Female"],
        default="Unknown",
    )
    return df


def add_gear_dummies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One-hot the '/'-separated horse_gear codes (e.g. 'TT/B' -> TT=1, B=1)."""
    df = df.copy()
    gear_dummies = df["horse_gear"].fillna("--").str.get_dummies(sep="/")
    gear_dummies = gear_dummies.drop(columns=["--"], errors="ignore")
    gear_dummies.columns = [f"gear_{c}" for c in gear_dummies.columns]
    df = pd.concat([df, gear_dummies], axis=1)
    return df, gear_dummies.columns.tolist()


def _prior_stats(df: pd.DataFrame, group_col: str, prefix: str) -> pd.DataFrame:
    df = df.sort_values([group_col, "date"])
    runs_col = f"{prefix}_prior_runs"
    rate_col = f"{prefix}_prior_win_rate"
    df[runs_col] = df.groupby(group_col).cumcount()
    df[rate_col] = (
        df.groupby(group_col)["won"]
        .apply(lambda s: s.shift().expanding().mean())
        .reset_index(level=0, drop=True)
    )
    return df


class PointInTimeFeatures:
    """Fits fill-values on TRAIN ONLY, applies identically to any split.

    This mirrors sklearn's fit/transform convention deliberately: it's what makes it
    structurally hard to leak train-period statistics into a fill value used on test rows,
    which was the original bug this class replaces.
    """

    GROUPS = ("horse_id", "jockey_id", "trainer_id")

    def __init__(self):
        self.fill_values_: dict[str, float] = {}

    def fit(self, train_df: pd.DataFrame) -> "PointInTimeFeatures":
        prepped = self._add_raw_stats(train_df)
        for group_col in self.GROUPS:
            prefix = group_col.replace("_id", "")
            rate_col = f"{prefix}_prior_win_rate"
            self.fill_values_[rate_col] = prepped[rate_col].mean()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._add_raw_stats(df)
        for group_col in self.GROUPS:
            prefix = group_col.replace("_id", "")
            rate_col = f"{prefix}_prior_win_rate"
            df[rate_col] = df[rate_col].fillna(self.fill_values_[rate_col])
        df["is_debut"] = (df["horse_prior_runs"] == 0).astype(int)
        return df.sort_values(["date", "race_id"]).reset_index(drop=True)

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)

    @staticmethod
    def _add_raw_stats(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for group_col in PointInTimeFeatures.GROUPS:
            prefix = group_col.replace("_id", "")
            df = _prior_stats(df, group_col, prefix)
        return df


NUMERIC_FEATURES = [
    "horse_age", "declared_weight", "actual_weight", "draw", "distance", "race_class",
    "field_size",
    "horse_prior_runs", "horse_prior_win_rate",
    "jockey_prior_runs", "jockey_prior_win_rate",
    "trainer_prior_runs", "trainer_prior_win_rate",
    "is_debut",
]

CATEGORICAL_FEATURES = ["venue", "surface", "going", "horse_sex", "horse_country"]
# gear_* columns are added dynamically by add_gear_dummies() and appended by build_features()

TARGET = "won"


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Apply the non-time-dependent feature steps (sex mapping, gear dummies).

    Point-in-time historical stats are NOT built here -- those depend on the train/test split
    and must go through PointInTimeFeatures.fit()/.transform() after splitting, never before.

    Returns (df, categorical_features, gear_cols). gear_cols are already binary 0/1 indicators
    (one per gear code), so they belong on the NUMERIC side of the preprocessor -- passing
    them through OneHotEncoder would needlessly double each into two one-hot columns.
    """
    df = add_horse_sex(df)
    df, gear_cols = add_gear_dummies(df)
    return df, CATEGORICAL_FEATURES, gear_cols