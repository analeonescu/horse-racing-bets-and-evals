import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import PointInTimeFeatures, add_gear_dummies, add_horse_sex


def _toy_df():
    """Horse A runs 3 times, horse B once. Jockey J1 rides A twice AND B once -- deliberately
    so jockey/trainer stats have to pool correctly across different horses, not just replicate
    the horse-level logic by coincidence."""
    return pd.DataFrame({
        "race_id": [1, 2, 3, 4],
        "horse_id": ["A", "A", "A", "B"],
        "jockey_id": ["J1", "J1", "J2", "J1"],
        "trainer_id": ["T1", "T1", "T1", "T1"],
        "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-01-15"]),
        "won": [1, 0, 1, 0],
    })


def test_first_appearance_is_debut():
    df = _toy_df()
    out = PointInTimeFeatures().fit(df).transform(df)
    row = out[(out["horse_id"] == "A") & (out["date"] == "2024-01-01")].iloc[0]
    assert row["horse_prior_runs"] == 0
    assert row["is_debut"] == 1


def test_second_appearance_not_debut():
    df = _toy_df()
    out = PointInTimeFeatures().fit(df).transform(df)
    row = out[(out["horse_id"] == "A") & (out["date"] == "2024-02-01")].iloc[0]
    assert row["horse_prior_runs"] == 1
    assert row["is_debut"] == 0


def test_prior_win_rate_validity():
    """Horse A won its 1st race (2024-01-01) and hasn't run again by its 2nd race
    (2024-02-01) -- prior_win_rate there must be 1.0, not influenced by the 3rd race."""
    df = _toy_df()
    out = PointInTimeFeatures().fit(df).transform(df)
    row = out[(out["horse_id"] == "A") & (out["date"] == "2024-02-01")].iloc[0]
    assert row["horse_prior_win_rate"] == pytest.approx(1.0)


def test_debut_row_prior_cyclicity():
    """The classic leakage bug: a horse's FIRST race has no history, so its prior_win_rate
    must come from the fitted fill value -- never from that same race's own won=1/0."""
    df = _toy_df()
    pit = PointInTimeFeatures().fit(df)
    out = pit.transform(df)
    row = out[(out["horse_id"] == "A") & (out["date"] == "2024-01-01")].iloc[0]
    assert row["horse_prior_win_rate"] == pytest.approx(pit.fill_values_["horse_prior_win_rate"])


def test_jockey_stats():
    """J1's 2nd-ever ride is on horse B (2024-01-15), after winning on horse A (2024-01-01).
    jockey_prior_win_rate must reflect that cross-horse history, not reset per horse."""
    df = _toy_df()
    out = PointInTimeFeatures().fit(df).transform(df)
    row = out[(out["jockey_id"] == "J1") & (out["date"] == "2024-01-15")].iloc[0]
    assert row["jockey_prior_runs"] == 1
    assert row["jockey_prior_win_rate"] == pytest.approx(1.0)


def test_trainer_stats():
    """T1 trains every horse in the fixture. By horse A's 3rd race (2024-03-01), T1 has
    3 prior runs across horses A and B combined: A won 2024-01-01, B lost 2024-01-15,
    A lost 2024-02-01 -- 1 win out of 3."""
    df = _toy_df()
    out = PointInTimeFeatures().fit(df).transform(df)
    row = out[(out["horse_id"] == "A") & (out["date"] == "2024-03-01")].iloc[0]
    assert row["trainer_prior_runs"] == 3
    assert row["trainer_prior_win_rate"] == pytest.approx(1 / 3)


def test_fill_value_computed_from_train_only():
    """Regression test for the original bug: filling debut NaNs with a value computed
    across the FULL dataset (including future/test rows) rather than train alone.

    Expected value is hardcoded rather than recomputed via the module's own internals,
    so this doesn't just check 'fit() agrees with itself' -- if the underlying stat were
    wrong, this would still catch it.
    """
    df = _toy_df()
    train = df[df["date"] < "2024-02-15"]  # excludes horse A's 3rd race (2024-03-01)
    pit = PointInTimeFeatures().fit(train)
    # within `train`, the only row with any prior history is horse A's 2nd race
    # (2024-02-01), where prior_win_rate = 1.0 (it won its one earlier race). Every other
    # row in `train` is a debut (prior_win_rate = NaN, excluded from the mean).
    assert pit.fill_values_["horse_prior_win_rate"] == pytest.approx(1.0)


def test_fill_value_ignores_rows_outside_the_fitted_split():
    """Fitting on progressively more of the same data should change the fill value once
    new non-debut rows enter the fitted window -- confirms fit() is actually sensitive to
    what it's given, not silently reading the full df regardless of the `train_df` argument."""
    df = _toy_df()
    train_early = df[df["date"] < "2024-01-20"]   # only horse A's 1st race + horse B's -- both debuts
    train_later = df[df["date"] < "2024-02-15"]    # also includes horse A's 2nd race (non-debut)

    fill_early = PointInTimeFeatures().fit(train_early).fill_values_["horse_prior_win_rate"]
    fill_later = PointInTimeFeatures().fit(train_later).fill_values_["horse_prior_win_rate"]

    assert np.isnan(fill_early)  # no non-debut horse rows yet -- mean of an all-NaN column
    assert fill_later == pytest.approx(1.0)


def test_all_debut():
    """Documents current behaviour rather than asserting it's 'correct': fitting on a split
    where every row is a debut yields NaN fill values, which will propagate silently into
    transform() output rather than raising. Worth knowing before this hits production."""
    df = pd.DataFrame({
        "race_id": [1, 2],
        "horse_id": ["A", "B"],
        "jockey_id": ["J1", "J2"],
        "trainer_id": ["T1", "T2"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "won": [1, 0],
    })
    pit = PointInTimeFeatures().fit(df)
    assert np.isnan(pit.fill_values_["horse_prior_win_rate"])

    out = pit.transform(df)
    assert out["horse_prior_win_rate"].isna().all()


@pytest.mark.parametrize("horse_type,expected", [
    ("Gelding", "Male"),
    ("Colt", "Male"),
    ("Horse", "Male"),
    ("Rig", "Male"),
    ("Mare", "Female"),
    ("Filly", "Female"),
    ("Roan", "Unknown"),   # coat colour, not a sex -- must not be guessed
    ("Brown", "Unknown"),
    ("Grey", "Unknown"),
])

def test_horse_sex_mapping(horse_type, expected):
    df = pd.DataFrame({"horse_type": [horse_type]})
    out = add_horse_sex(df)
    assert out["horse_sex"].iloc[0] == expected


def test_gear_dummies():
    df = pd.DataFrame({"horse_gear": ["TT/B", "B", "--", None]})
    out, gear_cols = add_gear_dummies(df)
    assert "gear_TT" in gear_cols and "gear_B" in gear_cols
    assert out.loc[0, "gear_TT"] == 1 and out.loc[0, "gear_B"] == 1
    assert out.loc[1, "gear_TT"] == 0 and out.loc[1, "gear_B"] == 1


def test_gear_dummies_no_gear():
    df = pd.DataFrame({"horse_gear": ["TT/B", "B", "--", None]})
    out, gear_cols = add_gear_dummies(df)
    assert "gear_--" not in gear_cols
    # rows 2 and 3 (no gear / missing) should have all gear indicators = 0
    no_gear_rows = out.loc[[2, 3], gear_cols]
    assert (no_gear_rows == 0).all().all()