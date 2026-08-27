"""De-vig market odds and compare against model predictions.

win_odds/place_odds are used HERE ONLY -- as the benchmark to compare against, never as a
model input. See models.py docstring.
"""
import numpy as np
import pandas as pd


def implied_prob(win_odds: pd.Series) -> pd.Series:
    return 1 / win_odds


def devig_within_race(df: pd.DataFrame, race_col: str, implied_col: str = "market_implied_prob") -> pd.Series:
    """Normalize implied probabilities within each race so they sum to 1, removing the
    bookmaker's overround/vig."""
    return df.groupby(race_col)[implied_col].transform(lambda p: p / p.sum())


def expected_value(model_prob: pd.Series, decimal_odds: pd.Series) -> pd.Series:
    """EV per unit staked at the given decimal odds, if model_prob is the true probability."""
    return model_prob * decimal_odds - 1


def kelly_fraction(model_prob: pd.Series, decimal_odds: pd.Series) -> pd.Series:
    b = decimal_odds - 1
    f = (model_prob * decimal_odds - 1) / b
    return f.clip(lower=0.0)


def realized_profit(won: pd.Series, decimal_odds: pd.Series, stake: pd.Series) -> pd.Series:
    return np.where(won == 1, stake * (decimal_odds - 1), -stake)


def backtest_ev_threshold(df: pd.DataFrame, ev_col: str, odds_col: str, won_col: str,
                           thresholds=(0.0, 0.05, 0.10, 0.20, 0.50)) -> pd.DataFrame:
    """Flat $1-stake backtest: for each EV threshold, bet on every horse whose model EV
    exceeds it, and report bet count / profit / ROI."""
    rows = []
    for t in thresholds:
        bets = df[df[ev_col] > t]
        profit = np.where(bets[won_col] == 1, bets[odds_col] - 1, -1).sum() if len(bets) else 0.0
        rows.append({
            "ev_threshold": t,
            "n_bets": len(bets),
            "profit": profit,
            "roi": profit / len(bets) if len(bets) else float("nan"),
        })
    return pd.DataFrame(rows)
