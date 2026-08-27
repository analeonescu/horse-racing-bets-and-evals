import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.betting import backtest_ev_threshold, devig_within_race, expected_value, implied_prob, realized_profit



def test_implied_prob():
    odds = pd.Series([2.0, 4.0, 8.0])
    result = implied_prob(odds)
    assert result.tolist() == pytest.approx([0.5, 0.25, 0.125])


def test_devig_norm():
    df = pd.DataFrame({
        "race_id": [1, 1, 1, 2, 2],
        "win_odds": [2.0, 4.0, 8.0, 3.0, 3.0],  # implied: 0.5, 0.25, 0.125 (sum=0.875); 0.333, 0.333
    })
    df["market_implied_prob"] = implied_prob(df["win_odds"])
    df["market_fair_prob"] = devig_within_race(df, "race_id")
    sums = df.groupby("race_id")["market_fair_prob"].sum()
    
    assert sums.values == pytest.approx([1.0, 1.0])


def test_devig_overround():
    """A race with real bookmaker margin (implied probs summing > 1) should end up with
    fair probs summing to exactly 1 after de-vigging."""
    
    df = pd.DataFrame({"race_id": [1, 1], "win_odds": [1.5, 1.5]})  # implied 0.667 each, sums to 1.33
    df["market_implied_prob"] = implied_prob(df["win_odds"])
    
    assert df["market_implied_prob"].sum() > 1.0  # confirms the vig is present before de-vig
    
    df["market_fair_prob"] = devig_within_race(df, "race_id")
    
    assert df["market_fair_prob"].sum() == pytest.approx(1.0)


def test_devig_relative_ratios():
    """De-vigging should rescale probabilities, not reorder or distort them -- a horse
    twice as likely as another (by raw implied prob) must still be exactly twice as
    likely after de-vigging."""
    
    df = pd.DataFrame({"race_id": [1, 1], "win_odds": [2.0, 4.0]})  # implied 0.5, 0.25 -> ratio 2:1
    df["market_implied_prob"] = implied_prob(df["win_odds"])
    df["market_fair_prob"] = devig_within_race(df, "race_id")
    ratio = df["market_fair_prob"].iloc[0] / df["market_fair_prob"].iloc[1]
    
    assert ratio == pytest.approx(2.0)


def test_devig_single_horse_race():
    """Degenerate case: a 'race' with one priced horse should always end up at
    fair prob 1.0 after normalization, regardless of its raw odds."""
    
    df = pd.DataFrame({"race_id": [1], "win_odds": [5.0]})
    df["market_implied_prob"] = implied_prob(df["win_odds"])
    df["market_fair_prob"] = devig_within_race(df, "race_id")
    
    assert df["market_fair_prob"].iloc[0] == pytest.approx(1.0)


def test_ev_positive():
    # p=0.5 at decimal odds 3.0: EV = 0.5*3.0 - 1 = 0.5
    
    ev = expected_value(pd.Series([0.5]), pd.Series([3.0]))
    
    assert ev.iloc[0] == pytest.approx(0.5)


def test_ev_negative():
    # p=0.3 at decimal odds 2.0: EV = 0.3*2.0 - 1 = -0.4
    
    ev = expected_value(pd.Series([0.3]), pd.Series([2.0]))
    
    assert ev.iloc[0] == pytest.approx(-0.4)


def test_ev_zero_breakeven():
    # p=0.5 at decimal odds 2.0 (i.e. model prob matches a fair, unvigged coin-flip price)
    
    ev = expected_value(pd.Series([0.5]), pd.Series([2.0]))
    
    assert ev.iloc[0] == pytest.approx(0.0)


def test_realized_profit_win_and_loss():
    
    won = pd.Series([1, 0])
    odds = pd.Series([3.0, 3.0])
    stake = pd.Series([1.0, 1.0])
    profit = realized_profit(won, odds, stake)
    # win: stake * (odds - 1) = 1 * 2 = 2.0 ; loss: -stake = -1.0
    
    assert profit.tolist() == pytest.approx([2.0, -1.0])


def test_realized_profit_scales():
    won = pd.Series([1])
    odds = pd.Series([5.0])
    stake = pd.Series([2.0])
    profit = realized_profit(won, odds, stake)
    
    assert profit[0] == pytest.approx(2.0 * (5.0 - 1))



def _toy_bets_df():
    # 4 horses: two clearly +EV, two clearly -EV/negative, at known odds/outcomes
    return pd.DataFrame({
        "ev_xgb": [0.30, 0.10, -0.10, -0.30],
        "win_odds": [4.0, 3.0, 2.0, 5.0],
        "won": [1, 0, 0, 0],
    })


def test_backtest_ev_threshold():
    
    df = _toy_bets_df()
    result = backtest_ev_threshold(df, "ev_xgb", "win_odds", "won", thresholds=(0.0, 0.2))
    counts = dict(zip(result["ev_threshold"], result["n_bets"]))
    
    assert counts[0.0] == 2   # ev > 0.0 -> rows with 0.30 and 0.10
    assert counts[0.2] == 1   # ev > 0.2 -> only the 0.30 row


def test_backtest_ev_threshold_profit():
    
    df = _toy_bets_df()
    result = backtest_ev_threshold(df, "ev_xgb", "win_odds", "won", thresholds=(0.0,))
    row = result.iloc[0]
    # at threshold 0.0: bets are (ev=0.30, odds=4.0, won=1) and (ev=0.10, odds=3.0, won=0)
    # profit = (4.0 - 1) win + (-1) loss = 3.0 - 1.0 = 2.0
    assert row["n_bets"] == 2
    assert row["profit"] == pytest.approx(2.0)
    assert row["roi"] == pytest.approx(1.0)  # profit / n_bets = 2.0 / 2


def test_backtest_ev_no_qualifying_bets():
    
    df = _toy_bets_df()
    result = backtest_ev_threshold(df, "ev_xgb", "win_odds", "won", thresholds=(1.0,))
    row = result.iloc[0]
    
    assert row["n_bets"] == 0
    assert row["profit"] == 0.0
    assert np.isnan(row["roi"])