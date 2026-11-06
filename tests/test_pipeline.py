import numpy as np
import pytest

from tests.synthetic import make_synthetic
from pipeline.parse_matches import parse_all
from pipeline.build_datasets import build_batting, build_bowling
from cricket_predictor import model


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = make_synthetic(tmp_path_factory.mktemp("raw"), n_matches=420)
    balls, matches = parse_all(d)
    return balls, matches, build_batting(balls, matches), build_bowling(balls, matches)


def test_parse_basics(data):
    balls, matches, _, _ = data
    assert len(matches) == 420
    assert set(balls["phase"]) == {"powerplay", "middle", "death"}
    assert matches["inn1_total"].between(30, 400).all()


def test_batting_rows_consistent(data):
    balls, _, bat, _ = data
    assert bat["runs"].sum() == balls["runs_batter"].sum()
    assert bat["out"].isin([0, 1]).all()
    assert bat["position"].between(1, 11).all()


def test_bowling_wickets_and_balls(data):
    balls, _, _, bowl = data
    assert bowl["wickets"].sum() == balls["bowler_wicket"].sum()
    assert bowl["balls"].sum() == balls["is_legal"].sum()


def test_no_leakage_in_baseline(data):
    """Baseline must equal the mean of strictly earlier innings, never include the current one."""
    _, _, bat, _ = data
    p = bat["batter_id"].value_counts().index[0]
    g = bat[bat["batter_id"] == p].sort_values(["date", "match_id"]).reset_index(drop=True)
    i = 10
    prev = g.iloc[max(0, i - 20):i]
    expected = prev["runs"].sum() * 100 / prev["balls"].sum()
    assert g.loc[i, "baseline_sr"] == pytest.approx(expected)
    assert np.isnan(g.loc[0, "baseline_sr"])


def test_league_level_uses_only_earlier_matches(data):
    _, _, bat, _ = data
    comp = bat["competition"].iloc[0]
    c = bat[bat["competition"] == comp]
    per_match = (c.groupby(["date", "match_id"]).agg(r=("runs", "sum"), b=("balls", "sum"))
                   .reset_index().sort_values(["date", "match_id"]).reset_index(drop=True))
    i = 15
    expected = per_match.loc[:i - 1, "r"].sum() * 100 / per_match.loc[:i - 1, "b"].sum()
    got = c[c["match_id"] == per_match.loc[i, "match_id"]]["league_level"].iloc[0]
    assert got == pytest.approx(expected)
    assert c[c["match_id"] == per_match.loc[0, "match_id"]]["league_level"].isna().all()


def test_new_league_flag_and_delta(data):
    _, _, bat, _ = data
    assert set(bat["competition"]) == {"Synthetic Premier League", "Synthetic Cup"}
    assert bat["new_league"].isin([0, 1]).all()
    assert bat["new_league"].sum() > 0
    # never flagged on a player's very first appearance
    assert (bat.loc[bat["prior_n"] == 0, "new_league"] == 0).all()
    assert bat["league_delta"].notna().sum() > 1000


def test_forward_window_target(data):
    """5-match target = runs/balls over this and the next 4 innings, vs the (earlier-only) baseline."""
    _, _, bat, _ = data
    p = bat["batter_id"].value_counts().index[0]
    g = bat[bat["batter_id"] == p].sort_values(["date", "match_id"]).reset_index(drop=True)
    i = 12
    w = g.iloc[i:i + 5]
    expected = w["runs"].sum() * 100 / w["balls"].sum() - g.loc[i, "baseline_sr"]
    assert g.loc[i, "residual_sr5"] == pytest.approx(expected)
    assert g.loc[i, "end_date"] == g.loc[i + 4, "date"]
    assert np.isnan(g.iloc[-1]["residual_sr5"])          # no full window at the end


def test_purge_removes_overlapping_training_rows(data):
    _, _, bat, _ = data
    cutoff = bat["date"].quantile(0.8)
    b = model.train_residual_model(bat, model.BAT_FEATURES, model.BAT_TARGET5, weight="fwd_balls",
                                   cutoff=cutoff, purge_col="end_date")
    d = bat[(bat["prior_n"] >= 5) & bat["residual_sr5"].notna() & bat["baseline_sr"].notna()]
    unpurged = ((d["date"] < cutoff)).sum()
    assert b.metrics["n_train"] < unpurged            # some rows reached into the test period
    assert b.metrics["mae_blend"] < b.metrics["mae_baseline_only"] * 1.1


def test_models_train_and_beat_zero_baseline_or_close(data):
    _, _, bat, bowl = data
    b = model.train_residual_model(bat, model.BAT_FEATURES, model.BAT_TARGET, weight="balls")
    assert b.metrics["n_test"] > 100
    # sanity: model should not be dramatically worse than predicting "no change"
    assert b.metrics["mae_blend"] < b.metrics["mae_baseline_only"] * 1.1
    w = model.train_residual_model(bowl, model.BOWL_FEATURES, model.BOWL_TARGET, weight="balls")
    assert w.metrics["mae_blend"] < w.metrics["mae_baseline_only"] * 1.1
