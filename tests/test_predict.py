from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from tests.synthetic import make_synthetic
from pipeline.parse_matches import parse_all
from pipeline.build_datasets import build_batting, build_bowling
from cricket_predictor import model, predict


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = make_synthetic(tmp_path_factory.mktemp("raw"), n_matches=420)
    balls, matches = parse_all(d)
    bat, bowl = build_batting(balls, matches), build_bowling(balls, matches)
    bb = model.train_residual_model(bat, model.BAT_FEATURES, model.BAT_TARGET5,
                                    weight="fwd_balls", purge_col="end_date")
    wb = model.train_residual_model(bowl, model.BOWL_FEATURES, model.BOWL_TARGET5,
                                    weight="fwd_balls", purge_col="end_date")
    return bat, bowl, bb, wb


def test_player_table_and_levels(world):
    bat, _, _, _ = world
    t = predict.player_table(bat, "batter", min_matches=10)
    assert len(t) > 20 and t["matches"].min() >= 10
    lv = predict.league_levels(bat)
    assert set(lv.index) == {"Synthetic Premier League", "Synthetic Cup"}
    assert lv["Synthetic Cup"] > lv["Synthetic Premier League"]       # the Cup scores faster


def test_batter_scenario_baseline_matches_history(world):
    bat, _, bb, _ = world
    pid = predict.player_table(bat, "batter").iloc[0]["batter_id"]
    res = predict.predict_scenario(bat, bb, "batter", pid, "Synthetic Cup", "Team A",
                                   position=3, today="2026-01-01")
    last = bat[bat["batter_id"] == pid].sort_values(["date", "match_id"]).tail(20)
    assert res["baseline"] == pytest.approx(last["runs"].sum() * 100 / last["balls"].sum())
    assert res["predicted"] == pytest.approx(res["baseline"] + res["delta"])
    assert set(res["breakdown"]["factor"]) == {"League change", "Team change", "Venue", "Batting position"}
    assert res["features"]["position"] == 3 and len(res["recent"]) <= 30


def test_no_change_scenario_has_zero_league_and_position_effect(world):
    bat, _, bb, _ = world
    row = predict.player_table(bat, "batter").iloc[0]
    pid, comp, team = row["batter_id"], row["competition"], row["team"]
    res = predict.predict_scenario(bat, bb, "batter", pid, comp, team, today="2026-01-01")
    f = res["features"]
    assert f["moved_team"] == 0 and f["new_league"] == 0
    eff = res["breakdown"].set_index("factor")["effect"]
    assert eff["Team change"] == pytest.approx(0, abs=1e-6)
    assert eff["Batting position"] == pytest.approx(0, abs=0.5)    # rounded slot may differ a little


def test_bowler_scenario_and_unknown_player(world):
    _, bowl, _, wb = world
    row = predict.player_table(bowl, "bowler").iloc[0]
    res = predict.predict_scenario(bowl, wb, "bowler", row["bowler_id"], "Synthetic Cup", "Team B",
                                   today="2026-01-01")
    assert res["unit"] == "economy" and 3 < res["baseline"] < 15
    assert "Batting position" not in set(res["breakdown"]["factor"])
    assert not np.isnan(res["predicted"])
    with pytest.raises(ValueError):
        predict.predict_scenario(bowl, wb, "bowler", "nobody", "Synthetic Cup", "Team B")


def test_streamlit_app_runs_end_to_end(world, tmp_path, monkeypatch):
    """Save the synthetic datasets + models where the app expects them and run it headlessly."""
    from streamlit.testing.v1 import AppTest
    from cricket_predictor import data_io, config
    bat, bowl, bb, wb = world
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    for name, p, obj in [("bat", "batting_dataset.csv.gz", bat), ("bowl", "bowling_dataset.csv.gz", bowl)]:
        obj.to_csv(tmp_path / p, index=False)
    joblib.dump(bb, tmp_path / "model_batters_5match.joblib")
    joblib.dump(wb, tmp_path / "model_bowlers_5match.joblib")
    monkeypatch.setattr(data_io, "BATTING", tmp_path / "batting_dataset.csv.gz")
    monkeypatch.setattr(data_io, "BOWLING", tmp_path / "bowling_dataset.csv.gz")

    app = str(Path(__file__).resolve().parents[1] / "app.py")
    at = AppTest.from_file(app, default_timeout=120).run()
    assert not at.exception, at.exception
    assert len(at.metric) >= 3
    at.sidebar.radio[0].set_value("bowler").run()
    assert not at.exception, at.exception
