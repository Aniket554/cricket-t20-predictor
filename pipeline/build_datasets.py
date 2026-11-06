"""balls + matches -> batting_dataset / bowling_dataset (one row per player per match),
with recent-form baselines and the residual targets the models learn."""
import pandas as pd

from cricket_predictor.config import DATA_DIR, PHASES, FORWARD_WINDOW
from cricket_predictor import data_io
from cricket_predictor.features import (add_prior_window, add_team_context, add_venue_par,
                                        add_league_level, add_league_change,
                                        add_forward_window, safe_div)


def _phase_counts(df, keys, prefix):
    pc = df.groupby(keys + ["phase"]).size().unstack(fill_value=0)
    pc = pc.reindex(columns=list(PHASES), fill_value=0)
    pc.columns = [f"{prefix}_{c}" for c in pc.columns]
    return pc.reset_index()


def _entry_positions(balls):
    """Batting position = order of first appearance (as batter or non-striker) in the innings."""
    cols = ["match_id", "innings", "seq"]
    a = balls[cols + ["batter_id"]].rename(columns={"batter_id": "pid"}).assign(k=0)
    b = balls[cols + ["non_striker_id"]].rename(columns={"non_striker_id": "pid"}).assign(k=1)
    long = pd.concat([a, b]).sort_values(cols + ["k"])
    long = long.drop_duplicates(["match_id", "innings", "pid"])
    long["position"] = long.groupby(["match_id", "innings"]).cumcount() + 1
    return long[["match_id", "innings", "pid", "position"]].rename(columns={"pid": "batter_id"})


def _attach_match_info(df, matches):
    par = add_venue_par(matches)
    info = matches[["match_id", "date", "venue", "competition"]].merge(par, on="match_id")
    df = df.merge(info, on="match_id", how="left")
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_batting(balls, matches):
    faced = balls[balls["is_faced"]]
    keys = ["match_id", "innings", "batter_id"]
    agg = faced.groupby(keys).agg(
        batter=("batter", "first"), team=("batting_team", "first"),
        runs=("runs_batter", "sum"), balls=("is_faced", "size"),
        fours=("is_four", "sum"), sixes=("is_six", "sum")).reset_index()
    agg = agg.merge(_phase_counts(faced, keys, "balls"), on=keys)

    outs = (balls[balls["player_out_id"].notna()]
            .groupby(["match_id", "innings", "player_out_id"]).size().clip(upper=1)
            .rename("out").reset_index().rename(columns={"player_out_id": "batter_id"}))
    agg = agg.merge(outs, on=keys, how="left")
    agg["out"] = agg["out"].fillna(0).astype(int)
    agg = agg.merge(_entry_positions(balls), on=["match_id", "innings", "batter_id"], how="left")

    agg = _attach_match_info(agg, matches)
    agg = add_league_level(agg, scale=100)
    agg = add_team_context(agg, "batter_id")
    agg = add_prior_window(agg, "batter_id",
                           sum_cols=["runs", "balls", "out", "balls_death", "balls_powerplay"],
                           mean_cols=["position", "league_level"])
    agg = add_league_change(agg, "batter_id")
    agg = add_forward_window(agg, "batter_id", ["runs", "balls"], FORWARD_WINDOW)

    agg["strike_rate"] = agg["runs"] / agg["balls"] * 100
    agg["baseline_sr"] = safe_div(agg["prior_runs"] * 100, agg["prior_balls"])
    agg["baseline_out_rate"] = safe_div(agg["prior_out"], agg["prior_balls"])
    agg["death_share"] = safe_div(agg["prior_balls_death"], agg["prior_balls"])
    agg["powerplay_share"] = safe_div(agg["prior_balls_powerplay"], agg["prior_balls"])
    agg["pos_mean"] = agg["prior_position_mean"]
    agg["pos_delta"] = agg["position"] - agg["pos_mean"]      # role change vs usual slot
    agg["residual_sr"] = agg["strike_rate"] - agg["baseline_sr"]
    # smoother target: strike rate over the next N innings vs the same baseline
    agg["residual_sr5"] = safe_div(agg["fwd_runs"] * 100, agg["fwd_balls"]) - agg["baseline_sr"]
    return agg


def build_bowling(balls, matches):
    keys = ["match_id", "innings", "bowler_id"]
    agg = balls.groupby(keys).agg(
        bowler=("bowler", "first"), team=("bowling_team", "first"),
        runs=("runs_bowler", "sum"), balls=("is_legal", "sum"),
        wickets=("bowler_wicket", "sum")).reset_index()
    agg = agg[agg["balls"] > 0]
    agg = agg.merge(_phase_counts(balls[balls["is_legal"]], keys, "balls"), on=keys)

    agg = _attach_match_info(agg, matches)
    agg = add_league_level(agg, scale=6)
    agg = add_team_context(agg, "bowler_id")
    agg = add_prior_window(agg, "bowler_id",
                           sum_cols=["runs", "balls", "wickets", "balls_death", "balls_powerplay"],
                           mean_cols=["balls", "league_level"])
    agg = add_league_change(agg, "bowler_id")
    agg = add_forward_window(agg, "bowler_id", ["runs", "balls"], FORWARD_WINDOW)

    agg["economy"] = agg["runs"] / agg["balls"] * 6
    agg["baseline_econ"] = safe_div(agg["prior_runs"] * 6, agg["prior_balls"])
    agg["baseline_wkts24"] = safe_div(agg["prior_wickets"] * 24, agg["prior_balls"])
    agg["death_share"] = safe_div(agg["prior_balls_death"], agg["prior_balls"])
    agg["powerplay_share"] = safe_div(agg["prior_balls_powerplay"], agg["prior_balls"])
    agg["workload_mean"] = agg["prior_balls_mean"]            # usual balls per game
    agg["residual_econ"] = agg["economy"] - agg["baseline_econ"]
    agg["residual_econ5"] = safe_div(agg["fwd_runs"] * 6, agg["fwd_balls"]) - agg["baseline_econ"]
    return agg


def main():
    balls, matches = data_io.load_balls(), data_io.load_matches()
    DATA_DIR.mkdir(exist_ok=True)
    bat, bowl = build_batting(balls, matches), build_bowling(balls, matches)
    bat.to_csv(data_io.BATTING, index=False)
    bowl.to_csv(data_io.BOWLING, index=False)
    print(f"batting rows: {len(bat):,}   bowling rows: {len(bowl):,}")


if __name__ == "__main__":
    main()
