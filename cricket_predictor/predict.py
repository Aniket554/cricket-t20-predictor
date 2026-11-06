"""Scenario predictions: "what if this player joined team X in league Y, batting at slot P?"

Takes a player's last BASELINE_WINDOW appearances as the recent-form baseline, builds the same
features the models were trained on for the hypothetical situation, and returns
baseline + predicted residual. A what-if breakdown shows how much each factor group moved the
forecast by switching it back to 'no change' and re-predicting."""
import numpy as np
import pandas as pd

from .config import BASELINE_WINDOW, DEFAULT_PAR

KINDS = {
    "batter": dict(id="batter_id", name="batter", baseline="baseline_sr", unit="strike rate",
                   metric="strike_rate", higher_is_better=True),
    "bowler": dict(id="bowler_id", name="bowler", baseline="baseline_econ", unit="economy",
                   metric="economy", higher_is_better=False),
}


def league_levels(df):
    """Latest known scoring level per competition (runs per 100 balls / per over)."""
    d = df.dropna(subset=["league_level"]).sort_values(["date", "match_id"])
    return d.groupby("competition")["league_level"].last()


def venues_for(df, competition):
    """venue -> latest prior-average first-innings total, for one competition."""
    d = df[df["competition"] == competition].sort_values(["date", "match_id"])
    return d.groupby("venue")["venue_par"].last().sort_values(ascending=False)


def player_table(df, kind, min_matches=10):
    """One row per player: name, appearances, last team/competition/date. For pickers."""
    k = KINDS[kind]
    d = df.sort_values(["date", "match_id"])
    g = d.groupby(k["id"])
    t = pd.DataFrame({
        "name": g[k["name"]].last(), "matches": g.size(), "team": g["team"].last(),
        "competition": g["competition"].last(), "last_date": g["date"].last()})
    t = t[t["matches"] >= min_matches].sort_values("last_date", ascending=False)
    return t.reset_index()


def _hist(df, kind, player_id):
    k = KINDS[kind]
    h = df[df[k["id"]] == player_id].sort_values(["date", "match_id"])
    if h.empty:
        raise ValueError(f"unknown player id: {player_id}")
    return h


def _base_features(h, kind):
    """Recent-form features from the player's last window of appearances."""
    last = h.tail(BASELINE_WINDOW)
    balls = last["balls"].sum()
    div = lambda a, b: float(a / b) if b > 0 else np.nan
    f = {"death_share": div(last["balls_death"].sum(), balls),
         "powerplay_share": div(last["balls_powerplay"].sum(), balls)}
    if kind == "batter":
        f.update(baseline_sr=div(last["runs"].sum() * 100, balls),
                 baseline_out_rate=div(last["out"].sum(), balls),
                 pos_mean=float(last["position"].mean()))
    else:
        f.update(baseline_econ=div(last["runs"].sum() * 6, balls),
                 baseline_wkts24=div(last["wickets"].sum() * 24, balls),
                 workload_mean=float(last["balls"].mean()))
    return f


def build_scenario(df, kind, player_id, competition, team, innings=1, position=None,
                   venue=None, today=None):
    h = _hist(df, kind, player_id)
    last = h.tail(BASELINE_WINDOW)
    f = _base_features(h, kind)
    levels = league_levels(df)
    usual_level = float(last["league_level"].mean())
    dest_level = float(levels.get(competition, np.nan))

    if venue is not None:
        vp = venues_for(df, competition).get(venue, np.nan)
    else:
        vp = np.nan
    if np.isnan(vp):
        v = df[df["competition"] == competition]["venue_par"]
        vp = float(v.mean()) if len(v) else DEFAULT_PAR

    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()
    days = min(max((today - h["date"].iloc[-1]).days, 0), 400)

    f.update(innings=innings, venue_par=float(vp),
             games_with_team=int((h["team"] == team).sum()),
             moved_team=int(h["team"].iloc[-1] != team),
             prior_n=len(h), days_since_last=days,
             league_level=dest_level, league_delta=dest_level - usual_level,
             new_league=int(competition not in set(h["competition"])))
    if kind == "batter":
        pos = float(position) if position is not None else float(round(f["pos_mean"]))
        f.update(position=pos, pos_delta=pos - f["pos_mean"])
    return f, usual_level, float(last["venue_par"].mean())


# factor group -> function(scenario row, neutral info) -> row with that group set to "no change"
def _neutralise(row, group, usual_level, usual_venue, h):
    r = dict(row)
    if group == "League change":
        r.update(league_level=usual_level, league_delta=0.0, new_league=0)
    elif group == "Team change":
        last_team = h["team"].iloc[-1]
        r.update(moved_team=0, games_with_team=int((h["team"] == last_team).sum()))
    elif group == "Venue":
        r.update(venue_par=usual_venue)
    elif group == "Batting position":
        r.update(position=r["pos_mean"], pos_delta=0.0)
    return r


def predict_scenario(df, bundle, kind, player_id, competition, team, innings=1, position=None,
                     venue=None, today=None):
    """Returns dict: baseline, predicted, delta, breakdown (DataFrame), unit, name, recent."""
    k = KINDS[kind]
    h = _hist(df, kind, player_id)
    row, usual_level, usual_venue = build_scenario(df, kind, player_id, competition, team,
                                                   innings, position, venue, today)
    X = pd.DataFrame([row])[bundle.features]
    baseline = row[k["baseline"]]
    delta = float(bundle.predict_residual(X)[0])
    pred = baseline + delta

    groups = ["League change", "Team change", "Venue"] + (["Batting position"] if kind == "batter" else [])
    rows = []
    for g in groups:
        cf = pd.DataFrame([_neutralise(row, g, usual_level, usual_venue, h)])[bundle.features]
        rows.append((g, delta - float(bundle.predict_residual(cf)[0])))
    breakdown = pd.DataFrame(rows, columns=["factor", "effect"])
    breakdown["effect"] = breakdown["effect"].round(2)

    recent = h.tail(30)[["date", k["metric"]]].rename(columns={k["metric"]: k["unit"]})
    return dict(baseline=float(baseline), predicted=float(pred), delta=delta, breakdown=breakdown,
                unit=k["unit"], name=h[k["name"]].iloc[-1], recent=recent, features=row,
                higher_is_better=k["higher_is_better"])
