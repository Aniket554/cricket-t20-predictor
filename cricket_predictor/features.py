"""Shared feature helpers. Everything here uses ONLY information from before the match
(shift(1) before any rolling/expanding op) so there is no leakage."""
import numpy as np
import pandas as pd

from .config import BASELINE_WINDOW, DEFAULT_PAR


def add_prior_window(df, key, sum_cols, mean_cols=(), window=BASELINE_WINDOW):
    """Rolling sums/means over each player's previous `window` appearances (current excluded)."""
    df = df.sort_values([key, "date", "match_id"]).reset_index(drop=True)
    g = df.groupby(key, sort=False)
    for c in sum_cols:
        df[f"prior_{c}"] = g[c].transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum())
    for c in mean_cols:
        df[f"prior_{c}_mean"] = g[c].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    df["prior_n"] = g.cumcount()
    return df


def add_team_context(df, player_key):
    """games_with_team: prior games for this franchise/side. moved_team: previous team in the
    same competition differs from the current one (the 'acquired player' signal)."""
    df = df.sort_values([player_key, "date", "match_id"]).reset_index(drop=True)
    df["games_with_team"] = df.groupby([player_key, "team"]).cumcount()
    prev = df.groupby([player_key, "competition"])["team"].shift(1)
    df["moved_team"] = (prev.notna() & (prev != df["team"])).astype(int)
    df["days_since_last"] = df.groupby(player_key)["date"].diff().dt.days
    return df


def add_venue_par(matches):
    """Expanding mean of PRIOR first-innings totals at each venue."""
    m = matches.sort_values(["date", "match_id"]).copy()
    m["venue_par"] = m.groupby("venue")["inn1_total"].transform(
        lambda s: s.shift(1).expanding().mean())
    m["venue_par"] = m["venue_par"].fillna(DEFAULT_PAR)
    m["venue_matches_prior"] = m.groupby("venue").cumcount()
    return m[["match_id", "venue_par", "venue_matches_prior"]]


def safe_div(a, b):
    return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def add_league_level(df, scale):
    """league_level = scoring level of this competition using only matches BEFORE this one
    (batting: runs per 100 balls, bowling: runs per 6 balls; pass scale=100 or 6).
    Columns used: competition, date, match_id, runs, balls."""
    m = (df.groupby(["competition", "date", "match_id"])
           .agg(r=("runs", "sum"), b=("balls", "sum")).reset_index()
           .sort_values(["date", "match_id"]))
    m["cum_r"] = m.groupby("competition")["r"].cumsum() - m["r"]
    m["cum_b"] = m.groupby("competition")["b"].cumsum() - m["b"]
    m["league_level"] = safe_div(m["cum_r"] * scale, m["cum_b"])
    return df.merge(m[["match_id", "competition", "league_level"]],
                    on=["match_id", "competition"], how="left")


def add_league_change(df, player_key):
    """league_delta: this league's level minus the level of the leagues the player usually plays
    in (prior window). new_league: first appearance in this competition after earlier ones elsewhere.
    Call AFTER add_prior_window(..., mean_cols=[..., 'league_level'])."""
    df["league_delta"] = df["league_level"] - df["prior_league_level_mean"]
    first_here = df.groupby([player_key, "competition"]).cumcount() == 0
    df["new_league"] = (first_here & (df["prior_n"] > 0)).astype(int)
    return df


def add_forward_window(df, key, sum_cols, window):
    """Sums over this appearance and the next window-1 (needs df sorted by key/date, as
    add_prior_window leaves it). NaN near the end of a player's history. end_date = date of
    the last appearance in the window, used to purge training rows that overlap the test period."""
    g = df.groupby(key, sort=False)
    for c in sum_cols:
        df[f"fwd_{c}"] = g[c].transform(
            lambda s: s[::-1].rolling(window, min_periods=window).sum()[::-1])
    df["end_date"] = g["date"].shift(-(window - 1))
    return df
