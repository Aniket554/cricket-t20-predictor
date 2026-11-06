"""Parse Cricsheet JSON -> one row per ball (balls.csv.gz) + one row per match (matches.csv)."""
import json
from pathlib import Path

import pandas as pd

from cricket_predictor.config import RAW_DIR, DATA_DIR, T20_TYPES, NON_BOWLER_KINDS, phase_of


def parse_match(path, genders=("male",)):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    info = d.get("info", {})
    if info.get("match_type") not in T20_TYPES:
        return None
    if genders and info.get("gender") not in genders:
        return None

    people = info.get("registry", {}).get("people", {})
    pid = lambda name: people.get(name, name)
    teams = info.get("teams", [])
    match_id = Path(path).stem
    event = info.get("event", {}).get("name", "")
    toss = info.get("toss", {})
    outcome = info.get("outcome", {})

    rows = []
    for inn_no, inn in enumerate(d.get("innings", []), 1):
        if inn.get("super_over"):
            continue
        bat_team = inn["team"]
        bowl_team = next((t for t in teams if t != bat_team), None)
        seq = 0
        for ov in inn.get("overs", []):
            over = ov["over"]
            for dl in ov.get("deliveries", []):
                ex = dl.get("extras", {})
                wide = ex.get("wides", 0) > 0
                noball = ex.get("noballs", 0) > 0
                r = dl["runs"]
                wk = dl.get("wickets", [])
                w = wk[0] if wk else None
                rows.append(dict(
                    match_id=match_id, innings=inn_no, seq=seq, over=over,
                    phase=phase_of(over),
                    batting_team=bat_team, bowling_team=bowl_team,
                    batter=dl["batter"], batter_id=pid(dl["batter"]),
                    bowler=dl["bowler"], bowler_id=pid(dl["bowler"]),
                    non_striker_id=pid(dl["non_striker"]),
                    runs_batter=r["batter"], runs_total=r["total"],
                    runs_bowler=r["batter"] + ex.get("wides", 0) + ex.get("noballs", 0),
                    is_legal=not (wide or noball),
                    is_faced=not wide,
                    is_four=r["batter"] == 4, is_six=r["batter"] == 6,
                    player_out_id=pid(w["player_out"]) if w else None,
                    bowler_wicket=bool(w and w["kind"] not in NON_BOWLER_KINDS),
                ))
                seq += 1

    match = dict(
        match_id=match_id, date=info["dates"][0], season=str(info.get("season", "")),
        venue=info.get("venue", ""), city=info.get("city", ""), competition=event,
        gender=info.get("gender"), team1=teams[0] if teams else None,
        team2=teams[1] if len(teams) > 1 else None,
        toss_winner=toss.get("winner"), toss_decision=toss.get("decision"),
        winner=outcome.get("winner"),
    )
    return match, rows


def parse_all(raw_dir=RAW_DIR, genders=("male",)):
    matches, balls = [], []
    for p in sorted(Path(raw_dir).rglob("*.json")):
        try:
            res = parse_match(p, genders)
        except (KeyError, ValueError) as e:      # skip malformed files, keep going
            print(f"skip {p.name}: {e}")
            continue
        if res:
            matches.append(res[0])
            balls.extend(res[1])
    balls = pd.DataFrame(balls)
    matches = pd.DataFrame(matches)
    if balls.empty:
        return balls, matches
    tot = balls.groupby(["match_id", "innings"])["runs_total"].sum().unstack()
    matches = matches.merge(
        tot.rename(columns={1: "inn1_total", 2: "inn2_total"})[["inn1_total", "inn2_total"]]
        .reset_index(), on="match_id", how="left")
    matches["date"] = pd.to_datetime(matches["date"])
    return balls, matches


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    b, m = parse_all()
    b.to_csv(DATA_DIR / "balls.csv.gz", index=False)
    m.to_csv(DATA_DIR / "matches.csv", index=False)
    print(f"{len(m):,} matches, {len(b):,} balls")
