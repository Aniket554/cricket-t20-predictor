"""Generate fake matches in Cricsheet's JSON schema so the pipeline can be tested offline.
Players have latent strike-rate skill; players move teams between seasons; one venue is high-scoring."""
import json
import random
from pathlib import Path


def make_synthetic(out_dir, n_matches=420, seed=7):
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    teams = [f"Team {c}" for c in "ABCDEF"]
    venues = {"Big Ground": 0.9, "Small Ground": 1.15, "Mid Ground": 1.0}
    players = {t: [f"{t[-1]}Player{i}" for i in range(14)] for t in teams}
    skill = {p: rng.gauss(125, 18) for t in teams for p in players[t]}
    bowl_skill = {p: rng.gauss(8.3, 0.7) for t in teams for p in players[t]}   # economy

    for m in range(n_matches):
        year = 2018 + m * 7 // n_matches
        if m > 0 and m % 60 == 0:            # offseason: a few players switch teams
            a, b = rng.sample(teams, 2)
            i, j = rng.randrange(14), rng.randrange(14)
            players[a][i], players[b][j] = players[b][j], players[a][i]
        t1, t2 = rng.sample(teams, 2)
        venue = rng.choice(list(venues))
        league = "Synthetic Premier League" if m % 3 else "Synthetic Cup"   # two competitions
        vmult = venues[venue] * (1.0 if m % 3 else 1.12)                  # the Cup scores faster
        date = f"{year}-{rng.randint(3, 9):02d}-{rng.randint(1, 28):02d}"
        reg = {p: f"id_{p}" for t in (t1, t2) for p in players[t]}
        innings = []
        for bat, bowl in ((t1, t2), (t2, t1)):
            order = players[bat][:11]
            bowlers = players[bowl][6:11]
            cur = [0, 1]
            nxt = 2
            overs = []
            wkts = 0
            for o in range(20):
                if wkts >= 10:
                    break
                bw = bowlers[o % 5]
                dels = []
                legal = 0
                while legal < 6 and wkts < 10:
                    striker, non = order[cur[0]], order[cur[1]]
                    phase_mult = 0.85 if o < 6 else 1.0 if o < 15 else 1.35
                    mean_sr = skill[striker] / 100 * vmult * phase_mult * (8.3 / bowl_skill[bw])
                    d = {"batter": striker, "bowler": bw, "non_striker": non}
                    if rng.random() < 0.03:
                        d["runs"] = {"batter": 0, "extras": 1, "total": 1}
                        d["extras"] = {"wides": 1}
                    else:
                        p = max(0.3, mean_sr)
                        r = rng.choices([0, 1, 2, 3, 4, 6], [0.38 / p, 0.34, 0.1, 0.01, 0.1 * p, 0.05 * p])[0]
                        d["runs"] = {"batter": r, "extras": 0, "total": r}
                        legal += 1
                        if rng.random() < 0.045 + (0.01 if o >= 15 else 0):
                            d["wickets"] = [{"player_out": striker, "kind": rng.choice(["bowled", "caught", "lbw", "run out"])}]
                            wkts += 1
                            if nxt < 11:
                                cur[0] = nxt
                                nxt += 1
                            else:
                                wkts = 10
                        elif r % 2 == 1:
                            cur = [cur[1], cur[0]]
                    dels.append(d)
                if legal == 6:
                    cur = [cur[1], cur[0]]
                overs.append({"over": o, "deliveries": dels})
            innings.append({"team": bat, "overs": overs})
        doc = {"info": {"dates": [date], "venue": venue, "city": "X", "teams": [t1, t2],
                        "match_type": "T20", "gender": "male", "season": str(year),
                        "event": {"name": league},
                        "toss": {"winner": t1, "decision": "bat"}, "outcome": {"winner": t1},
                        "registry": {"people": reg}},
               "innings": innings}
        (out / f"{1000 + m}.json").write_text(json.dumps(doc))
    return out
