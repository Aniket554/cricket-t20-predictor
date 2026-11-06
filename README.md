# T20 Player Performance Predictor

Predicts how a T20 batter or bowler will perform in a new context (new team, new batting slot,
new venue), modelled on the NHL "acquired players" approach: the model predicts the **residual**
between a player's recent-form baseline and what he actually does.

## Run it
```bash
pip install -r requirements.txt
python -m pipeline.download_cricsheet        # IPL, BBL, PSL, CPL, SA20, ILT20, MLC, T20 Blast, LPL, BPL, T20Is
python -m pipeline.parse_matches             # -> data/balls.csv.gz, data/matches.csv
python -m pipeline.build_datasets            # -> data/batting_dataset.csv.gz, bowling_dataset.csv.gz
python -m scripts.train --cutoff 2024-01-01  # 1-innings and next-5-innings targets, metrics + top factors
python -m pytest tests                       # runs on synthetic Cricsheet-format data
python -m streamlit run app.py               # forecast app (needs the steps above first)
```

## Factors modelled

| Group | Batter | Bowler |
|---|---|---|
| Form baseline | rolling 20-innings strike rate, dismissal rate | rolling economy, wickets per 24 balls |
| Phase role | powerplay / death share of balls faced | powerplay / death share of balls bowled |
| Role change | batting position vs usual slot (`pos_delta`) | usual workload |
| Venue | prior average first-innings total (`venue_par`) | same |
| Team move | `moved_team`, `games_with_team` (settling-in) | same |
| Match | innings (bat first / chase), days since last match | same |
| League | `league_level` (prior scoring level of the competition), `league_delta` (vs the leagues he usually plays in), `new_league` | same |

All baselines use only matches *before* the one being predicted (tested in `test_no_leakage_in_baseline`).

## The app

`app.py` lets you pick a batter or bowler, a destination league, team, venue, innings and (for batters)
batting slot, then shows recent form, the forecast, and a what-if table: how much each factor group
(league change, team change, venue, batting position) moved the forecast versus leaving it unchanged.
Logic lives in `cricket_predictor/predict.py`; it uses the 5-innings models.

## Targets

- `residual_sr` / `residual_econ`: one innings (or spell) minus the recent-form baseline. Very noisy.
- `residual_sr5` / `residual_econ5`: the next 5 appearances (this one included) minus the same baseline,
  weighted by balls. Smoother. Training rows whose 5-match window reaches into the test period are
  purged, so the test set is never seen through the target.

## Known limits / next steps

- `position` is known only once the match starts. For "what if he joins team X" predictions, supply
  the expected slot as a scenario input (like power-play role in the NHL app).
- Not yet included: weather/dew, batter-vs-bowler-type matchups, player age (Cricsheet's
  people register lacks DOB; needs a join to another source), expected-runs-per-ball model (xR, the xG analogue).
- Per-match residuals are noisy; weighting by balls faced/bowled helps, and season-level targets are a good next experiment.
- Swap `HistGradientBoostingRegressor` for LightGBM/CatBoost to match the NHL blend.
