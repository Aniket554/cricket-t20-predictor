"""Streamlit app: forecast a T20 player's strike rate / economy in a new league, team and role.
    python -m streamlit run app.py        (run from the project root, after scripts.train)"""
import joblib
import pandas as pd
import streamlit as st

from cricket_predictor import data_io, predict
from cricket_predictor.config import DATA_DIR

st.set_page_config(page_title="T20 Player Predictor", page_icon="🏏", layout="wide")

DATASETS = {"batter": data_io.BATTING, "bowler": data_io.BOWLING}
MODELS = {"batter": DATA_DIR / "model_batters_5match.joblib",
          "bowler": DATA_DIR / "model_bowlers_5match.joblib"}


@st.cache_data(show_spinner="Loading data...")
def load(kind):
    return data_io.load_dataset(DATASETS[kind])


@st.cache_resource(show_spinner="Loading model...")
def load_model(kind):
    return joblib.load(MODELS[kind])


st.title("🏏 T20 Player Predictor")
st.caption("How will a player perform after moving to a new league, team or batting slot? "
           "The model predicts how far he lands above or below his recent-form baseline "
           "(his last 20 innings or spells).")

kind = st.sidebar.radio("Player type", ["batter", "bowler"], format_func=str.title)
missing = [p for p in (DATASETS[kind], MODELS[kind]) if not p.exists()]
if missing:
    st.error("Missing files:\n\n" + "\n".join(f"- `{p}`" for p in missing) +
             "\n\nRun the pipeline and `python -m scripts.train` first (see README).")
    st.stop()

df, bundle = load(kind), load_model(kind)
players = predict.player_table(df, kind)
if players.empty:
    st.warning("No players with enough matches in the data.")
    st.stop()

# --- inputs ---------------------------------------------------------------------------------
idcol = predict.KINDS[kind]["id"]
labels = {r[idcol]: f"{r['name']}  ·  {r['team']} ({r['matches']} matches)"
          for _, r in players.iterrows()}
pid = st.sidebar.selectbox("Player", list(labels), format_func=labels.get)
cur = players[players[idcol] == pid].iloc[0]

levels = predict.league_levels(df)
comps = sorted(levels.index)
comp = st.sidebar.selectbox("Destination league", comps,
                            index=comps.index(cur["competition"]) if cur["competition"] in comps else 0)

teams = sorted(df[df["competition"] == comp]["team"].dropna().unique())
team = st.sidebar.selectbox("Destination team", teams)

venue_opts = ["(league average)"] + list(predict.venues_for(df, comp).index)
venue = st.sidebar.selectbox("Venue", venue_opts)
innings = st.sidebar.radio("Innings", [1, 2], format_func=lambda i: "Batting first" if i == 1 else "Chasing",
                           horizontal=True)
position = None
if kind == "batter":
    position = st.sidebar.slider("Expected batting position", 1, 11,
                                 int(round(df[df[idcol] == pid].tail(20)["position"].mean())))

res = predict.predict_scenario(df, bundle, kind, pid, comp, team, innings=innings, position=position,
                               venue=None if venue == "(league average)" else venue)

# --- outputs --------------------------------------------------------------------------------
unit = res["unit"].title()
good = (res["delta"] > 0) == res["higher_is_better"]
c1, c2, c3 = st.columns(3)
c1.metric(f"Recent-form {unit}", f"{res['baseline']:.2f}")
c2.metric(f"Predicted {unit}", f"{res['predicted']:.2f}",
          f"{res['delta']:+.2f}", delta_color="normal" if res["higher_is_better"] else "inverse")
c3.metric("Verdict", "Likely to improve" if good else "Likely to dip")
st.subheader(res["name"])

left, right = st.columns([3, 2])
with left:
    st.markdown(f"**Recent {res['unit']} (last 30 appearances)**")
    st.line_chart(res["recent"].set_index("date"))
with right:
    st.markdown("**What moves the forecast**")
    st.caption("Each row: forecast change versus the same player with that factor unchanged.")
    st.dataframe(res["breakdown"].rename(columns={"factor": "Factor", "effect": f"Effect on {res['unit']}"}),
                 hide_index=True, use_container_width=True)

with st.expander("Model inputs used for this forecast"):
    st.dataframe(pd.DataFrame([res["features"]]).T.rename(columns={0: "value"}))
st.caption("Forecasts are statistical estimates from past ball-by-ball data, not guarantees. "
           "Per-match T20 results are very noisy; the model's edge over 'same as recent form' is modest.")
