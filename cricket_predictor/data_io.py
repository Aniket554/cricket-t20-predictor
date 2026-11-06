import pandas as pd
from .config import DATA_DIR

BALLS = DATA_DIR / "balls.csv.gz"
MATCHES = DATA_DIR / "matches.csv"
BATTING = DATA_DIR / "batting_dataset.csv.gz"
BOWLING = DATA_DIR / "bowling_dataset.csv.gz"


def load_balls(path=BALLS):
    return pd.read_csv(path, low_memory=False)


def load_matches(path=MATCHES):
    return pd.read_csv(path, parse_dates=["date"])


def load_dataset(path):
    return pd.read_csv(path, parse_dates=["date"], low_memory=False)
