"""Residual models: predict how far a player lands above/below his recent-form baseline.
Blend = ridge (stable, linear) + gradient boosting (non-linear interactions).
Swap HistGradientBoosting for LightGBM/CatBoost once those are installed."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import MIN_PRIOR

BAT_FEATURES = ["baseline_sr", "baseline_out_rate", "death_share", "powerplay_share",
                "pos_mean", "position", "pos_delta", "innings", "venue_par",
                "games_with_team", "moved_team", "prior_n", "days_since_last",
                "league_level", "league_delta", "new_league"]
BAT_TARGET, BAT_BASELINE = "residual_sr", "baseline_sr"
BAT_TARGET5 = "residual_sr5"          # next-5-innings version, weighted by fwd_balls

BOWL_FEATURES = ["baseline_econ", "baseline_wkts24", "death_share", "powerplay_share",
                 "workload_mean", "innings", "venue_par", "games_with_team",
                 "moved_team", "prior_n", "days_since_last",
                 "league_level", "league_delta", "new_league"]
BOWL_TARGET, BOWL_BASELINE = "residual_econ", "baseline_econ"
BOWL_TARGET5 = "residual_econ5"


def wmae(y, p, w):
    return float(np.average(np.abs(np.asarray(y) - np.asarray(p)), weights=w))


@dataclass
class ResidualBundle:
    features: list
    target: str
    ridge: object
    gbm: object
    metrics: dict

    def predict_residual(self, X):
        X = X[self.features]
        return 0.5 * self.ridge.predict(X) + 0.5 * self.gbm.predict(X)


def train_residual_model(df, features, target, weight="balls", cutoff=None, min_prior=MIN_PRIOR,
                         purge_col=None):
    """purge_col: for forward-window targets pass 'end_date'. Training rows whose window reaches
    into the test period are dropped, so the test set is never seen through the target."""
    d = df[(df["prior_n"] >= min_prior) & df[target].notna() & df[weight].notna()].copy()
    d = d.dropna(subset=["baseline_sr" if "baseline_sr" in d else "baseline_econ"])
    cutoff = pd.Timestamp(cutoff) if cutoff is not None else d["date"].quantile(0.8)
    train, test = d[d["date"] < cutoff], d[d["date"] >= cutoff]
    if purge_col is not None:
        train = train[pd.to_datetime(train[purge_col]) < cutoff]
    if len(train) < 50 or len(test) < 10:
        raise ValueError(f"not enough data: train={len(train)}, test={len(test)}")

    ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
    gbm = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=300,
                                        min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    ridge.fit(train[features], train[target], ridge__sample_weight=train[weight])
    gbm.fit(train[features], train[target], sample_weight=train[weight])
    bundle = ResidualBundle(features, target, ridge, gbm, {})

    y, w = test[target], test[weight]
    preds = {"ridge": ridge.predict(test[features]), "gbm": gbm.predict(test[features])}
    preds["blend"] = bundle.predict_residual(test)
    m = {"n_train": len(train), "n_test": len(test), "cutoff": str(cutoff.date()),
         "mae_baseline_only": wmae(y, np.zeros(len(y)), w)}
    for k, p in preds.items():
        m[f"mae_{k}"] = wmae(y, p, w)
    moved = test["moved_team"].values == 1
    if moved.sum() >= 10:
        m["n_moved_test"] = int(moved.sum())
        m["mae_baseline_only_moved"] = wmae(y[moved], np.zeros(moved.sum()), w[moved])
        m["mae_blend_moved"] = wmae(y[moved], preds["blend"][moved], w[moved])
    bundle.metrics = m
    return bundle


def feature_importance(bundle, df, weight="balls", n=3000):
    """Permutation importance of the GBM on a sample (model-agnostic, cheap)."""
    from sklearn.inspection import permutation_importance
    d = df[df["prior_n"] >= MIN_PRIOR].dropna(subset=[bundle.target, weight]).tail(n)
    r = permutation_importance(bundle.gbm, d[bundle.features], d[bundle.target],
                               sample_weight=d[weight], n_repeats=5, random_state=0)
    return pd.Series(r.importances_mean, index=bundle.features).sort_values(ascending=False)
