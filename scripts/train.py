"""Train batter + bowler residual models on data/*.csv.gz and save them.
Each is trained twice: on the single-innings target and on the smoother next-5-innings target.

    python -m scripts.train [--cutoff 2024-01-01]"""
import argparse
import joblib

from cricket_predictor import data_io, model
from cricket_predictor.config import DATA_DIR

# name, dataset path, features, target, weight column, purge column
SPECS = [
    ("batters_1match", data_io.BATTING, model.BAT_FEATURES, model.BAT_TARGET, "balls", None),
    ("batters_5match", data_io.BATTING, model.BAT_FEATURES, model.BAT_TARGET5, "fwd_balls", "end_date"),
    ("bowlers_1match", data_io.BOWLING, model.BOWL_FEATURES, model.BOWL_TARGET, "balls", None),
    ("bowlers_5match", data_io.BOWLING, model.BOWL_FEATURES, model.BOWL_TARGET5, "fwd_balls", "end_date"),
]


def pct(base, new):
    return f"{(base - new) / base * 100:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default=None, help="train on matches before this date")
    args = ap.parse_args()

    cache = {}
    for name, path, feats, target, weight, purge in SPECS:
        if path not in cache:
            cache[path] = data_io.load_dataset(path)
        df = cache[path]
        b = model.train_residual_model(df, feats, target, weight=weight,
                                       cutoff=args.cutoff, purge_col=purge)
        m = b.metrics
        print(f"\n== {name} ==")
        for k, v in m.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
        print(f"  improvement over baseline (all):   {pct(m['mae_baseline_only'], m['mae_blend'])}")
        if "mae_blend_moved" in m:
            print(f"  improvement over baseline (moved): "
                  f"{pct(m['mae_baseline_only_moved'], m['mae_blend_moved'])}")
        print("  top factors:")
        print(model.feature_importance(b, df, weight=weight).head(8).round(4).to_string())
        joblib.dump(b, DATA_DIR / f"model_{name}.joblib")


if __name__ == "__main__":
    main()
