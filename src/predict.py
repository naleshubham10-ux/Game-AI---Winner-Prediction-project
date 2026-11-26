"""
Predict winPlacePerc for every player in a CSV with the PUBG schema (e.g. test_V2.csv).

    python src/predict.py --input data/test_V2.csv --output reports/predictions.csv
"""
import argparse

import joblib
import numpy as np
import pandas as pd

from config import MODEL_DIR
from preprocessing import build_group_features, postprocess, reduce_memory


def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    bundle = joblib.load(MODEL_DIR / "best_model.joblib")
    X, _, players = build_group_features(df.drop(columns=["winPlacePerc"], errors="ignore"))
    X = X.reindex(columns=bundle["features"], fill_value=0)
    pred = np.clip(bundle["model"].predict(X), 0, 1)
    if bundle.get("use_postprocess"):
        pred = postprocess(pred, X)
    group_pred = pd.Series(pred, index=X.index, name="winPlacePerc").reset_index()
    out = players[["Id", "matchId", "groupId"]].merge(group_pred, on=["matchId", "groupId"], how="left")
    return out[["Id", "winPlacePerc"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="reports/predictions.csv")
    args = ap.parse_args()
    df = reduce_memory(pd.read_csv(args.input))
    preds = predict_dataframe(df)
    preds.to_csv(args.output, index=False)
    print(f"Wrote {len(preds):,} predictions -> {args.output}")


if __name__ == "__main__":
    main()
