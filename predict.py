"""Score new connections with a trained model (run train.py first).

By default this labels the 22,544 unlabelled connections in data/Test_data.csv.

Usage:
    python predict.py
    python predict.py --model neural_network
    python predict.py --input my_traffic.csv --output my_predictions.csv
"""
import argparse
import sys

import joblib
import numpy as np
import pandas as pd
import torch

from src import config
from src.evaluate import compute_metrics
from src.models import IntrusionNet, predict_proba_nn

MODELS = ["random_forest", "decision_tree", "logistic_regression", "neural_network"]


def anomaly_probability(model_key, X):
    """Return P(anomaly) for each row and the decision threshold the model uses."""
    if model_key == "neural_network":
        bundle = torch.load(config.MODEL_DIR / "neural_network.pt")
        prep = joblib.load(config.MODEL_DIR / "nn_preprocessor.joblib")
        model = IntrusionNet(bundle["input_size"], bundle["hidden1"], bundle["hidden2"])
        model.load_state_dict(bundle["state_dict"])
        return predict_proba_nn(model, prep.transform(X)), bundle["threshold"]

    pipeline = joblib.load(config.MODEL_DIR / f"{model_key}.joblib")
    return pipeline.predict_proba(X)[:, 1], 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, default="random_forest")
    parser.add_argument("--input", default=str(config.TEST_CSV),
                        help="CSV with the 41 feature columns (a 'class' column is optional)")
    parser.add_argument("--output", default=str(config.RESULTS_DIR / "test_predictions.csv"))
    args = parser.parse_args()

    needed = config.MODEL_DIR / ("neural_network.pt" if args.model == "neural_network"
                                 else f"{args.model}.joblib")
    if not needed.exists():
        sys.exit(f"{needed} not found - run `python train.py` first.")

    df = pd.read_csv(args.input)
    labels = df.pop(config.TARGET) if config.TARGET in df.columns else None

    prob, threshold = anomaly_probability(args.model, df)
    pred = (prob >= threshold).astype(int)

    out = pd.DataFrame({
        "row": np.arange(len(df)),
        "prediction": np.where(pred == 1, "anomaly", "normal"),
        "anomaly_probability": prob.round(6),
    })
    out.to_csv(args.output, index=False)

    n_anom = int(pred.sum())
    print(f"Model: {args.model} (threshold {threshold:.4f})")
    print(f"Scored {len(df):,} connections from {args.input}")
    print(f"  normal : {len(df) - n_anom:,}")
    print(f"  anomaly: {n_anom:,} ({n_anom / len(df):.1%})")
    print(f"Predictions written to {args.output}")

    if labels is not None:
        y_true = (labels != "normal").astype(int)
        for k, v in compute_metrics(y_true, pred, prob).items():
            print(f"  {k:<9}: {v:.4f}")


if __name__ == "__main__":
    main()
