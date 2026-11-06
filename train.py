"""Train, tune and evaluate all five models, then save metrics, figures and models.

Models
  * Logistic Regression          - GridSearchCV (5-fold) over C
  * Neural Network (baseline)    - untuned config from the presentation, threshold 0.5
  * Neural Network (tuned)       - grid search on a validation split + tuned threshold
  * Decision Tree                - GridSearchCV (5-fold)
  * Random Forest                - GridSearchCV (5-fold)

Every model is scored on the same 5,039-row test set, which is never used for
tuning.

Usage:
    python train.py
"""
import itertools
import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src import config, plots
from src.data import build_preprocessor, load_data, split_features_target, train_test_split_data
from src.evaluate import compute_metrics, youden_threshold
from src.models import predict_proba_nn, sklearn_models, train_nn


def run_sklearn_models(X_train, y_train, X_test):
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    results = {}
    for name, (pipeline, grid) in sklearn_models().items():
        n_combos = int(np.prod([len(v) for v in grid.values()]))
        print(f"\n[{name}] grid search: {n_combos} combinations x {config.CV_FOLDS} folds")
        start = time.perf_counter()
        search = GridSearchCV(pipeline, grid, scoring="f1", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        seconds = time.perf_counter() - start

        best = search.best_estimator_
        params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        print(f"  best params: {params}  (CV F1 {search.best_score_:.4f}, {seconds:.1f}s)")
        results[name] = {
            "model": best,
            "y_prob": best.predict_proba(X_test)[:, 1],
            "y_pred": best.predict(X_test),
            "params": params,
            "threshold": 0.5,
            "seconds": seconds,
        }
    return results


def run_neural_networks(X_train, y_train, X_test):
    # Hold out part of the training data so the test set is never used for tuning.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=config.VAL_SIZE, stratify=y_train,
        random_state=config.SEED)
    prep = build_preprocessor(scale=True).fit(X_tr)
    Xtr, Xval, Xte = prep.transform(X_tr), prep.transform(X_val), prep.transform(X_test)
    print(f"\n[Neural Network] {Xtr.shape[1]} input features; "
          f"train {len(Xtr):,} / validation {len(Xval):,}")

    # Baseline: the "before tuning" configuration, threshold 0.5.
    start = time.perf_counter()
    base_model, _ = train_nn(Xtr, y_tr, Xval, y_val, **config.NN_BASELINE)
    base_seconds = time.perf_counter() - start
    base_prob = predict_proba_nn(base_model, Xte)

    # Grid search, scored on validation F1.
    print(f"  grid search over {np.prod([len(v) for v in config.NN_GRID.values()])} "
          f"configurations ({config.NN_EPOCHS} epochs each)")
    rows, trained = [], {}
    start = time.perf_counter()
    for values in itertools.product(*config.NN_GRID.values()):
        params = dict(zip(config.NN_GRID.keys(), values))
        model, history = train_nn(Xtr, y_tr, Xval, y_val, **params)
        val_pred = (predict_proba_nn(model, Xval) >= 0.5).astype(int)
        row = {**params,
               "val_accuracy": float((val_pred == np.asarray(y_val)).mean()),
               "val_f1": f1_score(y_val, val_pred)}
        rows.append(row)
        trained[values] = (model, history)
        print("    lr={lr:<6} h1={hidden1:<3} h2={hidden2:<3} bs={batch_size:<3} "
              "-> val acc {val_accuracy:.4f}  val F1 {val_f1:.4f}".format(**row))
    grid_seconds = time.perf_counter() - start

    grid_df = pd.DataFrame(rows).sort_values("val_f1", ascending=False, kind="stable")
    best_row = grid_df.iloc[0]
    best_params = {k: best_row[k] for k in config.NN_GRID}
    best_params = {k: (int(v) if k != "lr" else float(v)) for k, v in best_params.items()}
    best_model, best_history = trained[tuple(best_params.values())]

    # Threshold tuning on the validation set.
    val_prob = predict_proba_nn(best_model, Xval)
    threshold = youden_threshold(y_val, val_prob)
    test_prob = predict_proba_nn(best_model, Xte)
    print(f"  best: {best_params}, tuned threshold {threshold:.4f}")

    results = {
        "Neural Network (baseline)": {
            "y_prob": base_prob,
            "y_pred": (base_prob >= 0.5).astype(int),
            "params": dict(config.NN_BASELINE),
            "threshold": 0.5,
            "seconds": base_seconds,
        },
        "Neural Network (tuned)": {
            "model": best_model,
            "y_prob": test_prob,
            "y_pred": (test_prob >= threshold).astype(int),
            "params": best_params,
            "threshold": threshold,
            "seconds": grid_seconds,
        },
    }
    extras = {"grid": grid_df, "history": best_history, "prep": prep,
              "y_val": y_val, "val_prob": val_prob, "threshold": threshold,
              "best_params": best_params, "input_size": Xtr.shape[1]}
    return results, extras


def write_metrics(results, y_test):
    rows = []
    for name in plots.MODEL_ORDER:
        r = results[name]
        r["metrics"] = compute_metrics(y_test, r["y_pred"], r["y_prob"])
        rows.append({"model": name, **r["metrics"], "threshold": r["threshold"],
                     "tuning_seconds": r["seconds"], "params": json.dumps(r["params"])})
        print(f"\n===== {name} =====")
        print(classification_report(y_test, r["y_pred"], target_names=["normal", "anomaly"],
                                    digits=4))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(config.RESULTS_DIR / "metrics.csv", index=False)

    cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    best = {c: metrics[c].max() for c in cols}
    lines = ["| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
             "|---|---:|---:|---:|---:|---:|"]
    for _, row in metrics.iterrows():
        cells = []
        for c in cols:
            text = f"{row[c]:.2%}" if c != "roc_auc" else f"{row[c]:.5f}"
            cells.append(f"**{text}**" if np.isclose(row[c], best[c]) else text)
        lines.append(f"| {row['model']} | " + " | ".join(cells) + " |")
    (config.RESULTS_DIR / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics


def save_models(results, nn_extras):
    for name, stem in config.MODEL_FILES.items():
        if name.startswith("Neural Network"):
            torch.save({
                "state_dict": results[name]["model"].state_dict(),
                "input_size": nn_extras["input_size"],
                "hidden1": nn_extras["best_params"]["hidden1"],
                "hidden2": nn_extras["best_params"]["hidden2"],
                "threshold": nn_extras["threshold"],
            }, config.MODEL_DIR / f"{stem}.pt")
            joblib.dump(nn_extras["prep"], config.MODEL_DIR / "nn_preprocessor.joblib",
                        compress=3)
        else:
            joblib.dump(results[name]["model"], config.MODEL_DIR / f"{stem}.joblib",
                        compress=3)
    print(f"\nModels saved to {config.MODEL_DIR}")


def main():
    for d in (config.RESULTS_DIR, config.FIG_DIR, config.MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    plots.apply_style()

    X, y = split_features_target(load_data())
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    print(f"Train {len(X_train):,} rows | Test {len(X_test):,} rows "
          f"(normal {(y_test == 0).sum():,}, anomaly {(y_test == 1).sum():,})")

    results = run_sklearn_models(X_train, y_train, X_test)
    nn_results, nn_extras = run_neural_networks(X_train, y_train, X_test)
    results.update(nn_results)

    metrics = write_metrics(results, y_test)
    nn_extras["grid"].to_csv(config.RESULTS_DIR / "nn_grid_search.csv", index=False)
    best_params = {name: {"params": results[name]["params"],
                          "threshold": results[name]["threshold"]}
                   for name in plots.MODEL_ORDER}
    (config.RESULTS_DIR / "best_params.json").write_text(json.dumps(best_params, indent=2))

    print("\nWriting figures to results/figures/")
    plots.plot_metric_comparison(metrics, config.FIG_DIR / "model_comparison.png")
    plots.plot_roc_curves(results, y_test, config.FIG_DIR / "roc_curves.png")
    plots.plot_confusion_matrices(results, y_test, config.FIG_DIR / "confusion_matrices.png")
    plots.plot_learning_curve(nn_extras["history"], config.FIG_DIR / "nn_learning_curve.png")
    plots.plot_threshold(nn_extras["y_val"], nn_extras["val_prob"], nn_extras["threshold"],
                         config.FIG_DIR / "nn_threshold_tuning.png")
    rf = results["Random Forest"]["model"]
    plots.plot_feature_importance(rf.named_steps["prep"].get_feature_names_out(),
                                  rf.named_steps["clf"].feature_importances_,
                                  config.FIG_DIR / "rf_feature_importance.png")
    dt = results["Decision Tree"]["model"]
    plots.plot_decision_tree(dt.named_steps["clf"], dt.named_steps["prep"].get_feature_names_out(),
                             config.FIG_DIR / "decision_tree.png")

    save_models(results, nn_extras)

    print("\nSummary (test set):")
    print(metrics[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
