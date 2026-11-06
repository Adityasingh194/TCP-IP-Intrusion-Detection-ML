"""Exploratory data analysis: class balance, feature correlation and the PCA check.

Usage:
    python eda.py
"""
from src import config, plots
from src.data import build_preprocessor, load_data, split_features_target, train_test_split_data


def main():
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    plots.apply_style()

    df = load_data()
    X, y = split_features_target(df)
    print(f"Train_data.csv: {len(df):,} connections, {X.shape[1]} features")
    print(f"  normal  : {(y == 0).sum():,}")
    print(f"  anomaly : {(y == 1).sum():,}")
    print(f"  missing values: {df.isna().sum().sum()}, duplicate rows: {df.duplicated().sum()}")

    constant = [c for c in X.columns if X[c].nunique() == 1]
    print(f"  constant columns (no information): {constant}")

    print("\nWriting figures to results/figures/")
    plots.plot_class_distribution(y, config.FIG_DIR / "class_distribution.png")

    numeric = X.drop(columns=config.CATEGORICAL_COLS + constant)
    plots.plot_correlation(numeric, config.FIG_DIR / "correlation_heatmap.png")

    # PCA is fitted on the training split only, after one-hot + scaling.
    X_train, _, y_train, _ = train_test_split_data(X, y)
    X_scaled = build_preprocessor(scale=True).fit_transform(X_train)
    n90 = plots.plot_pca_variance(X_scaled, config.FIG_DIR / "pca_variance.png")
    plots.plot_pca_scatter(X_scaled, y_train, config.FIG_DIR / "pca_scatter.png")
    print(f"\nPCA: {n90} of {X_scaled.shape[1]} components are needed to keep 90% of the "
          "variance, so PCA gives little compression and is not used for the final models.")


if __name__ == "__main__":
    main()
