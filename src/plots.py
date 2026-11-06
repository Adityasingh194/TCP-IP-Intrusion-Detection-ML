"""All figures, drawn with one consistent matplotlib style."""
import matplotlib

matplotlib.use("Agg")  # write PNGs without needing a display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import PercentFormatter, StrMethodFormatter  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_curve  # noqa: E402
from sklearn.tree import plot_tree  # noqa: E402

# ---- palette (CVD-checked categorical order; see README) -------------------
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
AXIS = "#c9c8c2"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

CLASS_COLORS = {"Normal": SERIES[0], "Anomaly": SERIES[1]}

# Original models first, then the two added in this repo.
MODEL_ORDER = [
    "Logistic Regression",
    "Neural Network (baseline)",
    "Neural Network (tuned)",
    "Decision Tree",
    "Random Forest",
]
MODEL_COLORS = dict(zip(MODEL_ORDER, SERIES))

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#eef4fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                 "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec",
                     "#f2a7a6", "#e34948", "#a8302f"])


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelcolor": TEXT_2,
        "axes.labelsize": 10,
        "text.color": TEXT,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path.name}")


# ---- EDA -------------------------------------------------------------------
def plot_class_distribution(y, path):
    counts = [int((y == 0).sum()), int((y == 1).sum())]
    labels = ["Normal", "Anomaly"]
    total = sum(counts)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    bars = ax.bar(labels, counts, width=0.55,
                  color=[CLASS_COLORS[l] for l in labels],
                  edgecolor=SURFACE, linewidth=2)
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, n + total * 0.01,
                f"{n:,}  ({n / total:.1%})", ha="center", va="bottom",
                fontsize=9, color=TEXT)
    ax.set_ylabel("Connections")
    ax.set_ylim(0, max(counts) * 1.15)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="x", visible=False)
    ax.set_title("Class balance in Train_data.csv")
    _save(fig, path)


def plot_correlation(numeric_df, path):
    corr = numeric_df.corr()
    n = len(corr)
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(n), corr.columns, fontsize=7)
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.set_label("Pearson correlation", color=TEXT_2)
    ax.set_title("Correlation between numeric features")
    _save(fig, path)


def plot_pca_variance(X_scaled, path):
    """Cumulative variance of all principal components. Returns #components for 90%."""
    pca = PCA().fit(X_scaled)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = np.arange(1, len(cum) + 1)
    n90 = int(np.argmax(cum >= 0.90) + 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(k, cum, color=SERIES[0])
    ax.axhline(0.90, color=TEXT_2, linewidth=1, linestyle="--")
    ax.text(len(cum), 0.905, "90% of variance", ha="right", va="bottom",
            fontsize=9, color=TEXT_2)

    ax.plot([10], [cum[9]], "o", color=SERIES[0], markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.annotate(f"10 components keep only {cum[9]:.0%}", (10, cum[9]),
                xytext=(14, cum[9] - 0.12), fontsize=9, color=TEXT,
                arrowprops=dict(arrowstyle="-", color=TEXT_2, lw=0.8))
    ax.plot([n90], [cum[n90 - 1]], "o", color=SERIES[0], markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.annotate(f"{n90} components needed for 90%", (n90, cum[n90 - 1]),
                xytext=(n90 + 4, cum[n90 - 1] - 0.18), fontsize=9, color=TEXT,
                arrowprops=dict(arrowstyle="-", color=TEXT_2, lw=0.8))

    ax.set_xlabel(f"Number of principal components (of {len(cum)})")
    ax.set_ylabel("Cumulative explained variance")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, len(cum) + 1)
    ax.set_title("PCA: variance is spread across many components")
    _save(fig, path)
    return n90


def plot_pca_scatter(X_scaled, y, path):
    pcs = PCA(n_components=2).fit_transform(X_scaled)
    y = np.asarray(y)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for value, label in [(0, "Normal"), (1, "Anomaly")]:
        mask = y == value
        ax.scatter(pcs[mask, 0], pcs[mask, 1], s=5, alpha=0.35, linewidths=0,
                   color=CLASS_COLORS[label], label=label)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    leg = ax.legend(markerscale=3, loc="upper right")
    for handle in leg.legend_handles:
        handle.set_alpha(1)
    ax.set_title("Training connections on the first two principal components")
    _save(fig, path)


# ---- model results ---------------------------------------------------------
def plot_confusion_matrices(results, y_true, path):
    names = [n for n in MODEL_ORDER if n in results]
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.9))
    for ax, name in zip(np.atleast_1d(axes), names):
        cm = confusion_matrix(y_true, results[name]["y_pred"])
        row_pct = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(row_pct, cmap=SEQUENTIAL, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                dark = row_pct[i, j] > 0.5
                ax.text(j, i, f"{cm[i, j]:,}\n{row_pct[i, j]:.2%}", ha="center",
                        va="center", fontsize=9,
                        color="white" if dark else TEXT)
        ax.set_xticks([0, 1], ["Normal", "Anomaly"])
        ax.set_yticks([0, 1], ["Normal", "Anomaly"])
        ax.set_xlabel("Predicted")
        ax.grid(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(name, fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("Actual")
    fig.suptitle("Confusion matrices on the test set (count and % of actual class)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def plot_roc_curves(results, y_true, path):
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12, 5))
    for name in [n for n in MODEL_ORDER if n in results]:
        fpr, tpr, _ = roc_curve(y_true, results[name]["y_prob"])
        auc = results[name]["metrics"]["roc_auc"]
        for ax in (ax_full, ax_zoom):
            ax.plot(fpr, tpr, color=MODEL_COLORS[name], label=f"{name}  (AUC {auc:.5f})")
    ax_full.plot([0, 1], [0, 1], color=TEXT_2, linewidth=1, linestyle="--")
    ax_full.text(0.62, 0.55, "random guess", rotation=38, fontsize=8, color=TEXT_2)
    ax_full.set_xlim(0, 1)
    ax_full.set_ylim(0, 1.01)
    ax_full.set_title("ROC curves")
    ax_full.legend(loc="lower right")

    ax_zoom.set_xlim(0, 0.06)
    ax_zoom.set_ylim(0.90, 1.002)
    ax_zoom.set_title("Zoomed into the top-left corner")
    for ax in (ax_full, ax_zoom):
        ax.set_xlabel("False positive rate (normal traffic flagged)")
        ax.set_ylabel("True positive rate (attacks caught)")
    fig.tight_layout()
    _save(fig, path)


def plot_metric_comparison(metrics_df, path):
    metric_cols = [("accuracy", "Accuracy"), ("precision", "Precision"),
                   ("recall", "Recall"), ("f1", "F1 score")]
    df = metrics_df.set_index("model").loc[[n for n in MODEL_ORDER
                                            if n in set(metrics_df["model"])]]
    lo = np.floor(df[[c for c, _ in metric_cols]].values.min() * 100 - 0.5) / 100
    ypos = np.arange(len(df))[::-1]

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, (col, title) in zip(axes, metric_cols):
        for y, (name, row) in zip(ypos, df.iterrows()):
            ax.plot(row[col], y, "o", markersize=9, color=MODEL_COLORS[name],
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.text(row[col], y + 0.22, f"{row[col]:.2%}", ha="center",
                    va="bottom", fontsize=8.5, color=TEXT)
        ax.set_xlim(lo, 1.0 + (1 - lo) * 0.06)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=10)
    axes[0].set_yticks(ypos, df.index)
    axes[0].set_ylim(-0.6, len(df) - 0.4)
    fig.suptitle("Test-set performance (5,039 held-out connections)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def plot_learning_curve(history, path):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for key, label, color in [("train_loss", "Train", SERIES[0]),
                              ("val_loss", "Validation", SERIES[1])]:
        ax.plot(epochs, history[key], color=color, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1, label=label)
        ax.text(epochs[-1] + 0.2, history[key][-1], label, va="center",
                fontsize=9, color=TEXT)
    ax.set_xlim(0.5, epochs[-1] + 1.5)
    ax.set_xticks(epochs)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary cross-entropy loss")
    ax.legend(loc="upper right")
    ax.set_title("Tuned neural network: learning curve")
    _save(fig, path)


def plot_threshold(y_true, y_prob, best_threshold, path):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    keep = np.isfinite(thr) & (thr <= 1)
    fpr, tpr, thr = fpr[keep], tpr[keep], thr[keep]
    i = int(np.argmin(np.abs(thr - best_threshold)))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(thr, tpr, color=SERIES[0], label="True positive rate")
    ax.plot(thr, fpr, color=SERIES[1], label="False positive rate")
    ax.axvline(best_threshold, color=TEXT_2, linewidth=1, linestyle="--")
    ax.text(best_threshold + 0.02, 0.5,
            f"best threshold = {best_threshold:.3f}\nTPR {tpr[i]:.1%}, FPR {fpr[i]:.1%}",
            fontsize=9, color=TEXT, va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Decision threshold on P(anomaly)")
    ax.legend(loc="center left")
    ax.set_title("Threshold tuning on the validation set (Youden's J)")
    _save(fig, path)


def plot_feature_importance(feature_names, importances, path, top=15):
    order = np.argsort(importances)[::-1][:top][::-1]
    names = np.asarray(feature_names)[order]
    values = np.asarray(importances)[order]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh(names, values, height=0.7, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    for y, v in enumerate(values):
        ax.text(v + values.max() * 0.01, y, f"{v:.3f}", va="center", fontsize=8.5,
                color=TEXT)
    ax.set_xlim(0, values.max() * 1.15)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title(f"Random forest: top {top} features")
    _save(fig, path)


def plot_decision_tree(tree, feature_names, path, max_depth=3):
    fig, ax = plt.subplots(figsize=(30, 10))
    plot_tree(tree, max_depth=max_depth, feature_names=list(feature_names),
              class_names=["Normal", "Anomaly"], filled=False, rounded=True,
              impurity=False, proportion=True, fontsize=9, ax=ax)
    ax.set_title(f"Decision tree: first {max_depth} levels "
                 f"(full depth {tree.get_depth()}, {tree.get_n_leaves()} leaves)")
    _save(fig, path)
