"""Matplotlib figures shared by the report script and the Streamlit app."""

import matplotlib.pyplot as plt
import numpy as np

from src.config import CLASSES
from src.models import CLASS_COLORS, STAGE_COLORS

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "font.size": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
    }
)


def class_distribution(y):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    counts = y.value_counts().reindex(CLASSES)
    bars = ax.bar(CLASSES, counts.values, color=[CLASS_COLORS[c] for c in CLASSES], width=0.55)
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + counts.max() * 0.02,
            f"{count:,}\n({count / len(y):.1%})",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )
    ax.set_ylabel("Number of Students")
    ax.set_title(f"Target Outcome Distribution (Total N = {len(y):,})", fontsize=12, pad=12)
    ax.set_ylim(0, counts.max() * 1.22)
    fig.tight_layout()
    return fig


def eda_boxplots(X, y):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    configs = [
        (axes[0], "Admission grade", "Admission Grade by Outcome"),
        (axes[1], "Curricular units 1st sem (approved)", "1st-Sem Approved Units by Outcome"),
    ]
    for ax, col, title in configs:
        data = [X.loc[y == c, col].values for c in CLASSES]
        bp = ax.boxplot(data, tick_labels=CLASSES, patch_artist=True, showfliers=False, widths=0.45)
        for patch, c in zip(bp["boxes"], CLASSES):
            patch.set_facecolor(CLASS_COLORS[c])
            patch.set_alpha(0.75)
            patch.set_edgecolor("#333333")
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.4)
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_ylabel(col, fontsize=9.5)
    fig.tight_layout()
    return fig


def model_comparison(metrics):
    """Grouped bars of test-set macro-F1 per model, one colour per stage."""
    stages = metrics["stages"]
    model_names = list(next(iter(stages.values()))["models"])
    x = np.arange(len(model_names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for i, (stage, s) in enumerate(stages.items()):
        values = [s["models"][m]["macro_f1"] for m in model_names]
        bars = ax.bar(
            x + (i - 0.5) * width,
            values,
            width,
            label=s["label"],
            color=STAGE_COLORS[stage],
            alpha=0.88,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.012,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="semibold",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=9.5)
    ax.set_ylabel("Macro-F1 (Holdout Test Set)")
    ax.set_title("Predictive Performance: Enrollment-Only vs. Full Stage", fontsize=12, pad=12)
    ax.set_ylim(0, 0.90)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def confusion_matrix(cm, title):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = range(len(CLASSES))
    ax.set_xticks(ticks, CLASSES, fontsize=9)
    ax.set_yticks(ticks, CLASSES, fontsize=9)
    ax.set_xlabel("Predicted Outcome", labelpad=8)
    ax.set_ylabel("Actual Outcome", labelpad=8)
    ax.set_title(title, fontsize=11, pad=10)
    for i in ticks:
        for j in ticks:
            color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=10.5)
    fig.tight_layout()
    return fig


def feature_importance(importance, title, color, top=12):
    """Horizontal bars of the `top` most important features (dict or Series)."""
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:top][::-1]
    names, values = zip(*items)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bars = ax.barh(names, values, color=color, alpha=0.88, height=0.62)
    for bar, val in zip(bars, values):
        ax.text(
            val + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#222222",
        )
    ax.set_title(title, fontsize=11.5, pad=10)
    ax.set_xlabel("Aggregated Relative Importance (Gini)")
    ax.set_xlim(0, max(values) * 1.15)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    return fig


def probabilities(probs):
    """Horizontal bars of predicted class probabilities ({class: probability})."""
    fig, ax = plt.subplots(figsize=(6.0, 2.2))
    classes = CLASSES[::-1]
    values = [probs[c] for c in classes]
    ax.barh(classes, values, color=[CLASS_COLORS[c] for c in classes], height=0.6)
    for i, val in enumerate(values):
        ax.text(val + 0.01, i, f"{val:.0%}", va="center", fontsize=9.5)
    ax.set_xlim(0, 1.1)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    fig.tight_layout()
    return fig
