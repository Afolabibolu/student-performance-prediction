"""Regenerate the README figures into assets/ from artifacts/metrics.json.

Usage: python scripts/generate_report_assets.py
(Trains the models first if artifacts/ does not exist yet.)
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, plots  # noqa: E402
from src.data import load_data  # noqa: E402
from src.models import STAGE_COLORS  # noqa: E402
from src.train import load_metrics, train  # noqa: E402


def save(fig, name):
    path = config.ASSETS_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.relative_to(config.ROOT)}")


def main():
    metrics = load_metrics() if config.METRICS_PATH.exists() else train()
    X, y = load_data()
    config.ASSETS_DIR.mkdir(exist_ok=True)

    print("Writing figures:")
    save(plots.class_distribution(y), "fig_classdist.png")
    save(plots.eda_boxplots(X, y), "fig_eda.png")
    save(plots.model_comparison(metrics), "fig_modelcompare.png")

    for stage, s in metrics["stages"].items():
        best = s["best_model"]
        stage_name = "Full Stage" if stage == "full" else "Enrollment-only"
        save(
            plots.confusion_matrix(
                s["models"][best]["confusion_matrix"], f"Confusion Matrix: {best} ({stage_name})"
            ),
            f"fig_cm_{stage}.png",
        )

    importance_titles = {
        "enrollment": "Top Predictors at Matriculation (Enrollment-only)",
        "full": "Top Predictors (Full Feature Set with Semesters)",
    }
    for stage, s in metrics["stages"].items():
        save(
            plots.feature_importance(
                s["feature_importance"], importance_titles[stage], STAGE_COLORS[stage]
            ),
            f"fig_imp_{stage}.png",
        )


if __name__ == "__main__":
    main()
