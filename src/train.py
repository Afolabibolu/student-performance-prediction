"""Train every model on both feature stages and save pipelines + metrics.

Usage: python -m src.train
"""

import json

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from src import config
from src.data import build_preprocessor, load_data, split
from src.metrics import evaluate
from src.models import MODELS, USES_SAMPLE_WEIGHTS, slug


def fit_pipeline(model_name, cols, X_train, y_train):
    pipe = Pipeline([("pre", build_preprocessor(cols)), ("clf", MODELS[model_name]())])
    if model_name in USES_SAMPLE_WEIGHTS:
        weights = compute_sample_weight("balanced", y_train)
        pipe.fit(X_train[cols], y_train, clf__sample_weight=weights)
    else:
        pipe.fit(X_train[cols], y_train)
    return pipe


def feature_importance(pipe):
    """Tree importances with each one-hot column summed back into its original feature."""
    pre = pipe.named_steps["pre"]
    numeric = pre.transformers_[0][2]
    categorical = pre.transformers_[1][2]
    ohe = pre.named_transformers_["cat"]
    parents = list(numeric) + [col for col, cats in zip(categorical, ohe.categories_) for _ in cats]
    imp = pd.Series(pipe.named_steps["clf"].feature_importances_, index=parents)
    return imp.groupby(level=0, sort=False).sum().sort_values(ascending=False)


def train(stages=None, models=None, save=True, verbose=True):
    """Fit, score and (optionally) save each stage x model. Returns the metrics dict."""
    stages = stages or list(config.STAGES)
    models = models or list(MODELS)

    X, y = load_data()
    X_train, X_test, y_train, y_test = split(X, y)

    metrics = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "class_counts": y.value_counts().reindex(config.CLASSES).to_dict(),
        "stages": {},
    }

    for stage in stages:
        cols = config.STAGES[stage]
        if verbose:
            print(f"\n{config.STAGE_LABELS[stage]} ({len(cols)} features)")
        stage_metrics = {"label": config.STAGE_LABELS[stage], "features": cols, "models": {}}

        for name in models:
            pipe = fit_pipeline(name, cols, X_train, y_train)
            scores = evaluate(y_test, pipe.predict(X_test[cols]))
            stage_metrics["models"][name] = scores
            if verbose:
                acc, f1 = scores["accuracy"], scores["macro_f1"]
                print(f"  {name:22s} accuracy {acc:.3f}   macro-F1 {f1:.3f}")

            if name == "Random Forest":
                stage_metrics["feature_importance"] = feature_importance(pipe).to_dict()
            if save:
                path = config.ARTIFACTS_DIR / stage / f"{slug(name)}.joblib"
                path.parent.mkdir(parents=True, exist_ok=True)
                # Compressed: the Random Forests shrink from ~90 MB to under 20 MB.
                joblib.dump(pipe, path, compress=3)

        stage_metrics["best_model"] = max(
            stage_metrics["models"], key=lambda m: stage_metrics["models"][m]["macro_f1"]
        )
        metrics["stages"][stage] = stage_metrics

    if save:
        config.ARTIFACTS_DIR.mkdir(exist_ok=True)
        with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        if verbose:
            print(f"\nSaved pipelines and metrics to {config.ARTIFACTS_DIR}")
    return metrics


def results_table(metrics):
    """One row per stage x model with accuracy and macro-F1."""
    rows = [
        {
            "Stage": s["label"],
            "Model": name,
            "Accuracy": m["accuracy"],
            "Macro-F1": m["macro_f1"],
        }
        for s in metrics["stages"].values()
        for name, m in s["models"].items()
    ]
    return pd.DataFrame(rows)


def load_metrics():
    with open(config.METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    train()
