"""Model registry: display name -> builder, plus shared chart colours."""

from src.models import decision_tree, gradient_boosting, logistic_regression, random_forest

MODELS = {
    "Logistic Regression": logistic_regression.build,
    "Decision Tree": decision_tree.build,
    "Random Forest": random_forest.build,
    "Gradient Boosting": gradient_boosting.build,
}

# Models trained with balanced sample weights instead of a class_weight setting.
USES_SAMPLE_WEIGHTS = {"Gradient Boosting"}

CLASS_COLORS = {"Dropout": "#C1121F", "Enrolled": "#9A8C98", "Graduate": "#3A7D44"}
STAGE_COLORS = {"enrollment": "#3A7D44", "full": "#2E4B7A"}


def slug(model_name):
    """File-name form of a model name, e.g. 'Random Forest' -> 'random_forest'."""
    return model_name.lower().replace(" ", "_")
