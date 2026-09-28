from sklearn.ensemble import RandomForestClassifier

from src.config import RANDOM_STATE


def build():
    return RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
