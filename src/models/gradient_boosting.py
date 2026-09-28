from sklearn.ensemble import HistGradientBoostingClassifier

from src.config import RANDOM_STATE


def build():
    # No class_weight option here: train.py passes balanced sample weights instead.
    return HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.06,
        max_depth=6,
        random_state=RANDOM_STATE,
    )
