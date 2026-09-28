from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_STATE


def build():
    return LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)
