from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE


def build():
    return DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE)
