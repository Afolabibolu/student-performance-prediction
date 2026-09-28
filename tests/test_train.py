from src import config
from src.data import load_data, split
from src.train import fit_pipeline, train


def test_training_smoke():
    """Fast run: one quick model per stage, nothing written to artifacts/."""
    metrics = train(models=["Decision Tree"], save=False, verbose=False)

    assert set(metrics["stages"]) == {"enrollment", "full"}
    for stage in metrics["stages"].values():
        scores = stage["models"]["Decision Tree"]
        assert set(scores["per_class"]) == set(config.CLASSES)
        assert len(scores["confusion_matrix"]) == len(config.CLASSES)
        assert 0 < scores["accuracy"] < 1 and 0 < scores["macro_f1"] < 1


def test_pipeline_predicts_the_three_classes():
    X, y = load_data()
    X_train, X_test, y_train, _ = split(X, y)
    cols = config.ENROLLMENT_FEATURES
    pipe = fit_pipeline("Decision Tree", cols, X_train, y_train)

    assert list(pipe.classes_) == config.CLASSES
    probs = pipe.predict_proba(X_test[cols].head(5))
    assert probs.shape == (5, 3)
    assert set(pipe.predict(X_test[cols])) <= set(config.CLASSES)
