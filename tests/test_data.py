from src import config
from src.data import build_preprocessor, load_data, split


def test_feature_stage_sizes():
    assert len(config.SEMESTER_FEATURES) == 12
    assert len(config.ENROLLMENT_FEATURES) == 24
    assert len(config.FULL_FEATURES) == 36


def test_no_semester_column_leaks_into_enrollment_stage():
    assert not any(c.startswith("Curricular units") for c in config.ENROLLMENT_FEATURES)
    assert set(config.ENROLLMENT_FEATURES) | set(config.SEMESTER_FEATURES) == set(
        config.FULL_FEATURES
    )


def test_every_feature_is_either_numeric_or_categorical():
    num, cat = set(config.NUMERIC_FEATURES), set(config.CATEGORICAL_FEATURES)
    assert not num & cat
    assert num | cat == set(config.FULL_FEATURES)


def test_load_data_matches_config_and_keeps_labels():
    X, y = load_data()
    assert list(X.columns) == config.FULL_FEATURES
    assert len(X) == len(y) == 4424
    assert set(y) == set(config.CLASSES)
    assert y.value_counts().to_dict() == {"Graduate": 2209, "Dropout": 1421, "Enrolled": 794}


def test_split_is_stratified_80_20():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split(X, y)
    assert (len(X_train), len(X_test)) == (3539, 885)
    assert set(y_test) == set(config.CLASSES)
    assert abs(y_train.value_counts(normalize=True) - y.value_counts(normalize=True)).max() < 0.01


def test_preprocessor_uses_only_stage_columns():
    X, _ = load_data()
    cols = config.ENROLLMENT_FEATURES
    pre = build_preprocessor(cols).fit(X[cols])
    used = [c for _, _, group in pre.transformers_ for c in group]
    assert sorted(used) == sorted(cols)
