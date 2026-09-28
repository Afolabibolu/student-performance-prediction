"""Project-wide settings: paths, seed, class names and the two feature stages."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "data.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
ASSETS_DIR = ROOT / "assets"

RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET = "Target"
CLASSES = ["Dropout", "Enrolled", "Graduate"]

# All 36 input features, in the order they appear in data.csv (after stripping
# whitespace from the headers). The order matters: it fixes the column order
# the models see, and Random Forest results depend on it.
ALL_FEATURES = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Previous qualification (grade)",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Admission grade",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "Age at enrollment",
    "International",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

# The 12 first- and second-semester measures, only known after enrollment.
SEMESTER_FEATURES = [c for c in ALL_FEATURES if c.startswith("Curricular units")]

# Feature stages: what is known on registration day vs. after two semesters.
ENROLLMENT_FEATURES = [c for c in ALL_FEATURES if c not in SEMESTER_FEATURES]
FULL_FEATURES = list(ALL_FEATURES)

STAGES = {
    "enrollment": ENROLLMENT_FEATURES,
    "full": FULL_FEATURES,
}
STAGE_LABELS = {
    "enrollment": "Enrollment-only",
    "full": "Full (with semester results)",
}

# Continuous features are standardised; every other feature is an integer code
# (or a 0/1 flag) and is one-hot encoded.
NUMERIC_FEATURES = [
    "Application order",
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    *SEMESTER_FEATURES,
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]
CATEGORICAL_FEATURES = [c for c in ALL_FEATURES if c not in NUMERIC_FEATURES]
