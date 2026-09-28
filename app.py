"""Streamlit app: model comparison dashboard and a live outcome predictor.

Usage: streamlit run app.py
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import codebook, config, plots
from src.data import load_data
from src.models import MODELS, STAGE_COLORS, slug
from src.train import load_metrics, results_table, train

st.set_page_config(page_title="Student Outcome Prediction", layout="wide")

# Enrollment inputs shown up front, grouped into cards. These are the most
# influential enrollment-time features (Random Forest importance); every other
# enrollment feature goes under "Advanced".
INPUT_GROUPS = {
    "Programme and admission": [
        "Course",
        "Application mode",
        "Admission grade",
        "Previous qualification (grade)",
        "Age at enrollment",
    ],
    "Family background": [
        "Mother's qualification",
        "Father's qualification",
        "Mother's occupation",
        "Father's occupation",
    ],
    "Finances": ["Tuition fees up to date", "Scholarship holder"],
}
KEY_FEATURES = [f for group in INPUT_GROUPS.values() for f in group]

# Streamlit markdown colour names matching the class colours in the charts.
CLASS_TEXT_COLORS = {"Dropout": "red", "Enrolled": "gray", "Graduate": "green"}

PAGES = ["Model comparison", "Live predictor"]


def model_path(stage, name):
    return config.ARTIFACTS_DIR / stage / f"{slug(name)}.joblib"


@st.cache_resource(show_spinner=False)
def get_metrics():
    """Load metrics, training every model first if artifacts/ is incomplete."""
    have_models = all(model_path(s, m).exists() for s in config.STAGES for m in MODELS)
    if not (config.METRICS_PATH.exists() and have_models):
        with st.spinner("No trained models found. Training them now (under a minute)..."):
            return train(verbose=False)
    return load_metrics()


@st.cache_resource(show_spinner=False)
def get_model(stage, name):
    return joblib.load(model_path(stage, name))


@st.cache_data(show_spinner=False)
def get_data():
    return load_data()


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------- sidebar


def sidebar(metrics):
    with st.sidebar:
        st.markdown("## 🎓 Student outcomes")
        st.caption("Dropout · Enrolled · Graduate")
        page = st.radio("Go to", PAGES, label_visibility="collapsed")

        st.divider()
        counts = metrics["class_counts"]
        total = sum(counts.values())
        st.markdown(f"**Dataset:** {total:,} students, UCI")
        for cls in config.CLASSES:
            color = CLASS_TEXT_COLORS[cls]
            st.markdown(f":{color}[●] {cls}: **{counts[cls]:,}** ({counts[cls] / total:.0%})")
        show(plots.class_distribution(get_data()[1]))
        st.caption(
            f"{metrics['n_train']:,} students for training, {metrics['n_test']:,} held out "
            "for testing (stratified 80/20 split, seed 42)."
        )
    return page


# ---------------------------------------------------------------- dashboard


def dashboard(metrics):
    stages = metrics["stages"]
    early, full = stages["enrollment"], stages["full"]
    early_best, full_best = early["best_model"], full["best_model"]
    rf = full["models"]["Random Forest"]
    early_f1 = early["models"][early_best]["macro_f1"]
    full_f1 = full["models"][full_best]["macro_f1"]

    st.title("How well can we predict a student's outcome?")
    st.caption(
        f"Each model is trained twice: on the **{len(config.ENROLLMENT_FEATURES)} features known "
        f"at enrollment**, and on all **{len(config.FULL_FEATURES)} features** including first- "
        "and second-semester results. All scores are on the held-out test set."
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        f"At enrollment · {early_best}",
        f"{early_f1:.3f}",
        "macro-F1",
        delta_color="off",
        border=True,
    )
    k2.metric(
        f"After 2 semesters · {full_best}",
        f"{full_f1:.3f}",
        "macro-F1",
        delta_color="off",
        border=True,
    )
    k3.metric(
        "Gain from semester data",
        f"+{full_f1 - early_f1:.3f}",
        "macro-F1, best vs best",
        delta_color="off",
        border=True,
    )
    k4.metric(
        "Highest accuracy · Random Forest",
        f"{rf['accuracy']:.3f}",
        "full stage",
        delta_color="off",
        border=True,
    )

    with st.container(border=True):
        st.markdown(
            f"**Why macro-F1 is the headline.** The classes are uneven, and macro-F1 weights "
            f"each class equally, so the small Enrolled class counts as much as the others. "
            f"Random Forest reaches the highest full-set accuracy ({rf['accuracy']:.3f}) but a "
            f"lower macro-F1 ({rf['macro_f1']:.3f}) than {full_best} ({full_f1:.3f}), because it "
            f"does poorly on Enrolled students: it recalls only "
            f"{rf['per_class']['Enrolled']['recall']:.0%} of them."
        )

    scores_tab, errors_tab, drivers_tab = st.tabs(
        ["Scoreboard", "Where models go wrong", "What drives predictions"]
    )

    with scores_tab:
        chart_col, table_col = st.columns([3, 2], gap="large")
        with chart_col:
            show(plots.model_comparison(metrics))
        with table_col:
            table = results_table(metrics).pivot(index="Model", columns="Stage")
            table = table.reindex(list(MODELS)).swaplevel(axis=1)
            table = table[
                [
                    (s["label"], metric)
                    for s in stages.values()
                    for metric in ("Accuracy", "Macro-F1")
                ]
            ]
            styled = table.style.format("{:.3f}").highlight_max(
                props="background-color: rgba(58, 125, 68, 0.22)"
            )
            st.dataframe(styled, width="stretch")
            st.caption("Shaded: best model in each column.")

    with errors_tab:
        model = (
            st.segmented_control("Model", list(MODELS), default=full_best, key="cm_model")
            or full_best
        )
        cols = st.columns(len(stages), gap="large")
        for col, s in zip(cols, stages.values()):
            m = s["models"][model]
            with col:
                st.markdown(f"#### {s['label']}")
                show(plots.confusion_matrix(m["confusion_matrix"], model))
                recall = " · ".join(
                    f"{cls} {m['per_class'][cls]['recall']:.0%}" for cls in config.CLASSES
                )
                st.caption(f"Recall by class: {recall}")

    with drivers_tab:
        labels = {stage: s["label"] for stage, s in stages.items()}
        stage = (
            st.segmented_control(
                "Feature set",
                list(labels),
                format_func=labels.get,
                default="enrollment",
                key="imp_stage",
            )
            or "enrollment"
        )
        st.caption(
            "Random Forest feature importance. One-hot columns are summed back into their "
            "original feature."
        )
        show(
            plots.feature_importance(
                stages[stage]["feature_importance"], labels[stage], STAGE_COLORS[stage]
            )
        )


# ---------------------------------------------------------------- predictor


def feature_input(feature, X):
    """A widget for one feature, defaulting to its most common value or median."""
    values = X[feature]
    if feature in config.CATEGORICAL_FEATURES:
        options = values.value_counts().index.tolist()  # most common first
        return st.selectbox(
            feature, options, index=0, format_func=lambda c: codebook.label(feature, c)
        )
    if pd.api.types.is_integer_dtype(values):
        return st.number_input(
            feature, int(values.min()), int(values.max()), int(values.median()), step=1
        )
    return st.number_input(
        feature, float(values.min()), float(values.max()), float(values.median()), step=0.1
    )


def input_card(title, features, X, values, ncols=2):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        cols = st.columns(ncols)
        for i, f in enumerate(features):
            with cols[i % ncols]:
                values[f] = feature_input(f, X)


def predictor(metrics):
    st.title("Early-warning predictor")
    st.caption("Change any detail and the prediction on the right updates immediately.")

    X, _ = get_data()
    inputs_col, result_col = st.columns([3, 2], gap="large")

    with inputs_col:
        when = st.radio(
            "When are you predicting?",
            ["At enrollment", "After the first two semesters"],
            horizontal=True,
            help="After two semesters you can also enter the semester results, "
            "which makes the prediction much more reliable.",
        )
        stage = "full" if when.startswith("After") else "enrollment"
        s = metrics["stages"][stage]
        model_name = (
            st.segmented_control(
                "Model",
                list(MODELS),
                default=s["best_model"],
                key=f"model_{stage}",
                help="Defaults to the best model on macro-F1 for this stage.",
            )
            or s["best_model"]
        )

        values = {}
        for title, features in INPUT_GROUPS.items():
            input_card(title, features, X, values)

        with st.expander("Advanced: other enrollment details"):
            rest = [f for f in config.ENROLLMENT_FEATURES if f not in KEY_FEATURES]
            cols = st.columns(2)
            for i, f in enumerate(rest):
                with cols[i % 2]:
                    values[f] = feature_input(f, X)

        if stage == "full":
            with st.container(border=True):
                st.markdown("**Semester results**")
                sem1, sem2 = st.columns(2)
                sem1.caption("1st semester")
                sem2.caption("2nd semester")
                for f in config.SEMESTER_FEATURES:
                    with sem1 if "1st sem" in f else sem2:
                        values[f] = feature_input(f, X)

    row = pd.DataFrame([values])[config.STAGES[stage]]
    pipe = get_model(stage, model_name)
    probs = dict(zip(pipe.classes_, pipe.predict_proba(row)[0]))
    predicted = max(probs, key=probs.get)
    m = s["models"][model_name]

    with result_col, st.container(border=True):
        st.caption("PREDICTED OUTCOME")
        st.markdown(f"# :{CLASS_TEXT_COLORS[predicted]}[{predicted}]")
        st.markdown(f"{probs[predicted]:.0%} probability")
        st.divider()
        for cls in config.CLASSES:
            st.markdown(f":{CLASS_TEXT_COLORS[cls]}[●] {cls}: **{probs[cls]:.0%}**")
            st.progress(float(probs[cls]))
        st.divider()
        st.caption(
            f"{s['label']} · {model_name}: test accuracy {m['accuracy']:.3f}, "
            f"macro-F1 {m['macro_f1']:.3f}."
        )
        if stage == "enrollment":
            st.caption(
                "Early predictions are much less certain than end-of-year ones; treat this "
                "as a flag for follow-up, not a verdict."
            )


# ---------------------------------------------------------------- page

metrics = get_metrics()
page = sidebar(metrics)
if page == PAGES[0]:
    dashboard(metrics)
else:
    predictor(metrics)
