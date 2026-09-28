# CLAUDE.md

Student ML course project: 3-class prediction (Dropout / Enrolled / Graduate) on the UCI
student dropout dataset. Every model is trained on two feature stages: enrollment-only (24)
and full (36, adding the 12 `Curricular units ...` semester columns). The headline metric is
test-set macro-F1.

## Commands

```bash
.venv/bin/python -m src.train                          # train + write artifacts/ (~40 s)
.venv/bin/python scripts/generate_report_assets.py     # figures -> assets/
.venv/bin/streamlit run app.py                         # dashboard + predictor
.venv/bin/pytest                                       # tests (fast)
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

## Ground rules

- **Do not change the results.** The expected test-set scores are in the README table.
  After any change to `src/`, rerun `python -m src.train` and check the numbers match.
- scikit-learn is pinned to 1.3.2 on purpose; other versions shift Random Forest scores.
- The feature order in `config.ALL_FEATURES` matches the CSV and must stay that way
  (Random Forest results depend on column order).
- Feature stages, seed, paths and the numeric/categorical split live only in `src/config.py`.
- Preprocessing is fit on the training split only, inside each pipeline.
- Keep it simple: this is a course project. Allowed dependencies are scikit-learn, pandas,
  numpy, matplotlib, joblib and streamlit (plus pytest/ruff for dev).

## Layout

- `src/data.py`: load, split, `build_preprocessor(cols)`
- `src/models/`: one `build()` per model; the registry and chart colours are in `__init__.py`.
  Gradient Boosting gets balanced sample weights (`USES_SAMPLE_WEIGHTS`).
- `src/train.py`: `train()` returns a metrics dict and saves
  `artifacts/{stage}/{model}.joblib` plus `artifacts/metrics.json`
- `src/plots.py`: all figures (used by `scripts/` and `app.py`)
- `src/codebook.py`: integer code -> label maps from the UCI data dictionary (app form only)
- `artifacts/` is gitignored; `assets/` is committed.
