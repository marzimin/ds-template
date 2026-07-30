# The ML pipeline

Everything about data, models, and metrics. You should not need to edit Python
to change any of it — `cfg/config.yaml` is the interface.

For how this fits with the API and web interface, see
[`architecture.md`](architecture.md).

---

## The three steps

`make pipeline` runs them in order; each also runs alone with a flag.

| Step | Flag | What it does |
| --- | --- | --- |
| **Prepare** | `--prepare-data` | Reads `data/raw/<input_file>`, applies your transformations, writes `data/processed/<name>_prepared.csv` |
| **EDA** | `--eda` | Plots every column and its relationship to the target; logs them to MLflow |
| **Train** | `--train-model` | Splits, fits, evaluates, plots, and registers the model |

They communicate through files, so you can rerun one without repeating the
others.

**Your transformations go in `PrepareDataPipeline`** (`backend/src/ml/prepare_data.py`).
It ships with four empty placeholder steps chained together — replace them.

---

## What kind of problem is this?

The template supports **binary classification, multiclass classification, and
regression**, and infers which from your target column. It logs the decision
every run:

```text
Task inferred as regression from the target (214 distinct values, dtype float64).
Set `task:` in cfg/config.yaml to override.
```

That one decision drives everything downstream:

| Task | Metrics | Evaluation plots |
| --- | --- | --- |
| Binary classification | accuracy, precision, recall, f1 | confusion matrix, classification report, ROC curve, precision-recall curve |
| Multiclass classification | accuracy, macro precision / recall / f1 | confusion matrix, classification report |
| Regression | RMSE, MAE, R² | predicted-vs-actual, residuals |

### How inference works

- Boolean, text, or categorical target → **classes**
- Numeric with fractional values → **measurements**
- Numeric, whole-numbered, ≤ 20 distinct values → **classes**
- Numeric, whole-numbered, > 20 distinct values → **measurements**

This is a heuristic and will occasionally be wrong — integer counts you want to
regress onto look exactly like class labels. Override it:

```yaml
task: regression       # or: classification, or: auto (the default)
target_values: null    # a continuous target has no fixed value set
```

Setting `task` explicitly never hurts. If you already know what you are
modelling, saying so is clearer than relying on a guess.

### Two deliberate omissions

**Multiclass gets no ROC or precision-recall curve.** Those are defined for two
classes. Producing one for three would quietly give a one-vs-rest curve against
an arbitrary class — a plot that looks entirely reasonable and answers a
question you did not ask. A missing plot beats a misleading one.

**Macro averaging for multiclass** weights every class equally, so a large class
cannot mask poor performance on a small one.

---

## Choosing a model

Models are declared in config, not code. `model_registry` maps a short name to a
fully qualified class path, imported dynamically at training time:

```yaml
model_registry:
  # Classifiers
  xgb_classifier: "xgboost.XGBClassifier"
  rf_classifier: "sklearn.ensemble.RandomForestClassifier"
  logistic_regression: "sklearn.linear_model.LogisticRegression"
  # Regressors
  xgb_regressor: "xgboost.XGBRegressor"
  rf_regressor: "sklearn.ensemble.RandomForestRegressor"
  linear_regression: "sklearn.linear_model.LinearRegression"

model_name: "xgb_classifier"
```

Add a line and point `model_name` at it. Any class implementing the
scikit-learn `fit`/`predict` API works — LightGBM, CatBoost, your own.

The correct MLflow flavour is selected automatically from the root module of the
import path, which matters because `mlflow.sklearn` cannot serialise non-sklearn
estimators such as XGBoost.

**Use a classifier for labels and a regressor for continuous targets.** Getting
this backwards is caught before training with an error naming both — without
that check, a regressor fits class labels happily and only fails later inside
the metrics, with a message that never mentions the real mistake.

### Hyperparameters

`model_params` accepts two shapes. **Nested per model** lets settings for
several models coexist, so switching is a one-word change to `model_name`:

```yaml
model_params:
  xgb_classifier:
    n_estimators: 50
    max_depth: 10
  rf_regressor:
    n_estimators: 200
```

**Flat** is simpler and applies to whichever model is chosen, but then every
model must accept the same arguments:

```yaml
model_params:
  n_estimators: 50
```

Keys must be real constructor arguments for that estimator — `LinearRegression`
has no `n_estimators`.

---

## Using your own data

1. Put a CSV in `data/raw/`.
2. Set `data.input_file`, `target_column`, `target_values`, and `model_name`.

Column names are normalised on read — uppercased, non-alphanumerics replaced
with underscores — so `mean radius` becomes `MEAN_RADIUS`. `target_column` is
matched *after* normalisation.

The pipeline enforces two things it will not do for you:

- **Numeric features only.** Encode or drop categoricals in `PrepareDataPipeline`.
- **No missing values by training time.** Impute or drop them in the same place.

Both raise an error naming the offending columns. Encoding and imputation change
what a model learns, so the template refuses to guess.

### The bundled datasets

Three ship in `data/raw/`, one per task type. All are bundled with
scikit-learn, so `make sample-data` needs no network access.

| File | Task | Shape | Config |
| --- | --- | --- | --- |
| `breast_cancer.csv` | binary | 569 × 31 | `target_values: [0, 1]`, `model_name: "xgb_classifier"` |
| `iris.csv` | multiclass | 150 × 5 | `target_values: [0, 1, 2]`, `model_name: "rf_classifier"` |
| `diabetes.csv` | regression | 442 × 11 | `target_values: null`, `model_name: "rf_regressor"` |

Only the file named by `data.input_file` is read; the others sit unused. Each
names its target column `target`, so one `target_column` setting covers all
three. The same presets are repeated as a copy-paste block in `cfg/config.yaml`.

---

## Schemas

`backend/src/schemas.py` validates data whenever the pipeline reads or writes a
CSV, so a corrupted intermediate file is caught at the boundary rather than
three steps later.

Schemas validate in **non-strict** mode: only declared columns are checked, and
anything else passes through.

**Out of the box only the target is checked**, which is what lets any dataset run
without editing the file. Naming specific feature columns would make the
template fail on the first run with anyone else's data.

Add checks for columns whose silent corruption would ruin a model — a negative
age, a probability above 1:

```python
FEATURE_COLUMNS = {
    "MEAN_RADIUS": Column(float, checks=Check.ge(0)),
    "AGE": Column(int, checks=Check.in_range(0, 120)),
}
```

The target's permitted values come from `target_values`. Leave it `null` for
regression — a continuous target has no fixed value set, and checking one would
reject every valid row.

---

## Adding a metric

`compute_metrics` in `backend/src/ml/task.py` returns a plain dictionary, and
everything downstream follows it:

```python
metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
```

That one line makes it appear in MLflow *and* as a column in the web dashboard.
No other file changes.

Values must be plain `float` — MLflow rejects numpy scalars, which is why every
metric is wrapped in `float()`.

---

## Experiment tracking

Every run logs to MLflow: metrics, parameters, every plot, and the trained model
registered as a new version.

Experiment and registered model names default to `[project].name` in
`backend/pyproject.toml`, so renaming the project renames them too. Override via
the `tracking` block in `cfg/config.yaml`.

The **train/test split** is controlled by `test_size`, `random_state` (fixed for
reproducibility), and `stratify`. Stratification preserves class balance and is
recommended for imbalanced classification; it is ignored for regression, where
there are no classes to balance.

---

## What each step does differently per task

Useful when reading the code or extending it:

| Step | Task-dependent? | Detail |
| --- | --- | --- |
| `prepare_data.py` | **No** | Same path for every task. Its schema check skips target-value validation when `target_values` is null. |
| `eda.py` | **One plot** | The target gets a class-balance bar chart or a histogram — counting occurrences of a continuous target would draw one bar per row. |
| `train_model.py` | **Yes** | Metrics, evaluation plots, the estimator-family check, and whether `stratify` applies. |
