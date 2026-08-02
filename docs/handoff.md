# Hand-off from de-template

This project owns everything from a file on disk to a registered model.
[`de-template`](https://github.com/marzimin/de-template) owns everything
before it — extraction, loading, and the dbt transformations that end at the
`marts` schema. The seam is a file.

You do not need de-template to use this template. Read this when a colleague
runs the warehouse, or when you are setting up both halves yourself.

> de-template's [`docs/handoff.md`](https://github.com/marzimin/de-template/blob/main/docs/handoff.md)
> is the other side of this page. It covers what happens upstream: how the
> export is configured, how Postgres types survive the trip, and how to run the
> export as an Airflow task. This page covers what you have to do here.

---

## Why a file rather than a shared connection

A file is a snapshot. You can version it, hand it to a colleague, and reproduce
a model against it six months later. A live database connection makes every
training run depend on the warehouse being up, reachable, and unchanged — and
makes "the model got worse" impossible to distinguish from "someone edited a
staging model".

The cost is freshness, which is the right trade for modelling work.

---

## What arrives, and where

de-template writes exported marts into this project's `data/raw/`. Either it
does that directly — the two repositories checked out side by side, with
`DS_DATA_RAW_DIR` set over there to point at this project's `data/raw` — or
someone copies the file across. Nothing here has to be configured for it;
`data/raw/` is just a directory, and a CSV that appears in it is indistinguishable
from one you put there yourself.

```text
~/code/
├── de-template/            ← extraction, loading, dbt, exports
└── ds-template/            ← this project
```

Then name the file in [`cfg/config.yaml`](../cfg/config.yaml):

```yaml
data:
  raw_dir: "data/raw"
  input_file: "example_items.csv"

target_column: "ITEM_NAME_LENGTH"
target_values: null # or the class labels, for classification
model_name: "rf_regressor"
```

and run `make pipeline`.

---

## The one thing that will catch you: column case

**Write `target_column` in UPPER_SNAKE, even though the file's header is
lower_snake.**

The two projects normalise column names in opposite directions, and that is
fine — the round trip is lossless. But it means the name you configure is not
the name you see when you open the CSV:

| Stage | Convention | Example |
| --- | --- | --- |
| Source API | whatever it sends | `Item Name Length` |
| de-template's loader | `lower_snake` | `item_name_length` |
| The exported file | `lower_snake` | `item_name_length` |
| **This project, on read** | `UPPER_SNAKE` | `ITEM_NAME_LENGTH` |

Get it wrong and you get a "target column not found" error naming a column you
can plainly see in the file. See [`ml.md`](ml.md#using-your-own-data) for the
normalisation rule in full.

---

## What a mart has to look like before it can be trained on

This is the part that is easy to miss. **A mart that exports correctly is not
automatically a mart this project can train on.** The export is faithful — uuids,
timestamps, JSON blobs and binary columns all survive — and every one of those
is something the pipeline will refuse.

Two requirements, enforced at training time:

- **Numeric features only.** Identifiers, timestamps, JSON and text have to be
  cast, encoded, or dropped.
- **No missing values.** Impute or drop them.

Both raise an error naming the offending columns. Encoding and imputation change
what a model learns, so this template will not guess for you.

**Prefer to fix it in dbt, not here.** A `marts` model shaped for training —
numeric columns, a target, nothing else — makes the decision visible in SQL,
reviewable, and reusable. The alternative is burying it in
`PrepareDataPipeline` (`backend/src/ml/prepare_data.py`), where it is invisible
to everyone working upstream. Use the Python side for transformations that are
genuinely modelling decisions, not for cleaning up a mart that was never shaped
for its reader.

de-template ships `make demo-handoff`, which builds one deliberately awkward
mart and one training-shaped mart and checks both. It is the fastest way to see
the difference.

---

## Closing the loop on column constraints

de-template's `dbt/models/marts/schema.yml` declares `not_null` / `unique` tests
on every exported column. That is the contract on their side, and it fails their
build before the export runs.

Mirror the same constraints here as Pandera checks in
[`backend/src/schemas.py`](../backend/src/schemas.py), remembering the upper-case
names:

```python
FEATURE_COLUMNS = {
    "ITEM_ID": Column(int, checks=Check.ge(0)),
    "ITEM_NAME_LENGTH": Column(int, checks=Check.ge(0)),
}
```

The same violation then fails from both sides — there when dbt builds the table,
here when the file is read.

---

## CSV or Parquet

de-template's `exports.format` accepts both. This project reads **CSV** out of
the box; `read_data` in `backend/src/ml/io.py` calls `pandas.read_csv`. Parquet
carries its types in the file rather than re-inferring them from text, which is
worth having for large marts or `decimal` columns — but you have to teach `io.py`
to read it. It is a small change and a deliberate one.

---

## What each side owns

| | de-template | ds-template |
| --- | --- | --- |
| Extraction, loading, scheduling | ✅ | |
| SQL transformation up to `marts` | ✅ | |
| Feature engineering in SQL | ✅ | |
| Feature engineering in pandas | | ✅ |
| Training, evaluation, tuning | | ✅ |
| Experiment tracking, Model Registry | | ✅ |
| Serving predictions | | ✅ |

de-template has no MLflow and no `ml` dependency group; anything that fits or
scores a model belongs here. The reverse also holds: if a transformation could
be expressed in SQL over the warehouse and would be useful to more than one
model, it belongs there.

---

## Where the two projects have to agree

Three things are load-bearing across the seam. Changing any of them here breaks
something there, silently:

| What | Here | Why it matters |
| --- | --- | --- |
| `normalise_column_name` | `backend/src/schemas.py` | de-template reimplements it in `scripts/demo_handoff.py` to predict what this project will call each column. Change the rule and their check starts asserting the wrong names. |
| The CSV reader's expectations | `backend/src/ml/io.py` | Their exporter writes empty fields for NULL because that is what pandas reads as `NaN` here. |
| `pyarrow` floor | `backend/pyproject.toml` | Both pin `>=23.0.1` so a Parquet file written there is readable here without a version negotiation. |
