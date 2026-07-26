"""Data preparation, exploratory analysis, training, and inference.

Pipeline steps, each runnable on its own or in sequence from the CLI:

    pipeline.py       The abstract base class the steps implement.
    prepare_data.py   Raw CSV in, transformed CSV out.
    eda.py            Exploratory plots, saved locally and logged to MLflow.
    train_model.py    Fit, evaluate, and log a model with its signature.
    inference.py      Load a registered model and predict with it.

Supporting modules the steps share:

    io.py             CSV read/write with Pandera validation.
    plots.py          Matplotlib and seaborn helpers, kept apart so modules
                      needing only data access stay light to import.
    tracking.py       MLflow setup, flavour-aware model logging, run contexts.

This package knows nothing about HTTP. It raises domain exceptions —
``ModelNotAvailableError``, ``FeatureValidationError`` — and leaves the mapping
onto status codes to :mod:`src.api`, which is what keeps it testable without a
web server and usable from the CLI.
"""
