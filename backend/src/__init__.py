"""Backend for the data science template: pipelines, ML, and the API over them.

Layout, and the one rule that keeps it navigable:

    cli.py       Entry point for the terminal workflow (``uv run pipeline``).
    api/         Entry point for the browser workflow (``uvicorn src.api.app``).
    ml/          Data preparation, EDA, training, and inference.
    config.py    Paths, .env, and cfg/config.yaml. Shared, dependency-light.
    schemas.py   Pandera contracts for the DataFrames. Shared.

**``api/`` translates, ``ml/`` decides — and never the reverse.** Nothing in
``ml/`` imports from ``api/``. Prediction rules, validation, and MLflow access
live in ``ml/``; the API only maps their results and exceptions onto HTTP status
codes. That is what lets the pipelines be tested without a web server, the API
be tested without MLflow, and the CLI keep working regardless of the API.

Note the vocabulary, since the obvious words are overloaded here: a *model* is a
trained estimator, *schemas* are Pandera DataFrame contracts, and HTTP payload
shapes are called *contracts* (:mod:`src.api.contracts`) to keep all three
distinct.
"""
