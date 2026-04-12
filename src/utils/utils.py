import logging
import os
import warnings
from pathlib import Path
from typing import Any, cast

import boto3
import pandas as pd
import pandera as pa
import snowflake.connector
import yaml
from cryptography.hazmat.primitives import serialization
from dotenv import load_dotenv
from snowflake.connector.pandas_tools import write_pandas

from src.utils.schemas import schemas

load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def read_config() -> dict[str, Any]:
    """Read the YAML configuration file and return its contents.

    Returns:
        dict[str, Any]: Configuration settings loaded from ``cfg/config.yaml``.
    """
    config_file_path = os.path.join("cfg", "config.yaml")
    with open(config_file_path, "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)
    return cast(dict[str, Any], config_data)


def _load_private_key_der() -> bytes:
    """Load the Snowflake private key in DER (PKCS8) format from local filesystem.

    Returns:
        bytes: Private key in DER format.

    Raises:
        EnvironmentError: If required environment variables are missing.
    """
    key_path = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH", "")
    if not key_path:
        raise EnvironmentError(
            "SNOWFLAKE_PRIVATE_KEY_PATH is not set in the project environment."
        )

    passphrase = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", "")
    with open(key_path, "rb") as file:
        key = serialization.load_pem_private_key(
            file.read(),
            password=passphrase.encode("utf-8") if passphrase else None,
        )

    return key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def read_data(
    sql_query: str | None = None,
    table_name: str | None = None,
    schema_obj: str | None = None,
    account: str | None = os.getenv("SNOWFLAKE_ACCOUNT"),
    warehouse: str | None = os.getenv("SNOWFLAKE_WAREHOUSE"),
    schema: str | None = os.getenv("SNOWFLAKE_SCHEMA"),
) -> pd.DataFrame:
    """Read data from Snowflake.

    Automatically selects user, role, and database based on environment:
    - EXECUTION_ENV='local': Uses LOCAL variants
    - EXECUTION_ENV='aws' + DEPLOYMENT_ENV='dev': Uses DEV variants
    - EXECUTION_ENV='aws' + DEPLOYMENT_ENV='prod': Uses PROD variants

    Args:
        sql_query: SQL query to execute.
        table_name: Target table name (used if sql_query not provided).
        schema_obj: Optional Pandera schema key for validation.
        account: Snowflake account. Defaults to SNOWFLAKE_ACCOUNT env var.
        warehouse: Snowflake warehouse. Defaults to SNOWFLAKE_WAREHOUSE env var.
        schema: Snowflake schema. Defaults to SNOWFLAKE_SCHEMA env var.

    Returns:
        pandas.DataFrame: Loaded data.

    Raises:
        ValueError: If neither sql_query nor table_name is provided,
            or if DEPLOYMENT_ENV is invalid in AWS environment.
        EnvironmentError: If SNOWFLAKE_ACCOUNT not set.
    """
    if sql_query is None and table_name is None:
        raise ValueError("Either sql_query or table_name must be provided.")

    if not account:
        raise EnvironmentError("SNOWFLAKE_ACCOUNT must be provided.")

    exec_env = os.getenv("EXECUTION_ENV", "local").lower()
    deployment_env = os.getenv("DEPLOYMENT_ENV", "dev").lower()
    logger.debug(f"EXECUTION_ENV={exec_env}, DEPLOYMENT_ENV={deployment_env}")

    # Select user, role, and database based on environment
    if exec_env == "local":
        user = os.getenv("SNOWFLAKE_USER_LOCAL")
        database = os.getenv("SNOWFLAKE_DATABASE_LOCAL")
        role = os.getenv("SNOWFLAKE_ROLE_LOCAL")
    elif exec_env == "aws":
        if deployment_env == "prod":
            user = os.getenv("SNOWFLAKE_USER_PROD")
            database = os.getenv("SNOWFLAKE_DATABASE_PROD")
            role = os.getenv("SNOWFLAKE_ROLE_PROD")
        else:  # default to dev
            user = os.getenv("SNOWFLAKE_USER_DEV")
            database = os.getenv("SNOWFLAKE_DATABASE_DEV")
            role = os.getenv("SNOWFLAKE_ROLE_DEV")
    else:
        raise ValueError(f"Invalid EXECUTION_ENV: {exec_env}")

    conn_params: dict[str, Any]
    if exec_env == "aws":
        conn_params = {
            "account": account,
            "user": user,
            "authenticator": "WORKLOAD_IDENTITY",
            "workload_identity_provider": "AWS",
            "warehouse": warehouse,
            "schema": schema,
            "role": role,
            "database": database,
            # use default role for Snowflake user
        }
    else:
        conn_params = {
            "user": user,
            "account": account,
            "warehouse": warehouse,
            "database": database,
            "schema": schema,
            "private_key": _load_private_key_der(),
            "role": role,
        }

    # Log connection params (exclude private_key)
    safe_params = {k: v for k, v in conn_params.items() if k != "private_key"}
    logger.info("Snowflake connection params: %s", safe_params)
    if table_name:
        logger.info("Querying table: %s.%s.%s", database, schema, table_name)

    with snowflake.connector.connect(**conn_params) as conn:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="pandas only supports SQLAlchemy connectable",
                category=UserWarning,
            )
            if sql_query and table_name:
                logger.warning(
                    "Both sql_query and table_name provided. Using sql_query."
                )
                df = pd.read_sql(sql_query, conn)
            elif sql_query:
                df = pd.read_sql(sql_query, conn)
            else:
                df = pd.read_sql(f"SELECT * FROM {table_name};", conn)

    logger.info("Data successfully read from Snowflake. Shape: %s", df.shape)

    if schema_obj:
        try:
            schemas[schema_obj].validate(df)
            logger.info("Data schema validation passed.")
        except pa.errors.SchemaError as exc:
            logger.error("Data schema validation failed: %s", exc)
            raise

    return df


def write_data(
    df: pd.DataFrame,
    table_name: str,
    schema_obj: str | None,
    account: str | None = os.getenv("SNOWFLAKE_ACCOUNT"),
    warehouse: str | None = os.getenv("SNOWFLAKE_WAREHOUSE"),
    schema: str | None = os.getenv("SNOWFLAKE_SCHEMA"),
) -> None:
    """Write data to Snowflake.

    Automatically selects user, role, and database based on environment:
    - EXECUTION_ENV='local': Uses LOCAL variants
    - EXECUTION_ENV='aws' + DEPLOYMENT_ENV='dev': Uses DEV variants
    - EXECUTION_ENV='aws' + DEPLOYMENT_ENV='prod': Uses PROD variants

    Args:
        df: DataFrame to write.
        table_name: Target table name.
        schema_obj: Optional Pandera schema key for validation.
        account: Snowflake account. Defaults to SNOWFLAKE_ACCOUNT env var.
        warehouse: Snowflake warehouse. Defaults to SNOWFLAKE_WAREHOUSE env var.
        schema: Snowflake schema. Defaults to SNOWFLAKE_SCHEMA env var.
        role: Snowflake role (used for local execution). Defaults to SNOWFLAKE_ROLE_LOCAL env var.

    Raises:
        ValueError: If DEPLOYMENT_ENV is invalid in AWS environment.
        EnvironmentError: If SNOWFLAKE_ACCOUNT not set.
    """
    if df.empty:
        logger.info("DataFrame is empty; skipping write.")
        return

    if not account:
        raise EnvironmentError("SNOWFLAKE_ACCOUNT must be provided.")

    if schema_obj:
        try:
            schemas[schema_obj].validate(df)
            logger.info("Data schema validation passed.")
        except pa.errors.SchemaError as exc:
            logger.error("Data schema validation failed: %s", exc)
            raise

    exec_env = os.getenv("EXECUTION_ENV", "local").lower()
    deployment_env = os.getenv("DEPLOYMENT_ENV", "dev").lower()
    logger.debug(f"EXECUTION_ENV={exec_env}, DEPLOYMENT_ENV={deployment_env}")

    # Select user, role, and database based on environment
    if exec_env == "local":
        user = os.getenv("SNOWFLAKE_USER_LOCAL")
        database = os.getenv("SNOWFLAKE_DATABASE_LOCAL")
        role = os.getenv("SNOWFLAKE_ROLE_LOCAL")
    elif exec_env == "aws":
        if deployment_env == "prod":
            user = os.getenv("SNOWFLAKE_USER_PROD")
            database = os.getenv("SNOWFLAKE_DATABASE_PROD")
            role = os.getenv("SNOWFLAKE_ROLE_PROD")
        else:  # default to dev
            user = os.getenv("SNOWFLAKE_USER_DEV")
            database = os.getenv("SNOWFLAKE_DATABASE_DEV")
            role = os.getenv("SNOWFLAKE_ROLE_DEV")
    else:
        raise ValueError(f"Invalid EXECUTION_ENV: {exec_env}")

    conn_params: dict[str, Any]
    if exec_env == "aws":
        conn_params = {
            "account": account,
            "user": user,
            "authenticator": "WORKLOAD_IDENTITY",
            "workload_identity_provider": "AWS",
            "warehouse": warehouse,
            "schema": schema,
            "database": database,
            "role": role,
            # use default role for Snowflake user
        }
    else:
        conn_params = {
            "user": user,
            "account": account,
            "warehouse": warehouse,
            "database": database,
            "schema": schema,
            "private_key": _load_private_key_der(),
            "role": role,
        }

    # Log connection params (exclude private_key)
    safe_params = {k: v for k, v in conn_params.items() if k != "private_key"}
    logger.info("Snowflake write connection params: %s", safe_params)
    logger.info("Writing to table: %s.%s.%s", database, schema, table_name)

    with snowflake.connector.connect(**conn_params) as conn:
        success, _, nrows, _ = write_pandas(
            conn=conn,
            df=df,
            table_name=table_name,
            database=database,
            schema=schema,
            quote_identifiers=True,
            auto_create_table=True,
            overwrite=True,
        )

    if success:
        logger.info(
            "Data successfully written to Snowflake table '%s'. Rows: %s",
            table_name,
            nrows,
        )
    else:
        raise RuntimeError(f"Failed to write data to Snowflake table '{table_name}'.")


def upload_mlruns_to_s3() -> None:
    """Upload MLflow local file-store runs to S3.

    Expects the following environment variables:
    - MLFLOW_MLRUNS_BUCKET: S3 bucket to upload to (required)
    - MLFLOW_MLRUNS_PREFIX: S3 prefix/folder (optional, defaults to "mlruns")
    - AWS_REGION: AWS region for S3 (optional, defaults to "eu-west-2")
    - EXECUTION_ENV: Should be 'AWS' to enable upload
    """
    # Only upload if running on AWS
    if os.getenv("EXECUTION_ENV", "local").lower() != "aws":
        logger.info("Not running on AWS; skipping MLflow S3 upload.")
        return

    bucket = os.getenv("MLFLOW_MLRUNS_BUCKET")
    if not bucket:
        logger.warning("MLFLOW_MLRUNS_BUCKET not set; skipping MLflow upload.")
        return

    prefix = os.getenv("MLFLOW_MLRUNS_PREFIX", "mlruns")
    region = os.getenv("AWS_REGION", "eu-west-2")

    base = Path("/opt/ml/processing/mlruns")
    if not base.exists():
        logger.warning("No local mlruns directory found; skipping upload")
        return

    logger.info(f"Uploading MLflow runs from {base} to s3://{bucket}/{prefix}")

    s3 = boto3.client("s3", region_name=region)

    uploaded = 0
    for path in base.rglob("*"):
        if path.is_file():
            key = f"{prefix}/{path.relative_to(base)}"
            s3.upload_file(str(path), bucket, key)
            uploaded += 1

    logger.info(f"Uploaded {uploaded} MLflow files to S3.")
