from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def create_spark_session(app_name: str = "FraudDetectionBigData") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def load_transactions(spark: SparkSession, csv_path: Path) -> DataFrame:
    df = spark.read.csv(str(csv_path), header=True, inferSchema=True)

    # Kaggle file has an unnamed index column. Spark reads it as `_c0`.
    if "_c0" in df.columns:
        df = df.drop("_c0")

    return (
        df.withColumn(
            "trans_date_trans_time",
            F.to_timestamp("trans_date_trans_time", "yyyy-MM-dd HH:mm:ss"),
        )
        .withColumn("transaction_hour", F.hour("trans_date_trans_time"))
        .withColumn("transaction_dayofweek", F.dayofweek("trans_date_trans_time"))
        .withColumn("transaction_month", F.month("trans_date_trans_time"))
        .withColumn(
            "weekend_flag",
            F.when(F.col("transaction_dayofweek").isin([1, 7]), F.lit(1)).otherwise(
                F.lit(0)
            ),
        )
    )


def save_pandas_table(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def dataset_summary(df: DataFrame) -> pd.DataFrame:
    total_rows = df.count()
    total_columns = len(df.columns)
    fraud_count = df.filter(F.col("is_fraud") == 1).count()
    non_fraud_count = total_rows - fraud_count

    return pd.DataFrame(
        [
            {
                "total_rows": total_rows,
                "total_columns": total_columns,
                "fraud_count": fraud_count,
                "non_fraud_count": non_fraud_count,
                "fraud_rate_percent": fraud_count / total_rows * 100,
            }
        ]
    )


def fraud_by_category(df: DataFrame) -> pd.DataFrame:
    return (
        df.groupBy("category")
        .agg(
            F.count("*").alias("transactions"),
            F.sum("is_fraud").alias("fraud_count"),
            F.avg("is_fraud").alias("fraud_rate"),
            F.avg("amt").alias("avg_amount"),
            F.expr("percentile_approx(amt, 0.5)").alias("median_amount"),
        )
        .withColumn("fraud_rate_percent", F.col("fraud_rate") * 100)
        .orderBy(F.desc("fraud_rate_percent"), F.desc("transactions"))
        .toPandas()
    )


def fraud_by_state(df: DataFrame) -> pd.DataFrame:
    return (
        df.groupBy("state")
        .agg(
            F.count("*").alias("transactions"),
            F.sum("is_fraud").alias("fraud_count"),
            F.avg("is_fraud").alias("fraud_rate"),
            F.avg("amt").alias("avg_amount"),
        )
        .withColumn("fraud_rate_percent", F.col("fraud_rate") * 100)
        .orderBy(F.desc("fraud_rate_percent"), F.desc("transactions"))
        .toPandas()
    )


def high_risk_merchants(df: DataFrame, min_transactions: int = 100) -> pd.DataFrame:
    return (
        df.groupBy("merchant")
        .agg(
            F.count("*").alias("transactions"),
            F.sum("is_fraud").alias("fraud_count"),
            F.avg("is_fraud").alias("fraud_rate"),
            F.avg("amt").alias("avg_amount"),
        )
        .filter(F.col("transactions") >= min_transactions)
        .withColumn("fraud_rate_percent", F.col("fraud_rate") * 100)
        .orderBy(F.desc("fraud_rate_percent"), F.desc("fraud_count"))
        .limit(25)
        .toPandas()
    )


def fraud_by_hour(df: DataFrame) -> pd.DataFrame:
    return (
        df.groupBy("transaction_hour")
        .agg(
            F.count("*").alias("transactions"),
            F.sum("is_fraud").alias("fraud_count"),
            F.avg("is_fraud").alias("fraud_rate"),
            F.avg("amt").alias("avg_amount"),
        )
        .withColumn("fraud_rate_percent", F.col("fraud_rate") * 100)
        .orderBy("transaction_hour")
        .toPandas()
    )


def amount_by_target(df: DataFrame) -> pd.DataFrame:
    return (
        df.groupBy("is_fraud")
        .agg(
            F.count("*").alias("transactions"),
            F.avg("amt").alias("avg_amount"),
            F.expr("percentile_approx(amt, 0.5)").alias("median_amount"),
            F.min("amt").alias("min_amount"),
            F.max("amt").alias("max_amount"),
        )
        .orderBy("is_fraud")
        .toPandas()
    )


def compare_pandas_and_spark_category(raw_csv_path: Path, spark_df: DataFrame) -> pd.DataFrame:
    start = time.perf_counter()
    pandas_df = pd.read_csv(raw_csv_path, usecols=["category", "is_fraud"])
    pandas_result = (
        pandas_df.groupby("category")["is_fraud"]
        .agg(transactions="size", fraud_count="sum", fraud_rate="mean")
        .reset_index()
    )
    pandas_seconds = time.perf_counter() - start

    start = time.perf_counter()
    spark_result = (
        spark_df.groupBy("category")
        .agg(
            F.count("*").alias("transactions"),
            F.sum("is_fraud").alias("fraud_count"),
            F.avg("is_fraud").alias("fraud_rate"),
        )
        .toPandas()
    )
    spark_seconds = time.perf_counter() - start

    return pd.DataFrame(
        [
            {
                "method": "Pandas",
                "seconds": pandas_seconds,
                "result_rows": len(pandas_result),
            },
            {
                "method": "Spark",
                "seconds": spark_seconds,
                "result_rows": len(spark_result),
            },
        ]
    )


def run_spark_processing(raw_csv_path: Path, table_dir: Path) -> dict[str, Path]:
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    outputs = {
        "spark_dataset_summary": table_dir / "spark_dataset_summary.csv",
        "spark_fraud_by_category": table_dir / "spark_fraud_by_category.csv",
        "spark_fraud_by_state": table_dir / "spark_fraud_by_state.csv",
        "spark_high_risk_merchants": table_dir / "spark_high_risk_merchants.csv",
        "spark_fraud_by_hour": table_dir / "spark_fraud_by_hour.csv",
        "spark_amount_by_target": table_dir / "spark_amount_by_target.csv",
        "spark_pandas_comparison": table_dir / "spark_pandas_comparison.csv",
    }

    try:
        transactions = load_transactions(spark, raw_csv_path).cache()
        transactions.count()

        tables = {
            "spark_dataset_summary": dataset_summary(transactions),
            "spark_fraud_by_category": fraud_by_category(transactions),
            "spark_fraud_by_state": fraud_by_state(transactions),
            "spark_high_risk_merchants": high_risk_merchants(transactions),
            "spark_fraud_by_hour": fraud_by_hour(transactions),
            "spark_amount_by_target": amount_by_target(transactions),
            "spark_pandas_comparison": compare_pandas_and_spark_category(
                raw_csv_path, transactions
            ),
        }

        for name, table in tables.items():
            save_pandas_table(table, outputs[name])

        manifest = {
            "input": str(raw_csv_path),
            "outputs": {name: str(path) for name, path in outputs.items()},
            "spark_version": spark.version,
        }
        (table_dir / "spark_processing_report.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

        return outputs
    finally:
        spark.stop()


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Run Spark aggregations for fraud data.")
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=root / "data" / "raw" / "fraudTrain.csv",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=root / "reports" / "tables",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_spark_processing(args.raw_csv, args.table_dir)
    print("Spark processing completed.")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
