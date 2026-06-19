from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DROP_COLUMNS = [
    "cc_num",
    "first",
    "last",
    "street",
    "trans_num",
    "trans_date_trans_time",
    "dob",
    "merchant",
    "job",
    "city",
]

ONE_HOT_COLUMNS = ["category", "gender", "state"]
FREQUENCY_COLUMNS = ["merchant", "job", "city"]
SCALE_COLUMNS = [
    "amt",
    "city_pop",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "unix_time",
    "customer_age",
    "customer_merchant_distance_km",
]


class StandardScalerLite:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, data: pd.DataFrame) -> "StandardScalerLite":
        self.mean_ = data.mean(axis=0).to_numpy()
        self.scale_ = data.std(axis=0, ddof=0).replace(0, 1).to_numpy()
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fitted before transform.")
        transformed = (data.to_numpy() - self.mean_) / self.scale_
        return pd.DataFrame(transformed, columns=data.columns, index=data.index)

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.fit(data).transform(data)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def haversine_distance_km(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def load_raw_data(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = raw_dir / "fraudTrain.csv"
    test_path = raw_dir / "fraudTest.csv"

    date_cols = ["trans_date_trans_time", "dob"]
    train_df = pd.read_csv(train_path, index_col=0, parse_dates=date_cols)
    test_df = pd.read_csv(test_path, index_col=0, parse_dates=date_cols)
    return train_df, test_df


def add_time_and_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    trans_time = result["trans_date_trans_time"]

    result["transaction_hour"] = trans_time.dt.hour
    result["transaction_dayofweek"] = trans_time.dt.dayofweek
    result["transaction_month"] = trans_time.dt.month
    result["weekend_flag"] = result["transaction_dayofweek"].isin([5, 6]).astype(int)

    result["customer_age"] = ((trans_time - result["dob"]).dt.days / 365.25).astype(
        int
    )
    result["customer_merchant_distance_km"] = haversine_distance_km(
        result["lat"],
        result["long"],
        result["merch_lat"],
        result["merch_long"],
    )
    return result


def add_frequency_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: list[str] = FREQUENCY_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    train_result = train_df.copy()
    test_result = test_df.copy()
    mappings: dict[str, dict[str, float]] = {}

    for column in columns:
        frequencies = train_result[column].value_counts(normalize=True)
        feature_name = f"{column}_frequency"
        train_result[feature_name] = train_result[column].map(frequencies).fillna(0)
        test_result[feature_name] = test_result[column].map(frequencies).fillna(0)
        mappings[column] = {str(k): float(v) for k, v in frequencies.items()}

    return train_result, test_result, mappings


def one_hot_encode(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: list[str] = ONE_HOT_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(
        [train_df.assign(_dataset="train"), test_df.assign(_dataset="test")],
        axis=0,
        ignore_index=True,
    )
    combined = pd.get_dummies(combined, columns=columns, drop_first=False, dtype=int)

    train_encoded = combined[combined["_dataset"] == "train"].drop(columns="_dataset")
    test_encoded = combined[combined["_dataset"] == "test"].drop(columns="_dataset")
    return train_encoded.reset_index(drop=True), test_encoded.reset_index(drop=True)


def scale_numeric_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: list[str] = SCALE_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScalerLite]:
    train_result = train_df.copy()
    test_result = test_df.copy()

    scaler = StandardScalerLite()
    train_result[columns] = scaler.fit_transform(train_result[columns])
    test_result[columns] = scaler.transform(test_result[columns])
    return train_result, test_result, scaler


def preprocess(
    raw_dir: Path,
    processed_dir: Path,
    table_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = load_raw_data(raw_dir)

    train_df = add_time_and_distance_features(train_df)
    test_df = add_time_and_distance_features(test_df)

    train_df, test_df, frequency_mappings = add_frequency_features(train_df, test_df)
    train_df, test_df = one_hot_encode(train_df, test_df)

    train_df = train_df.drop(columns=DROP_COLUMNS)
    test_df = test_df.drop(columns=DROP_COLUMNS)

    y_train = train_df.pop("is_fraud")
    y_test = test_df.pop("is_fraud")

    train_df, test_df, scaler = scale_numeric_features(train_df, test_df)

    train_df["is_fraud"] = y_train.astype(int).values
    test_df["is_fraud"] = y_test.astype(int).values

    processed_dir.mkdir(parents=True, exist_ok=True)
    train_output = processed_dir / "fraud_train_processed.csv"
    test_output = processed_dir / "fraud_test_processed.csv"
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)

    if table_dir is not None:
        table_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_columns": int(train_df.shape[1]),
            "test_columns": int(test_df.shape[1]),
            "target_column": "is_fraud",
            "one_hot_columns": ONE_HOT_COLUMNS,
            "frequency_encoded_columns": FREQUENCY_COLUMNS,
            "scaled_columns": SCALE_COLUMNS,
            "dropped_columns": DROP_COLUMNS,
            "train_output": str(train_output),
            "test_output": str(test_output),
        }
        (table_dir / "preprocessing_report.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        feature_summary = pd.DataFrame(
            {
                "feature": train_df.columns,
                "dtype": train_df.dtypes.astype(str).values,
                "missing_count": train_df.isna().sum().values,
                "n_unique": train_df.nunique(dropna=True).values,
            }
        )
        feature_summary.to_csv(table_dir / "processed_feature_summary.csv", index=False)

        frequency_sizes = pd.DataFrame(
            {
                "column": list(frequency_mappings.keys()),
                "unique_values": [len(v) for v in frequency_mappings.values()],
            }
        )
        frequency_sizes.to_csv(table_dir / "frequency_encoding_summary.csv", index=False)

        scaler_stats = pd.DataFrame(
            {
                "feature": SCALE_COLUMNS,
                "mean": scaler.mean_,
                "scale": scaler.scale_,
            }
        )
        scaler_stats.to_csv(table_dir / "scaler_summary.csv", index=False)

    return train_df, test_df


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Preprocess fraud detection dataset.")
    parser.add_argument("--raw-dir", type=Path, default=root / "data" / "raw")
    parser.add_argument(
        "--processed-dir", type=Path, default=root / "data" / "processed"
    )
    parser.add_argument("--table-dir", type=Path, default=root / "reports" / "tables")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df, test_df = preprocess(args.raw_dir, args.processed_dir, args.table_dir)
    print("Preprocessing completed.")
    print(f"Train processed shape: {train_df.shape}")
    print(f"Test processed shape: {test_df.shape}")
    print(f"Processed files saved to: {args.processed_dir}")


if __name__ == "__main__":
    main()
