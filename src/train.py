from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from evaluate import (
    evaluate_predictions,
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_roc_curves,
)


TARGET_COLUMN = "is_fraud"
RANDOM_STATE = 42


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_processed_data(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = processed_dir / "fraud_train_processed.csv"
    test_path = processed_dir / "fraud_test_processed.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return downcast_dataframe(train_df), downcast_dataframe(test_df)


def downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in result.columns:
        if column == TARGET_COLUMN:
            result[column] = result[column].astype("int8")
        elif pd.api.types.is_float_dtype(result[column]):
            result[column] = result[column].astype("float32")
        elif pd.api.types.is_integer_dtype(result[column]):
            result[column] = pd.to_numeric(result[column], downcast="integer")
    return result


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=TARGET_COLUMN)
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def make_training_sample(
    train_df: pd.DataFrame,
    negative_to_positive_ratio: int,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    fraud_df = train_df[train_df[TARGET_COLUMN] == 1]
    non_fraud_df = train_df[train_df[TARGET_COLUMN] == 0]

    sample_size = min(len(non_fraud_df), len(fraud_df) * negative_to_positive_ratio)
    sampled_non_fraud = non_fraud_df.sample(n=sample_size, random_state=random_state)

    sampled = pd.concat([fraud_df, sampled_non_fraud], axis=0)
    return sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)


def build_models() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=80,
            max_depth=14,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def predict_scores(model: object, X_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        return (scores - scores.min()) / (scores.max() - scores.min())
    raise TypeError("Model must expose predict_proba or decision_function.")


def train_and_evaluate(
    processed_dir: Path,
    model_dir: Path,
    table_dir: Path,
    figure_dir: Path,
    negative_to_positive_ratio: int,
) -> pd.DataFrame:
    train_df, test_df = load_processed_data(processed_dir)
    sampled_train_df = make_training_sample(train_df, negative_to_positive_ratio)

    X_train, y_train = split_features_target(sampled_train_df)
    X_test, y_test = split_features_target(test_df)

    model_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, float]] = []
    report_tables: list[pd.DataFrame] = []
    roc_inputs: dict[str, tuple[pd.Series, pd.Series]] = {}
    saved_models: dict[str, str] = {}

    for model_name, model in build_models().items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)

        y_pred = pd.Series(model.predict(X_test), name="prediction")
        y_score = pd.Series(predict_scores(model, X_test), name="score")

        metrics, report_df = evaluate_predictions(model_name, y_test, y_pred, y_score)
        metrics_rows.append(metrics)
        report_tables.append(report_df)
        roc_inputs[model_name] = (y_test, y_score)

        model_path = model_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        saved_models[model_name] = str(model_path)

        report_df.to_csv(
            table_dir / f"classification_report_{model_name}.csv",
            index=False,
        )
        plot_confusion_matrix(model_name, y_test, y_pred, figure_dir)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("f1_score", ascending=False)
    metrics_df.to_csv(table_dir / "model_metrics.csv", index=False)

    all_reports = pd.concat(report_tables, ignore_index=True)
    all_reports.to_csv(table_dir / "classification_reports_all_models.csv", index=False)

    plot_roc_curves(roc_inputs, figure_dir)
    plot_metrics_comparison(metrics_df, figure_dir)

    modeling_report = {
        "train_rows_full": int(len(train_df)),
        "train_rows_sampled": int(len(sampled_train_df)),
        "test_rows": int(len(test_df)),
        "features": int(X_train.shape[1]),
        "negative_to_positive_ratio": negative_to_positive_ratio,
        "models": saved_models,
        "metrics_output": str(table_dir / "model_metrics.csv"),
    }
    (table_dir / "modeling_report.json").write_text(
        json.dumps(modeling_report, indent=2),
        encoding="utf-8",
    )

    print("Training and evaluation completed.")
    print(metrics_df.to_string(index=False))
    return metrics_df


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Train fraud detection models.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=root / "data" / "processed",
    )
    parser.add_argument("--model-dir", type=Path, default=root / "models")
    parser.add_argument("--table-dir", type=Path, default=root / "reports" / "tables")
    parser.add_argument("--figure-dir", type=Path, default=root / "reports" / "figures")
    parser.add_argument("--negative-ratio", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_evaluate(
        processed_dir=args.processed_dir,
        model_dir=args.model_dir,
        table_dir=args.table_dir,
        figure_dir=args.figure_dir,
        negative_to_positive_ratio=args.negative_ratio,
    )


if __name__ == "__main__":
    main()
