from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_predictions(
    model_name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    y_score: pd.Series,
) -> tuple[dict[str, float], pd.DataFrame]:
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
    }

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "class"})
    report_df.insert(0, "model", model_name)
    return metrics, report_df


def plot_confusion_matrix(
    model_name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    figure_dir: Path,
) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_path = figure_dir / f"confusion_matrix_{model_name}.png"

    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Legitimate", "Fraud"],
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(f"Confusion Matrix - {model_name.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_roc_curves(
    roc_inputs: dict[str, tuple[pd.Series, pd.Series]],
    figure_dir: Path,
) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_path = figure_dir / "roc_curve_models.png"

    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, (y_true, y_score) in roc_inputs.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, label=f"{model_name.replace('_', ' ').title()} (AUC={auc:.4f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_title("ROC Curve Comparison")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_metrics_comparison(metrics_df: pd.DataFrame, figure_dir: Path) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_path = figure_dir / "model_metrics_comparison.png"

    metric_cols = ["precision", "recall", "f1_score", "roc_auc"]
    plot_df = metrics_df.melt(
        id_vars="model",
        value_vars=metric_cols,
        var_name="metric",
        value_name="score",
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=plot_df, x="metric", y="score", hue="model", ax=ax)
    ax.set_title("Model Metrics Comparison")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(title="Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path
