"""Small, report-ready table, metric, sample, and figure exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def prepare_output_directories(output_root: Path, include_models: bool = False) -> dict:
    directories = {
        "root": output_root,
        "tables": output_root / "tables",
        "metrics": output_root / "metrics",
        "figures": output_root / "figures",
        "predictions": output_root / "predictions",
    }
    if include_models:
        directories["models"] = output_root / "models"
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV table: {path}")
    fieldnames = []
    for row in rows:
        for name in row.keys():
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False, default=str)
        handle.write("\n")


def error_analysis_rows(predictions: DataFrame, per_error_type: int = 100) -> List[dict]:
    columns = [
        "flight_key",
        "flight_date",
        "carrier",
        "flight_number",
        "origin",
        "dest",
        "crs_dep_time",
        "crs_arr_time",
        "label",
        "probability_score",
        "prediction",
    ]
    false_positives = (
        predictions.filter((F.col("label") == 0.0) & (F.col("prediction") == 1.0))
        .withColumn("error_type", F.lit("false_positive"))
        .withColumn("error_confidence", F.col("probability_score"))
        .orderBy(F.desc("error_confidence"), F.asc("flight_key"))
        .limit(per_error_type)
    )
    false_negatives = (
        predictions.filter((F.col("label") == 1.0) & (F.col("prediction") == 0.0))
        .withColumn("error_type", F.lit("false_negative"))
        .withColumn("error_confidence", 1.0 - F.col("probability_score"))
        .orderBy(F.desc("error_confidence"), F.asc("flight_key"))
        .limit(per_error_type)
    )
    records = false_positives.unionByName(false_negatives).select(
        *columns, "error_type", "error_confidence"
    ).collect()
    rows = []
    for record in records:
        item = record.asDict()
        item["flight_date"] = item["flight_date"].isoformat()
        rows.append(item)
    return rows


def plot_model_selection(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    best_by_reg = {}
    for row in rows:
        reg = float(row["reg_param"])
        if reg not in best_by_reg or float(row["f1"]) > float(best_by_reg[reg]["f1"]):
            best_by_reg[reg] = row
    ordered = [best_by_reg[key] for key in sorted(best_by_reg)]
    labels = [f"{float(row['reg_param']):g}\n(t={float(row['threshold']):.2f})" for row in ordered]
    values = [float(row["f1"]) for row in ordered]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(labels, values, color="#3366A5")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xlabel("L2 regularization (best validation threshold shown)")
    ax.set_ylabel("Validation F1")
    ax.set_title("Chronological validation model selection")
    ax.set_ylim(0, min(1.0, max(values) * 1.18 if values else 1.0))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_confusion_matrix(metrics: Mapping[str, object], path: Path) -> None:
    matrix = np.array(
        [
            [int(metrics["true_negative"]), int(metrics["false_positive"])],
            [int(metrics["false_negative"]), int(metrics["true_positive"])],
        ]
    )
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
                fontsize=12,
                fontweight="bold",
            )
    ax.set_xticks([0, 1], labels=["On time", "Delayed"])
    ax.set_yticks([0, 1], labels=["On time", "Delayed"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("Final chronological test confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_roc_curve(rows: Sequence[Mapping[str, object]], auc: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot(
        [float(row["false_positive_rate"]) for row in rows],
        [float(row["true_positive_rate"]) for row in rows],
        color="#3366A5",
        linewidth=2,
        label=f"Logistic regression (AUC={auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="No-skill")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_title("Final chronological test ROC curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_precision_recall_curve(
    rows: Sequence[Mapping[str, object]],
    delay_rate: float,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot(
        [float(row["recall"]) for row in rows],
        [float(row["precision"]) for row in rows],
        color="#9A4D1D",
        linewidth=2,
        label="Logistic regression",
    )
    ax.axhline(delay_rate, linestyle="--", color="#777777", label="Delay prevalence")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Final chronological test precision-recall curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_coefficients(
    coefficient_rows: Sequence[Mapping[str, object]],
    path: Path,
    top_n: int = 20,
) -> None:
    selected = list(coefficient_rows[:top_n])[::-1]
    labels = [str(row["feature_name"]) for row in selected]
    values = [float(row["coefficient"]) for row in selected]
    colors = ["#B84A4A" if value > 0 else "#3E7A5E" for value in values]
    height = max(6.0, 0.34 * len(selected) + 1.6)
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_xlabel("Coefficient on preprocessed feature scale")
    ax.set_title(f"Top {len(selected)} logistic-regression coefficients by magnitude")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
