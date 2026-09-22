"""Distributed classification metrics and curve generation."""

from __future__ import annotations

from typing import List, Mapping, Optional, Sequence, Tuple

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def with_probability_score(predictions: DataFrame) -> DataFrame:
    if "probability_score" in predictions.columns:
        return predictions
    return predictions.withColumn(
        "probability_score", vector_to_array(F.col("probability"))[1].cast("double")
    )


def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "row_count": int(total),
    }


def roc_auc(predictions: DataFrame) -> float:
    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    return float(evaluator.evaluate(predictions))


def threshold_metric_rows(
    predictions: DataFrame,
    thresholds: Sequence[float],
    common_values: Optional[Mapping[str, object]] = None,
    include_roc_auc: bool = True,
) -> List[dict]:
    """Evaluate every threshold with one distributed aggregation."""

    scored = with_probability_score(predictions)
    expressions = []
    for index, threshold in enumerate(thresholds):
        positive = F.col("probability_score") >= F.lit(float(threshold))
        actual_positive = F.col("label") == 1.0
        expressions.extend(
            [
                F.sum((positive & actual_positive).cast("long")).alias(f"tp_{index}"),
                F.sum((positive & ~actual_positive).cast("long")).alias(f"fp_{index}"),
                F.sum((~positive & ~actual_positive).cast("long")).alias(f"tn_{index}"),
                F.sum((~positive & actual_positive).cast("long")).alias(f"fn_{index}"),
            ]
        )
    counts = scored.agg(*expressions).first().asDict()
    area = roc_auc(predictions) if include_roc_auc else None
    rows = []
    for index, threshold in enumerate(thresholds):
        row = dict(common_values or {})
        row.update(
            metrics_from_counts(
                int(counts[f"tp_{index}"] or 0),
                int(counts[f"fp_{index}"] or 0),
                int(counts[f"tn_{index}"] or 0),
                int(counts[f"fn_{index}"] or 0),
            )
        )
        row["threshold"] = float(threshold)
        row["roc_auc"] = area
        rows.append(row)
    return rows


def apply_threshold(predictions: DataFrame, threshold: float) -> DataFrame:
    scored = with_probability_score(predictions)
    return scored.withColumn(
        "prediction",
        (F.col("probability_score") >= F.lit(float(threshold))).cast("double"),
    )


def evaluate_at_threshold(predictions: DataFrame, threshold: float) -> Tuple[dict, DataFrame]:
    rows = threshold_metric_rows(predictions, [threshold])
    return rows[0], apply_threshold(predictions, threshold)


def majority_class(labelled: DataFrame) -> int:
    row = labelled.agg(
        F.count(F.lit(1)).alias("rows"),
        F.sum("label").alias("positives"),
    ).first()
    rows = int(row["rows"] or 0)
    positives = int(row["positives"] or 0)
    if rows == 0:
        raise ValueError("Cannot determine a baseline class from an empty dataset.")
    return 1 if positives > (rows - positives) else 0


def evaluate_constant_baseline(labelled: DataFrame, predicted_class: int) -> dict:
    positive = F.lit(int(predicted_class)) == 1
    actual_positive = F.col("label") == 1.0
    row = labelled.agg(
        F.sum((positive & actual_positive).cast("long")).alias("tp"),
        F.sum((positive & ~actual_positive).cast("long")).alias("fp"),
        F.sum((~positive & ~actual_positive).cast("long")).alias("tn"),
        F.sum((~positive & actual_positive).cast("long")).alias("fn"),
    ).first()
    result = metrics_from_counts(
        int(row["tp"] or 0),
        int(row["fp"] or 0),
        int(row["tn"] or 0),
        int(row["fn"] or 0),
    )
    result.update(
        {
            "predicted_class": int(predicted_class),
            "threshold": None,
            "roc_auc": None,
        }
    )
    return result


def binary_curve_rows(
    predictions: DataFrame,
    number_of_points: int = 101,
) -> Tuple[List[dict], List[dict]]:
    """Return approximate ROC and PR coordinates on a fixed score grid.

    Counts for every threshold are computed in one distributed aggregation, so no
    prediction rows are collected by the driver.
    """

    if number_of_points < 2:
        raise ValueError("number_of_points must be at least 2.")
    thresholds = [index / (number_of_points - 1) for index in range(number_of_points)]
    metric_rows = threshold_metric_rows(
        predictions, thresholds, include_roc_auc=False
    )
    roc_rows = []
    pr_rows = []
    for row in metric_rows:
        false_positive_denominator = row["false_positive"] + row["true_negative"]
        false_positive_rate = (
            row["false_positive"] / false_positive_denominator
            if false_positive_denominator
            else 0.0
        )
        roc_rows.append(
            {
                "threshold": row["threshold"],
                "false_positive_rate": false_positive_rate,
                "true_positive_rate": row["recall"],
            }
        )
        pr_rows.append(
            {
                "threshold": row["threshold"],
                "recall": row["recall"],
                "precision": row["precision"],
            }
        )
    roc_rows.sort(key=lambda item: (item["false_positive_rate"], item["true_positive_rate"]))
    pr_rows.sort(key=lambda item: item["recall"])
    return roc_rows, pr_rows
