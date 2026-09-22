"""Command-line orchestration for the CS777 term-project implementation."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Sequence

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .cleaning import accepted_flights, cleaning_quality_rows, stage_cleaning
from .config import (
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_NUMERIC_FEATURES,
    ProjectConfig,
)
from .evaluation import (
    binary_curve_rows,
    evaluate_at_threshold,
    evaluate_constant_baseline,
    majority_class,
)
from .exports import (
    error_analysis_rows,
    plot_coefficients,
    plot_confusion_matrix,
    plot_model_selection,
    plot_precision_recall_curve,
    plot_roc_curve,
    prepare_output_directories,
    write_csv_rows,
    write_json,
)
from .features import (
    add_leakage_safe_features,
    chronological_partitions,
    historical_coverage_rows,
    split_summary_rows,
)
from .modeling import (
    coefficient_rows,
    fit_final_logistic,
    fit_preprocessor,
    select_logistic_model,
)


LOGGER = logging.getLogger("flight_delay_pipeline")


def _date_value(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Dates must use YYYY-MM-DD.") from exc


def _float_list(value: str) -> Sequence[float]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of numbers.") from exc
    if not values:
        raise argparse.ArgumentTypeError("At least one numeric value is required.")
    return values


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the proposal-approved leakage-safe PySpark "
            "logistic-regression flight-delay model."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more extracted BTS CSV paths or glob patterns.",
    )
    parser.add_argument(
        "--output",
        default="outputs/final_run",
        help="Local output directory for tables, metrics, figures, and samples.",
    )
    parser.add_argument("--start-date", type=_date_value, default=date(2023, 1, 1))
    parser.add_argument("--train-end-date", type=_date_value, default=date(2024, 12, 31))
    parser.add_argument(
        "--validation-end-date", type=_date_value, default=date(2025, 6, 30)
    )
    parser.add_argument("--test-end-date", type=_date_value, default=date(2025, 12, 31))
    parser.add_argument(
        "--regularization-grid",
        type=_float_list,
        default=(0.0, 0.001, 0.01, 0.1),
        help="Comma-separated L2 regParam candidates selected on validation F1.",
    )
    parser.add_argument(
        "--threshold-grid",
        type=_float_list,
        default=(0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70),
        help="Comma-separated probability thresholds selected on validation F1.",
    )
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--historical-prior-weight", type=float, default=20.0)
    parser.add_argument("--historical-default-rate", type=float, default=0.20)
    parser.add_argument("--shuffle-partitions", type=int, default=200)
    parser.add_argument(
        "--master",
        default=None,
        help="Optional Spark master for direct Python runs, for example local[*].",
    )
    parser.add_argument(
        "--no-refit",
        action="store_true",
        help="Do not refit the selected settings on train+validation before final testing.",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help=(
            "Optionally serialize Spark preprocessing/model artifacts. Disabled by "
            "default because the assignment does not require them."
        ),
    )
    parser.add_argument(
        "--export-test-predictions",
        action="store_true",
        help="Also save all test predictions as Parquet (can require substantial disk space).",
    )
    parser.add_argument("--log-level", default="WARN")
    return parser


def _build_spark(args: argparse.Namespace) -> SparkSession:
    builder = (
        SparkSession.builder.appName("CS777_Flight_Delay_Term_Project")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    )
    if args.master:
        builder = builder.master(args.master)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(args.log_level.upper())
    return spark


def _read_input(spark: SparkSession, input_paths: Sequence[str]) -> DataFrame:
    return (
        spark.read.option("header", "true")
        .option("inferSchema", "false")
        .option("mode", "PERMISSIVE")
        .option("multiLine", "false")
        .csv(list(input_paths))
    )


def _class_distribution_rows(split_rows: Sequence[dict]) -> List[dict]:
    rows = []
    for split in split_rows:
        total = int(split["row_count"])
        for class_label, class_name, count_key in (
            (0, "on_time_or_under_15_minutes_late", "on_time_count"),
            (1, "at_least_15_minutes_late", "delayed_count"),
        ):
            count = int(split[count_key])
            rows.append(
                {
                    "split": split["split"],
                    "class_label": class_label,
                    "class_name": class_name,
                    "row_count": count,
                    "share": count / total if total else 0.0,
                }
            )
    return rows


def _confusion_matrix_rows(metrics: dict) -> List[dict]:
    return [
        {
            "actual_class": "on_time_or_under_15_minutes_late",
            "predicted_on_time": int(metrics["true_negative"]),
            "predicted_delayed": int(metrics["false_positive"]),
        },
        {
            "actual_class": "at_least_15_minutes_late",
            "predicted_on_time": int(metrics["false_negative"]),
            "predicted_delayed": int(metrics["true_positive"]),
        },
    ]


def _final_metric_rows(
    logistic_metrics: dict,
    baseline_metrics: dict,
    best_selection: dict,
) -> List[dict]:
    logistic = {
        "model": "logistic_regression",
        "split": "test",
        "reg_param": best_selection["reg_param"],
        "elastic_net_param": 0.0,
        "threshold": best_selection["threshold"],
        **logistic_metrics,
    }
    baseline = {
        "model": "majority_class_baseline",
        "split": "test",
        "reg_param": None,
        "elastic_net_param": None,
        **baseline_metrics,
    }
    return [logistic, baseline]


def _validation_metric_rows(best_selection: dict, baseline_metrics: dict) -> List[dict]:
    logistic = dict(best_selection)
    baseline = {
        "model": "majority_class_baseline",
        "split": "validation",
        "reg_param": None,
        "elastic_net_param": None,
        "max_iterations": None,
        **baseline_metrics,
    }
    return [logistic, baseline]


def _safe_unpersist(*frames: DataFrame) -> None:
    for frame in frames:
        if frame is not None:
            frame.unpersist()


def execute(args: argparse.Namespace) -> Path:
    started_at = datetime.now(timezone.utc)
    output_root = Path(args.output).expanduser().resolve()
    directories = prepare_output_directories(
        output_root, include_models=bool(args.save_models)
    )
    config = ProjectConfig(
        start_date=args.start_date,
        train_end_date=args.train_end_date,
        validation_end_date=args.validation_end_date,
        test_end_date=args.test_end_date,
        historical_prior_weight=args.historical_prior_weight,
        historical_default_delay_rate=args.historical_default_rate,
        regularization_grid=tuple(args.regularization_grid),
        threshold_grid=tuple(args.threshold_grid),
        max_iterations=args.max_iterations,
    )
    config.validate()

    spark = _build_spark(args)
    LOGGER.info("Spark %s started.", spark.version)
    staged = accepted = featured = train = validation = test = None
    train_transformed = validation_transformed = final_train_transformed = None
    final_training_raw = None
    final_test_transformed = raw_test_predictions = test_predictions = None
    try:
        LOGGER.info("1/9 Reading BTS CSV data.")
        raw = _read_input(spark, args.input)

        LOGGER.info("2/9 Cleaning completed flights and creating the 15-minute label.")
        staged = stage_cleaning(raw, config.start_date, config.test_end_date).persist(
            StorageLevel.MEMORY_AND_DISK
        )
        accepted = accepted_flights(staged).persist(StorageLevel.MEMORY_AND_DISK)
        quality_rows = list(cleaning_quality_rows(staged, accepted))

        LOGGER.info("3/9 Building scheduled and strictly lagged historical features.")
        featured = add_leakage_safe_features(
            accepted,
            config.historical_prior_weight,
            config.historical_default_delay_rate,
        ).persist(StorageLevel.MEMORY_AND_DISK)
        featured.count()
        _safe_unpersist(staged, accepted)
        staged = accepted = None

        LOGGER.info("4/9 Creating chronological train, validation, and test partitions.")
        train, validation, test = chronological_partitions(
            featured,
            config.start_date,
            config.train_end_date,
            config.validation_end_date,
            config.test_end_date,
        )
        train = train.persist(StorageLevel.MEMORY_AND_DISK)
        validation = validation.persist(StorageLevel.MEMORY_AND_DISK)
        test = test.persist(StorageLevel.MEMORY_AND_DISK)
        partitions = {"train": train, "validation": validation, "test": test}
        split_rows = split_summary_rows(partitions)
        history_rows = historical_coverage_rows(partitions)
        featured.unpersist()
        featured = None

        LOGGER.info("5/9 Fitting preprocessing on training data only.")
        selection_preprocessor = fit_preprocessor(
            train,
            DEFAULT_CATEGORICAL_FEATURES,
            DEFAULT_NUMERIC_FEATURES,
        )
        train_transformed = selection_preprocessor.transform(train).persist(
            StorageLevel.MEMORY_AND_DISK
        )
        validation_transformed = selection_preprocessor.transform(validation).persist(
            StorageLevel.MEMORY_AND_DISK
        )
        train_transformed.count()
        validation_transformed.count()

        LOGGER.info("6/9 Selecting L2 regularization and threshold on validation F1.")
        best_model, best_selection, selection_rows = select_logistic_model(
            train_transformed,
            validation_transformed,
            config.regularization_grid,
            config.threshold_grid,
            config.max_iterations,
        )
        validation_baseline_class = majority_class(train)
        validation_baseline = evaluate_constant_baseline(
            validation, validation_baseline_class
        )
        _safe_unpersist(train_transformed, validation_transformed)
        train_transformed = validation_transformed = None

        LOGGER.info("7/9 Fitting the selected model without using final-test outcomes.")
        if args.no_refit:
            final_preprocessor = selection_preprocessor
            final_model = best_model
            final_training_raw = train
            final_test_transformed = selection_preprocessor.transform(test).persist(
                StorageLevel.MEMORY_AND_DISK
            )
        else:
            final_training_raw = train.unionByName(validation).persist(
                StorageLevel.MEMORY_AND_DISK
            )
            final_preprocessor = fit_preprocessor(
                final_training_raw,
                DEFAULT_CATEGORICAL_FEATURES,
                DEFAULT_NUMERIC_FEATURES,
            )
            final_train_transformed = final_preprocessor.transform(
                final_training_raw
            ).persist(StorageLevel.MEMORY_AND_DISK)
            final_test_transformed = final_preprocessor.transform(test).persist(
                StorageLevel.MEMORY_AND_DISK
            )
            final_train_transformed.count()
            final_test_transformed.count()
            final_model = fit_final_logistic(
                final_train_transformed,
                float(best_selection["reg_param"]),
                config.max_iterations,
            )

        raw_test_predictions = final_model.transform(final_test_transformed).persist(
            StorageLevel.MEMORY_AND_DISK
        )
        raw_test_predictions.count()
        logistic_test_metrics, test_predictions = evaluate_at_threshold(
            raw_test_predictions, float(best_selection["threshold"])
        )
        test_predictions = test_predictions.persist(StorageLevel.MEMORY_AND_DISK)
        test_predictions.count()
        test_baseline_class = majority_class(final_training_raw)
        test_baseline_metrics = evaluate_constant_baseline(test, test_baseline_class)

        LOGGER.info("8/9 Exporting report-ready metrics, tables, and figures.")
        coefficient_table = coefficient_rows(
            final_model, final_test_transformed, DEFAULT_NUMERIC_FEATURES
        )
        roc_rows, pr_rows = binary_curve_rows(raw_test_predictions)
        error_rows = error_analysis_rows(test_predictions)
        final_metric_rows = _final_metric_rows(
            logistic_test_metrics, test_baseline_metrics, best_selection
        )
        validation_metric_rows = _validation_metric_rows(
            best_selection, validation_baseline
        )

        write_csv_rows(directories["tables"] / "data_quality_summary.csv", quality_rows)
        write_csv_rows(directories["tables"] / "split_summary.csv", split_rows)
        write_csv_rows(
            directories["tables"] / "class_distribution.csv",
            _class_distribution_rows(split_rows),
        )
        write_csv_rows(
            directories["tables"] / "historical_feature_coverage.csv", history_rows
        )
        write_csv_rows(
            directories["tables"] / "model_selection_validation.csv", selection_rows
        )
        write_csv_rows(
            directories["tables"] / "selected_validation_metrics.csv",
            validation_metric_rows,
        )
        write_csv_rows(
            directories["tables"] / "final_test_metrics.csv", final_metric_rows
        )
        write_csv_rows(
            directories["tables"] / "final_test_confusion_matrix.csv",
            _confusion_matrix_rows(logistic_test_metrics),
        )
        write_csv_rows(
            directories["tables"] / "feature_coefficients.csv", coefficient_table
        )
        write_csv_rows(directories["tables"] / "roc_curve.csv", roc_rows)
        write_csv_rows(directories["tables"] / "precision_recall_curve.csv", pr_rows)
        if error_rows:
            write_csv_rows(
                directories["tables"] / "error_analysis_sample.csv", error_rows
            )

        plot_model_selection(
            selection_rows, directories["figures"] / "validation_model_selection.png"
        )
        plot_confusion_matrix(
            logistic_test_metrics,
            directories["figures"] / "final_test_confusion_matrix.png",
        )
        plot_roc_curve(
            roc_rows,
            float(logistic_test_metrics["roc_auc"]),
            directories["figures"] / "final_test_roc_curve.png",
        )
        delay_rate = next(
            float(row["delay_rate"]) for row in split_rows if row["split"] == "test"
        )
        plot_precision_recall_curve(
            pr_rows,
            delay_rate,
            directories["figures"] / "final_test_precision_recall_curve.png",
        )
        plot_coefficients(
            coefficient_table,
            directories["figures"] / "top_feature_coefficients.png",
        )

        model_serialization = {
            "requested": bool(args.save_models),
            "saved": False,
            "error": None,
        }
        if args.save_models:
            try:
                final_preprocessor.write().overwrite().save(
                    str(directories["models"] / "preprocessor")
                )
                final_model.write().overwrite().save(
                    str(directories["models"] / "logistic_regression")
                )
                model_serialization["saved"] = True
            except Exception as exc:
                model_serialization["error"] = f"{type(exc).__name__}: {exc}"
                LOGGER.warning(
                    "Optional Spark model serialization failed; required result "
                    "exports will still be completed: %s",
                    exc,
                )

        if args.export_test_predictions:
            test_predictions.select(
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
            ).write.mode("overwrite").parquet(
                str(directories["predictions"] / "test_predictions.parquet")
            )

        finished_at = datetime.now(timezone.utc)
        final_summary = {
            "project": "CS777 Flight Delay Prediction",
            "target": "completed flight arrival delay at least 15 minutes",
            "data_period": {
                "start_date": config.start_date.isoformat(),
                "train_end_date": config.train_end_date.isoformat(),
                "validation_end_date": config.validation_end_date.isoformat(),
                "test_end_date": config.test_end_date.isoformat(),
            },
            "historical_feature_rule": (
                "Each entity delay rate uses completed flights from strictly earlier "
                "flight dates; same-date and future labels are excluded."
            ),
            "selected_model": {
                "type": "PySpark logistic regression",
                "reg_param": float(best_selection["reg_param"]),
                "elastic_net_param": 0.0,
                "classification_threshold": float(best_selection["threshold"]),
                "max_iterations": config.max_iterations,
                "refit_on_train_plus_validation": not args.no_refit,
                "intercept": float(final_model.intercept),
                "iterations_completed": int(final_model.summary.totalIterations),
                "serialization": model_serialization,
            },
            "validation": {
                "logistic_regression": best_selection,
                "majority_class_baseline": validation_baseline,
            },
            "test": {
                "logistic_regression": logistic_test_metrics,
                "majority_class_baseline": test_baseline_metrics,
            },
            "split_summary": split_rows,
        }
        write_json(directories["metrics"] / "metrics.json", final_summary)
        write_json(
            directories["metrics"] / "run_manifest.json",
            {
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": finished_at.isoformat(),
                "spark_version": spark.version,
                "python_version": sys.version,
                "input_paths": list(args.input),
                "output_root": str(output_root),
                "arguments": vars(args),
            },
        )

        LOGGER.info("9/9 Complete. Results written to %s", output_root)
        return output_root
    finally:
        _safe_unpersist(
            staged,
            accepted,
            featured,
            train_transformed,
            validation_transformed,
            final_train_transformed,
            final_test_transformed,
            raw_test_predictions,
            test_predictions,
            final_training_raw if final_training_raw is not train else None,
            train,
            validation,
            test,
        )
        spark.stop()


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    execute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
