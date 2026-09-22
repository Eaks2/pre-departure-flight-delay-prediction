"""Leakage-safe scheduled and historical feature engineering."""

from __future__ import annotations

import math
from datetime import date
from typing import Dict, List, Sequence, Tuple

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def _global_prior_by_date(
    flights: DataFrame,
    default_rate: float,
) -> DataFrame:
    """Build the global prior without an unpartitioned Spark window.

    The approved project spans only 1,096 dates. Spark first performs the large
    distributed aggregation to one row per date; the ordered daily totals are then
    collected and prefix-summed locally. This avoids forcing the flight-level data
    through a single-partition window while retaining the exact FlightDate < t rule.
    """

    daily_rows = (
        flights.groupBy("flight_date")
        .agg(
            F.sum(F.col("label").cast("long")).alias("daily_delays"),
            F.count(F.lit(1)).alias("daily_flights"),
        )
        .orderBy("flight_date")
        .collect()
    )
    prior_rows = []
    cumulative_delays = 0
    cumulative_flights = 0
    for row in daily_rows:
        prior_rate = (
            cumulative_delays / cumulative_flights
            if cumulative_flights > 0
            else float(default_rate)
        )
        prior_rows.append(
            (row["flight_date"], int(cumulative_flights), float(prior_rate))
        )
        cumulative_delays += int(row["daily_delays"] or 0)
        cumulative_flights += int(row["daily_flights"] or 0)

    return flights.sparkSession.createDataFrame(
        prior_rows,
        schema=(
            "flight_date date, hist_global_count long, "
            "hist_global_delay_rate double"
        ),
    )


def _group_history_table(
    flights: DataFrame,
    group_columns: Sequence[str],
    feature_stem: str,
) -> DataFrame:
    """Create one partitioned prior-date history table for an entity type."""

    daily = flights.groupBy(*group_columns, "flight_date").agg(
        F.sum(F.col("label").cast("long")).alias("daily_group_delays"),
        F.count(F.lit(1)).alias("daily_flights"),
    )
    daily = daily.withColumnRenamed("daily_flights", "daily_group_flights")
    prior_window = (
        Window.partitionBy(*group_columns)
        .orderBy("flight_date")
        .rowsBetween(Window.unboundedPreceding, -1)
    )
    return (
        daily.withColumn(
            f"hist_{feature_stem}_delay_count",
            F.sum("daily_group_delays").over(prior_window),
        )
        .withColumn(
            f"hist_{feature_stem}_count",
            F.sum("daily_group_flights").over(prior_window),
        )
        .select(
            *group_columns,
            "flight_date",
            f"hist_{feature_stem}_delay_count",
            f"hist_{feature_stem}_count",
        )
    )


def _join_group_history(
    featured: DataFrame,
    history: DataFrame,
    group_columns: Sequence[str],
    feature_stem: str,
    prior_weight: float,
) -> DataFrame:
    """Join a history table and calculate its smoothed rate."""

    count_column = f"hist_{feature_stem}_count"
    delay_count_column = f"hist_{feature_stem}_delay_count"
    rate_column = f"hist_{feature_stem}_delay_rate"
    joined = featured.join(history, [*group_columns, "flight_date"], "left")
    return (
        joined.withColumn(count_column, F.coalesce(F.col(count_column), F.lit(0.0)))
        .withColumn(
            delay_count_column,
            F.coalesce(F.col(delay_count_column), F.lit(0.0)),
        )
        .withColumn(
            rate_column,
            (
                F.col(delay_count_column)
                + F.lit(float(prior_weight)) * F.col("hist_global_delay_rate")
            )
            / (F.col(count_column) + F.lit(float(prior_weight))),
        )
        .drop(delay_count_column)
    )


def add_leakage_safe_features(
    flights: DataFrame,
    prior_weight: float = 20.0,
    default_rate: float = 0.20,
) -> DataFrame:
    """Add features using outcomes from strictly earlier flight dates only.

    Histories are aggregated to one row per entity and date before the lagged
    cumulative window is applied. Therefore, no flight can use its own label or
    another flight's label from the same date.
    """

    global_history = _global_prior_by_date(flights, default_rate)
    featured = flights.join(F.broadcast(global_history), "flight_date", "left")
    for group_columns, feature_stem in (
        (("carrier",), "carrier"),
        (("route",), "route"),
        (("origin",), "origin"),
        (("dest",), "dest"),
    ):
        history = _group_history_table(
            flights,
            group_columns,
            feature_stem,
        )
        featured = _join_group_history(
            featured,
            history,
            group_columns,
            feature_stem,
            prior_weight,
        )

    full_day_minutes = 24.0 * 60.0
    two_pi = 2.0 * math.pi
    return (
        featured.withColumn("day_of_week_cat", F.col("day_of_week").cast("string"))
        .withColumn("month_cat", F.col("month").cast("string"))
        .withColumn(
            "sched_dep_time_sin",
            F.sin(F.lit(two_pi) * F.col("crs_dep_minutes") / F.lit(full_day_minutes)),
        )
        .withColumn(
            "sched_dep_time_cos",
            F.cos(F.lit(two_pi) * F.col("crs_dep_minutes") / F.lit(full_day_minutes)),
        )
        .withColumn(
            "sched_arr_time_sin",
            F.sin(F.lit(two_pi) * F.col("crs_arr_minutes") / F.lit(full_day_minutes)),
        )
        .withColumn(
            "sched_arr_time_cos",
            F.cos(F.lit(two_pi) * F.col("crs_arr_minutes") / F.lit(full_day_minutes)),
        )
    )


def chronological_partitions(
    featured: DataFrame,
    start_date: date,
    train_end_date: date,
    validation_end_date: date,
    test_end_date: date,
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    start = F.lit(start_date.isoformat()).cast("date")
    train_end = F.lit(train_end_date.isoformat()).cast("date")
    validation_end = F.lit(validation_end_date.isoformat()).cast("date")
    test_end = F.lit(test_end_date.isoformat()).cast("date")

    train = featured.filter(
        (F.col("flight_date") >= start) & (F.col("flight_date") <= train_end)
    )
    validation = featured.filter(
        (F.col("flight_date") > train_end)
        & (F.col("flight_date") <= validation_end)
    )
    test = featured.filter(
        (F.col("flight_date") > validation_end)
        & (F.col("flight_date") <= test_end)
    )
    return train, validation, test


def split_summary_rows(partitions: Dict[str, DataFrame]) -> List[dict]:
    rows: List[dict] = []
    for name, frame in partitions.items():
        row = frame.agg(
            F.count(F.lit(1)).alias("row_count"),
            F.sum("label").alias("delayed_count"),
            F.avg("label").alias("delay_rate"),
            F.min("flight_date").alias("minimum_date"),
            F.max("flight_date").alias("maximum_date"),
        ).first()
        count = int(row["row_count"] or 0)
        if count == 0:
            raise ValueError(f"Chronological partition '{name}' contains no records.")
        delayed = int(row["delayed_count"] or 0)
        rows.append(
            {
                "split": name,
                "minimum_date": row["minimum_date"].isoformat(),
                "maximum_date": row["maximum_date"].isoformat(),
                "row_count": count,
                "delayed_count": delayed,
                "on_time_count": count - delayed,
                "delay_rate": float(row["delay_rate"]),
            }
        )
    return rows


def historical_coverage_rows(partitions: Dict[str, DataFrame]) -> List[dict]:
    count_columns = (
        "hist_carrier_count",
        "hist_route_count",
        "hist_origin_count",
        "hist_dest_count",
    )
    rows: List[dict] = []
    for split_name, frame in partitions.items():
        expressions = [F.count(F.lit(1)).alias("row_count")]
        for column in count_columns:
            expressions.extend(
                [
                    F.avg(F.col(column)).alias(f"{column}__mean"),
                    F.avg((F.col(column) <= 0).cast("double")).alias(
                        f"{column}__zero_share"
                    ),
                ]
            )
        result = frame.agg(*expressions).first().asDict()
        for column in count_columns:
            rows.append(
                {
                    "split": split_name,
                    "history_entity": column.removeprefix("hist_").removesuffix("_count"),
                    "mean_prior_flight_count": float(result[f"{column}__mean"] or 0.0),
                    "share_with_no_entity_history": float(
                        result[f"{column}__zero_share"] or 0.0
                    ),
                }
            )
    return rows
