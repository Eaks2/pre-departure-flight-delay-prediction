"""Input-column resolution, cleaning, and target construction.

Data definitions were checked against the official BTS TranStats table:
https://www.transtats.bts.gov/DL_SelectFields.aspx?QO_fu146_anzr=b0-gvzr&gnoyr_VQ=FGJ
"""

from __future__ import annotations

import re
from datetime import date
from typing import Dict, Iterable, Mapping, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


COLUMN_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "flight_date_raw": ("FlightDate", "FL_DATE"),
    "carrier": ("Reporting_Airline", "UniqueCarrier", "OP_UNIQUE_CARRIER"),
    "flight_number": (
        "Flight_Number_Reporting_Airline",
        "FlightNum",
        "OP_CARRIER_FL_NUM",
    ),
    "origin": ("Origin",),
    "dest": ("Dest", "Destination"),
    "crs_dep_time_raw": ("CRSDepTime", "CRS_DEP_TIME"),
    "crs_arr_time_raw": ("CRSArrTime", "CRS_ARR_TIME"),
    "crs_elapsed_time_raw": ("CRSElapsedTime", "CRS_ELAPSED_TIME"),
    "distance_raw": ("Distance",),
    "arr_delay_raw": ("ArrDelay", "ARR_DELAY"),
    "cancelled_raw": ("Cancelled", "CANCELED"),
    "diverted_raw": ("Diverted",),
}


def _canonical_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def resolve_required_columns(raw: DataFrame) -> DataFrame:
    """Select required fields while tolerating documented legacy aliases."""

    available: Dict[str, str] = {}
    duplicates = set()
    for column in raw.columns:
        canonical = _canonical_header(column)
        if canonical in available:
            duplicates.add(canonical)
        else:
            available[canonical] = column
    if duplicates:
        raise ValueError(
            "Input contains columns that become duplicates after header normalization: "
            + ", ".join(sorted(duplicates))
        )

    resolved = []
    missing = []
    for output_name, aliases in COLUMN_ALIASES.items():
        source_name = next(
            (available[_canonical_header(alias)] for alias in aliases if _canonical_header(alias) in available),
            None,
        )
        if source_name is None:
            missing.append(f"{output_name}: one of {aliases}")
        else:
            resolved.append(F.col(source_name).alias(output_name))

    if missing:
        raise ValueError(
            "Required BTS columns are missing:\n- " + "\n- ".join(missing)
        )
    return raw.select(*resolved)


def _valid_hhmm(column: F.Column) -> F.Column:
    return (
        column.isNotNull()
        & (column >= 0)
        & (column <= 2400)
        & ((column == 2400) | ((column % 100) < 60))
        & ((column == 2400) | (F.floor(column / 100) < 24))
    )


def _minutes_after_midnight(column: F.Column) -> F.Column:
    normalized = F.when(column == 2400, F.lit(0)).otherwise(column)
    return (F.floor(normalized / 100) * 60 + (normalized % 100)).cast("double")


def stage_cleaning(
    raw: DataFrame,
    start_date: date,
    end_date: date,
) -> DataFrame:
    """Cast raw strings and attach one auditable rejection reason per row."""

    selected = resolve_required_columns(raw)
    staged = (
        selected.withColumn(
            "flight_date",
            F.coalesce(
                F.to_date("flight_date_raw", "yyyy-MM-dd"),
                F.to_date("flight_date_raw", "M/d/yyyy"),
                F.to_date("flight_date_raw", "yyyyMMdd"),
            ),
        )
        .withColumn("carrier", F.upper(F.trim("carrier")))
        .withColumn("flight_number", F.trim("flight_number").cast("string"))
        .withColumn("origin", F.upper(F.trim("origin")))
        .withColumn("dest", F.upper(F.trim("dest")))
        .withColumn("crs_dep_time", F.trim("crs_dep_time_raw").cast("int"))
        .withColumn("crs_arr_time", F.trim("crs_arr_time_raw").cast("int"))
        .withColumn(
            "crs_elapsed_time", F.trim("crs_elapsed_time_raw").cast("double")
        )
        .withColumn("distance", F.trim("distance_raw").cast("double"))
        .withColumn("arr_delay", F.trim("arr_delay_raw").cast("double"))
        .withColumn("cancelled", F.trim("cancelled_raw").cast("double"))
        .withColumn("diverted", F.trim("diverted_raw").cast("double"))
    )

    required_text_missing = (
        F.col("carrier").isNull()
        | (F.col("carrier") == "")
        | F.col("flight_number").isNull()
        | (F.col("flight_number") == "")
        | F.col("origin").isNull()
        | (F.col("origin") == "")
        | F.col("dest").isNull()
        | (F.col("dest") == "")
    )

    return staged.withColumn(
        "rejection_reason",
        F.when(F.col("flight_date").isNull(), F.lit("INVALID_FLIGHT_DATE"))
        .when(
            (F.col("flight_date") < F.lit(start_date.isoformat()).cast("date"))
            | (F.col("flight_date") > F.lit(end_date.isoformat()).cast("date")),
            F.lit("OUTSIDE_PROJECT_DATE_RANGE"),
        )
        .when(F.col("cancelled") == 1.0, F.lit("CANCELLED"))
        .when(F.col("diverted") == 1.0, F.lit("DIVERTED"))
        .when(required_text_missing, F.lit("MISSING_IDENTIFIER"))
        .when(
            ~_valid_hhmm(F.col("crs_dep_time"))
            | ~_valid_hhmm(F.col("crs_arr_time")),
            F.lit("INVALID_SCHEDULED_TIME"),
        )
        .when(
            F.col("crs_elapsed_time").isNull()
            | (F.col("crs_elapsed_time") <= 0),
            F.lit("INVALID_SCHEDULED_DURATION"),
        )
        .when(
            F.col("distance").isNull() | (F.col("distance") <= 0),
            F.lit("INVALID_DISTANCE"),
        )
        .when(F.col("arr_delay").isNull(), F.lit("MISSING_ARRIVAL_DELAY"))
        .when(
            F.col("cancelled").isNull() | F.col("diverted").isNull(),
            F.lit("MISSING_STATUS_FLAG"),
        )
        .otherwise(F.lit("ACCEPTED")),
    )


def accepted_flights(staged: DataFrame) -> DataFrame:
    """Return unique, completed flights with the binary delay label."""

    accepted = (
        staged.filter(F.col("rejection_reason") == "ACCEPTED")
        .withColumn("label", (F.col("arr_delay") >= 15.0).cast("double"))
        .withColumn("crs_dep_minutes", _minutes_after_midnight(F.col("crs_dep_time")))
        .withColumn("crs_arr_minutes", _minutes_after_midnight(F.col("crs_arr_time")))
        .withColumn("month", F.month("flight_date").cast("int"))
        .withColumn(
            "day_of_week",
            (F.pmod(F.dayofweek("flight_date") + F.lit(5), F.lit(7)) + F.lit(1)).cast(
                "int"
            ),
        )
        .withColumn("route", F.concat_ws("_", "origin", "dest"))
    )

    identity_columns = [
        "flight_date",
        "carrier",
        "flight_number",
        "origin",
        "dest",
        "crs_dep_time",
    ]
    deduplicated = accepted.dropDuplicates(identity_columns)
    return deduplicated.withColumn(
        "flight_key",
        F.sha2(
            F.concat_ws(
                "|", *[F.coalesce(F.col(name).cast("string"), F.lit("")) for name in identity_columns]
            ),
            256,
        ),
    ).select(
        "flight_key",
        "flight_date",
        "carrier",
        "flight_number",
        "origin",
        "dest",
        "route",
        "crs_dep_time",
        "crs_arr_time",
        "crs_dep_minutes",
        "crs_arr_minutes",
        "crs_elapsed_time",
        "distance",
        "month",
        "day_of_week",
        "arr_delay",
        "label",
    )


def cleaning_quality_rows(staged: DataFrame, accepted: DataFrame) -> Iterable[dict]:
    """Collect the small rejection summary used in later reporting."""

    reason_rows = {
        row["rejection_reason"]: int(row["count"])
        for row in staged.groupBy("rejection_reason").count().collect()
    }
    input_count = sum(reason_rows.values())
    accepted_before_dedupe = reason_rows.get("ACCEPTED", 0)
    accepted_after_dedupe = accepted.count()
    rows = [
        {
            "stage": "input",
            "reason": "ALL_ROWS",
            "row_count": input_count,
        }
    ]
    for reason, count in sorted(reason_rows.items()):
        rows.append({"stage": "cleaning", "reason": reason, "row_count": count})
    rows.extend(
        [
            {
                "stage": "deduplication",
                "reason": "DUPLICATE_ROWS_REMOVED",
                "row_count": accepted_before_dedupe - accepted_after_dedupe,
            },
            {
                "stage": "final_clean_data",
                "reason": "ACCEPTED_UNIQUE_FLIGHTS",
                "row_count": accepted_after_dedupe,
            },
        ]
    )
    return rows

