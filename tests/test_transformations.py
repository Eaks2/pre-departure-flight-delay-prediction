"""Small local tests; these never execute the three-year pipeline."""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from pyspark.sql import SparkSession

    from flight_delay.cleaning import accepted_flights, stage_cleaning
    from flight_delay.features import add_leakage_safe_features, chronological_partitions
    from flight_delay.main import build_argument_parser

    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False


@unittest.skipUnless(PYSPARK_AVAILABLE, "PySpark is not installed")
class TransformationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("CS777_transformation_tests")
            .config("spark.ui.enabled", "false")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def test_cleaning_exclusions_and_15_minute_boundary(self) -> None:
        columns = [
            "FlightDate",
            "Reporting_Airline",
            "Flight_Number_Reporting_Airline",
            "Origin",
            "Dest",
            "CRSDepTime",
            "CRSArrTime",
            "CRSElapsedTime",
            "Distance",
            "ArrDelay",
            "Cancelled",
            "Diverted",
        ]
        rows = [
            ("2023-01-01", "AA", "1", "SEA", "LAX", "800", "1030", "150", "954", "14", "0", "0"),
            ("2023-01-01", "AA", "2", "SEA", "LAX", "900", "1130", "150", "954", "15", "0", "0"),
            ("2023-01-01", "AA", "3", "SEA", "LAX", "1000", "1230", "150", "954", "", "1", "0"),
            ("2023-01-01", "AA", "4", "SEA", "LAX", "1100", "1330", "150", "954", "", "0", "1"),
        ]
        raw = self.spark.createDataFrame(rows, columns)
        staged = stage_cleaning(raw, date(2023, 1, 1), date(2025, 12, 31))
        clean = accepted_flights(staged)
        labels = [int(row["label"]) for row in clean.orderBy("flight_number").collect()]
        self.assertEqual(labels, [0, 1])
        reasons = {
            row["rejection_reason"]: row["count"]
            for row in staged.groupBy("rejection_reason").count().collect()
        }
        self.assertEqual(reasons["CANCELLED"], 1)
        self.assertEqual(reasons["DIVERTED"], 1)

    def test_history_uses_strictly_earlier_dates(self) -> None:
        columns = [
            "flight_key",
            "flight_date",
            "carrier",
            "origin",
            "dest",
            "route",
            "crs_dep_minutes",
            "crs_arr_minutes",
            "crs_elapsed_time",
            "distance",
            "month",
            "day_of_week",
            "label",
        ]
        rows = [
            ("a", date(2023, 1, 1), "AA", "SEA", "LAX", "SEA_LAX", 480.0, 630.0, 150.0, 954.0, 1, 7, 0.0),
            ("b", date(2023, 1, 1), "AA", "SEA", "LAX", "SEA_LAX", 540.0, 690.0, 150.0, 954.0, 1, 7, 1.0),
            ("c", date(2023, 1, 2), "AA", "SEA", "LAX", "SEA_LAX", 480.0, 630.0, 150.0, 954.0, 1, 1, 1.0),
            ("d", date(2023, 1, 2), "AA", "SEA", "LAX", "SEA_LAX", 540.0, 690.0, 150.0, 954.0, 1, 1, 1.0),
        ]
        base = self.spark.createDataFrame(rows, columns)
        featured = add_leakage_safe_features(base, prior_weight=20.0, default_rate=0.2)
        result = {row["flight_key"]: row.asDict() for row in featured.collect()}
        self.assertEqual(result["a"]["hist_carrier_count"], 0)
        self.assertEqual(result["b"]["hist_carrier_count"], 0)
        self.assertAlmostEqual(result["a"]["hist_carrier_delay_rate"], 0.2, places=8)
        self.assertAlmostEqual(result["b"]["hist_carrier_delay_rate"], 0.2, places=8)
        self.assertEqual(result["a"]["hist_global_count"], 0)
        self.assertEqual(result["b"]["hist_global_count"], 0)
        self.assertEqual(result["c"]["hist_carrier_count"], 2)
        self.assertEqual(result["d"]["hist_carrier_count"], 2)
        self.assertEqual(result["c"]["hist_global_count"], 2)
        self.assertEqual(result["d"]["hist_global_count"], 2)
        self.assertAlmostEqual(result["c"]["hist_global_delay_rate"], 0.5, places=8)
        self.assertAlmostEqual(result["d"]["hist_global_delay_rate"], 0.5, places=8)
        self.assertAlmostEqual(result["c"]["hist_carrier_delay_rate"], 0.5, places=8)
        self.assertAlmostEqual(result["d"]["hist_carrier_delay_rate"], 0.5, places=8)
        plan = featured._jdf.queryExecution().executedPlan().toString()
        self.assertNotIn("SinglePartition", plan)

    def test_model_serialization_is_opt_in(self) -> None:
        parser = build_argument_parser()
        default_args = parser.parse_args(["--input", "data/*.csv"])
        enabled_args = parser.parse_args(
            ["--input", "data/*.csv", "--save-models"]
        )
        self.assertFalse(default_args.save_models)
        self.assertTrue(enabled_args.save_models)

    def test_chronological_partitions_do_not_overlap(self) -> None:
        base = self.spark.createDataFrame(
            [
                (date(2024, 12, 31), "train"),
                (date(2025, 1, 1), "validation"),
                (date(2025, 6, 30), "validation"),
                (date(2025, 7, 1), "test"),
                (date(2025, 12, 31), "test"),
            ],
            ["flight_date", "expected"],
        )
        train, validation, test = chronological_partitions(
            base,
            date(2023, 1, 1),
            date(2024, 12, 31),
            date(2025, 6, 30),
            date(2025, 12, 31),
        )
        self.assertEqual(train.count(), 1)
        self.assertEqual(validation.count(), 2)
        self.assertEqual(test.count(), 2)


if __name__ == "__main__":
    unittest.main()
