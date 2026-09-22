"""Configuration shared by the flight-delay pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence


DEFAULT_CATEGORICAL_FEATURES = (
    "carrier",
    "origin",
    "dest",
    "route",
    "day_of_week_cat",
    "month_cat",
)

DEFAULT_NUMERIC_FEATURES = (
    "crs_elapsed_time",
    "distance",
    "sched_dep_time_sin",
    "sched_dep_time_cos",
    "sched_arr_time_sin",
    "sched_arr_time_cos",
    "hist_carrier_delay_rate",
    "hist_route_delay_rate",
    "hist_origin_delay_rate",
    "hist_dest_delay_rate",
)

HISTORICAL_COUNT_COLUMNS = (
    "hist_carrier_count",
    "hist_route_count",
    "hist_origin_count",
    "hist_dest_count",
)


@dataclass(frozen=True)
class ProjectConfig:
    """Reproducible defaults derived from the submitted proposal."""

    start_date: date = date(2023, 1, 1)
    train_end_date: date = date(2024, 12, 31)
    validation_end_date: date = date(2025, 6, 30)
    test_end_date: date = date(2025, 12, 31)
    historical_prior_weight: float = 20.0
    historical_default_delay_rate: float = 0.20
    regularization_grid: Sequence[float] = (0.0, 0.001, 0.01, 0.1)
    threshold_grid: Sequence[float] = (
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    )
    max_iterations: int = 100

    def validate(self) -> None:
        if not (
            self.start_date
            <= self.train_end_date
            < self.validation_end_date
            < self.test_end_date
        ):
            raise ValueError(
                "Dates must be ordered as start <= train end < validation end < test end."
            )
        if self.historical_prior_weight <= 0:
            raise ValueError("historical_prior_weight must be positive.")
        if not 0.0 <= self.historical_default_delay_rate <= 1.0:
            raise ValueError("historical_default_delay_rate must be in [0, 1].")
        if not self.regularization_grid:
            raise ValueError("At least one regularization value is required.")
        if any(value < 0 for value in self.regularization_grid):
            raise ValueError("Regularization values cannot be negative.")
        if not self.threshold_grid:
            raise ValueError("At least one classification threshold is required.")
        if any(not 0.0 < value < 1.0 for value in self.threshold_grid):
            raise ValueError("Classification thresholds must be between 0 and 1.")

