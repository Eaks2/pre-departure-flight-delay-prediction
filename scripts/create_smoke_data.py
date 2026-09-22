#!/usr/bin/env python3
"""Create a tiny deterministic BTS-shaped CSV for an end-to-end smoke run."""

from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence


HEADERS = [
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


def _period_rows(start: date, days: int, flights_per_day: int, number_start: int):
    carriers = ("AA", "DL", "WN")
    routes = (("SEA", "LAX", 954), ("SEA", "SFO", 679), ("LAX", "DEN", 862))
    number = number_start
    for day_offset in range(days):
        flight_date = start + timedelta(days=day_offset)
        for flight_index in range(flights_per_day):
            carrier = carriers[(day_offset + flight_index) % len(carriers)]
            origin, dest, distance = routes[(day_offset * 2 + flight_index) % len(routes)]
            departure = 600 + (flight_index * 230) % 1500
            departure_hour = departure // 100
            departure_minute = departure % 100
            if departure_minute >= 60:
                departure += 40
            elapsed = 95 + (flight_index % 3) * 25
            arrival_minutes = ((departure // 100) * 60 + departure % 100 + elapsed) % 1440
            arrival = (arrival_minutes // 60) * 100 + arrival_minutes % 60
            delayed = ((day_offset + flight_index * 2) % 5) in (0, 1)
            arr_delay = 25 + (day_offset % 12) if delayed else -5 + (flight_index % 8)
            yield {
                "FlightDate": flight_date.isoformat(),
                "Reporting_Airline": carrier,
                "Flight_Number_Reporting_Airline": str(number),
                "Origin": origin,
                "Dest": dest,
                "CRSDepTime": departure,
                "CRSArrTime": arrival,
                "CRSElapsedTime": elapsed,
                "Distance": distance,
                "ArrDelay": arr_delay,
                "Cancelled": 0,
                "Diverted": 0,
            }
            number += 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/smoke/bts_smoke.csv")
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        *_period_rows(date(2023, 1, 1), 20, 6, 1000),
        *_period_rows(date(2024, 7, 1), 20, 6, 2000),
        *_period_rows(date(2025, 2, 1), 20, 6, 3000),
        *_period_rows(date(2025, 8, 1), 20, 6, 4000),
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} smoke-test rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

