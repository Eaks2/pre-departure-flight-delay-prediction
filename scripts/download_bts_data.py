#!/usr/bin/env python3
"""Download and extract official BTS Reporting Carrier monthly CSV archives.

Official source consulted:
https://www.transtats.bts.gov/PREZIP/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence


BASE_URL = "https://www.transtats.bts.gov/PREZIP"
ARCHIVE_TEMPLATE = (
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)
REQUIRED_HEADERS = {
    "flightdate",
    "reportingairline",
    "flightnumberreportingairline",
    "origin",
    "dest",
    "crsdeptime",
    "crsarrtime",
    "crselapsedtime",
    "distance",
    "arrdelay",
    "cancelled",
    "diverted",
}
PROJECT_COLUMNS = (
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
)


def _canonical(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path, retries: int) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CS777-Term-Project-Downloader/1.0"},
    )
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response, partial.open(
                "wb"
            ) as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            if partial.stat().st_size == 0:
                raise OSError("Downloaded file is empty.")
            partial.replace(destination)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if partial.exists():
                partial.unlink()
            if attempt == retries:
                raise RuntimeError(f"Download failed after {retries} attempts: {url}") from exc
            wait_seconds = min(2**attempt, 15)
            print(f"  attempt {attempt} failed; retrying in {wait_seconds}s", flush=True)
            time.sleep(wait_seconds)


def _extract_required_columns(
    archive: Path, csv_directory: Path
) -> Sequence[tuple[Path, Optional[int]]]:
    """Stream only the proposal-required fields from each archived CSV."""

    extracted = []
    with zipfile.ZipFile(archive) as bundle:
        csv_members = [
            member
            for member in bundle.infolist()
            if not member.is_dir() and member.filename.lower().endswith(".csv")
        ]
        if not csv_members:
            raise RuntimeError(f"No CSV file was found in {archive.name}.")
        for member in csv_members:
            target = csv_directory / Path(member.filename).name
            resolved_target = target.resolve()
            if csv_directory.resolve() not in resolved_target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
            if target.exists():
                _validate_headers(target)
                extracted.append((target, None))
                continue

            partial = target.with_suffix(target.suffix + ".part")
            try:
                with bundle.open(member) as binary_source, io.TextIOWrapper(
                    binary_source, encoding="utf-8-sig", newline=""
                ) as text_source, partial.open(
                    "w", encoding="utf-8", newline=""
                ) as output:
                    reader = csv.DictReader(text_source)
                    source_headers = reader.fieldnames or []
                    by_canonical = {_canonical(name): name for name in source_headers}
                    missing = [
                        name
                        for name in PROJECT_COLUMNS
                        if _canonical(name) not in by_canonical
                    ]
                    if missing:
                        raise RuntimeError(
                            f"{member.filename} is missing required fields: "
                            + ", ".join(missing)
                        )
                    writer = csv.DictWriter(output, fieldnames=PROJECT_COLUMNS)
                    writer.writeheader()
                    for row in reader:
                        writer.writerow(
                            {
                                output_name: row[by_canonical[_canonical(output_name)]]
                                for output_name in PROJECT_COLUMNS
                            }
                        )
                partial.replace(target)
            except Exception:
                if partial.exists():
                    partial.unlink()
                raise
            extracted.append((target, len(source_headers)))
    return extracted


def _validate_headers(csv_path: Path) -> Sequence[str]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            headers = next(reader)
        except StopIteration as exc:
            raise RuntimeError(f"CSV is empty: {csv_path}") from exc
    canonical_headers = {_canonical(header) for header in headers}
    missing = sorted(REQUIRED_HEADERS - canonical_headers)
    if missing:
        raise RuntimeError(
            f"{csv_path.name} is missing required headers: {', '.join(missing)}"
        )
    return headers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download official monthly BTS Reporting Carrier CSV archives."
    )
    parser.add_argument("--start-year", type=int, default=2023)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--output-dir", default="data/bts_reporting_carrier_2023_2025")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--delete-archives-after-extract",
        action="store_true",
        help="Delete ZIP archives only after their CSV files pass header validation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the 36 expected URLs without downloading them.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start_year > args.end_year:
        raise SystemExit("--start-year cannot be later than --end-year.")
    if args.start_year < 1987:
        raise SystemExit("The Reporting Carrier series begins in 1987.")

    output_root = Path(args.output_dir).expanduser().resolve()
    archive_directory = output_root / "archives"
    csv_directory = output_root / "csv"
    archive_directory.mkdir(parents=True, exist_ok=True)
    csv_directory.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    for year in range(args.start_year, args.end_year + 1):
        for month in range(1, 13):
            filename = ARCHIVE_TEMPLATE.format(year=year, month=month)
            url = f"{BASE_URL}/{filename}"
            if args.dry_run:
                print(url)
                continue
            archive_path = archive_directory / filename
            print(f"[{year}-{month:02d}] {filename}", flush=True)
            if not archive_path.exists():
                _download(url, archive_path, args.retries)
            else:
                print("  archive already exists; download skipped", flush=True)

            extracted_paths = _extract_required_columns(archive_path, csv_directory)
            csv_entries = []
            for csv_path, source_column_count in extracted_paths:
                headers = _validate_headers(csv_path)
                csv_entries.append(
                    {
                        "path": str(csv_path.relative_to(output_root)),
                        "bytes": csv_path.stat().st_size,
                        "column_count": len(headers),
                        "source_column_count": source_column_count,
                    }
                )
            manifest_entries.append(
                {
                    "year": year,
                    "month": month,
                    "source_url": url,
                    "archive_name": filename,
                    "archive_bytes": archive_path.stat().st_size,
                    "archive_sha256": _sha256(archive_path),
                    "csv_files": csv_entries,
                }
            )
            if args.delete_archives_after_extract:
                archive_path.unlink()

    if not args.dry_run:
        manifest = {
            "dataset": "BTS Reporting Carrier On-Time Performance (1987-present)",
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "start_year": args.start_year,
            "end_year": args.end_year,
            "expected_month_count": (args.end_year - args.start_year + 1) * 12,
            "files": manifest_entries,
        }
        manifest_path = output_root / "download_manifest.json"
        temporary_path = manifest_path.with_suffix(".json.part")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
        temporary_path.replace(manifest_path)
        print(f"Complete: {len(manifest_entries)} monthly archives documented.")
        print(f"CSV input directory: {csv_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
