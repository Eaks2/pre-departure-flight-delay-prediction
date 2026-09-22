#!/usr/bin/env python3
"""Stable spark-submit entry point for the CS777 project."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from flight_delay.main import main


if __name__ == "__main__":
    raise SystemExit(main())

