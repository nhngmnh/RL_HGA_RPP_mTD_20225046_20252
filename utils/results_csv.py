from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


_FIELDNAMES = [
    "STT",
    "Althgorithm",
    "Datasetname",
    "num_trucks",
    "drones_per_truck",
    "makespan (hours)",
    "Fitness",
    "run time",
    "Service seq",
    "Vehicle asgn",
]


def _safe_max_stt(rows: Iterable[list[str]]) -> int:
    max_stt = 0
    for row in rows:
        if not row:
            continue
        try:
            stt = int(str(row[0]).strip())
        except Exception:
            continue
        max_stt = max(max_stt, stt)
    return max_stt


def append_result(
    csv_path: str | Path,
    *,
    algorithm: str,
    datasetname: str,
    num_trucks: int,
    drones_per_truck: int,
    makespan_hours: float,
    fitness: float,
    runtime_seconds: float,
    service_seq: Any,
    vehicle_asgn: Any,
) -> Path:
    """Append one run result row to results.csv.

    - Creates the file (and writes header) if missing.
    - Appends a new row (no overwrite).
    - Auto-increments STT based on existing rows.
    """

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists()
    next_stt = 1

    existing_fieldnames: list[str] | None = None
    existing_rows: list[dict[str, str]] = []

    if file_exists:
        with path.open("r", newline="", encoding="utf-8") as f:
            sniffer_sample = f.read(4096)
            f.seek(0)
            has_header = sniffer_sample.lstrip().startswith("STT")
            if has_header:
                reader = csv.DictReader(f)
                existing_fieldnames = list(reader.fieldnames or [])
                existing_rows = [dict(r) for r in reader]
                next_stt = _safe_max_stt([[r.get("STT", "")] for r in existing_rows]) + 1
            else:
                # Fallback for a malformed/old file without header
                reader = csv.reader(f)
                rows = list(reader)
                next_stt = _safe_max_stt(rows) + 1

    # If the file exists but header is missing new columns, rewrite it with the new schema
    if file_exists and existing_fieldnames is not None:
        missing = [c for c in _FIELDNAMES if c not in existing_fieldnames]
        if missing:
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
                writer.writeheader()
                for r in existing_rows:
                    upgraded = {k: r.get(k, "") for k in _FIELDNAMES}
                    writer.writerow(upgraded)

    needs_header = (not file_exists) or path.stat().st_size == 0

    row = {
        "STT": next_stt,
        "Althgorithm": algorithm,
        "Datasetname": datasetname,
        "num_trucks": int(num_trucks),
        "drones_per_truck": int(drones_per_truck),
        "makespan (hours)": float(makespan_hours),
        "Fitness": float(fitness),
        "run time": float(runtime_seconds),
        "Service seq": json.dumps(service_seq, ensure_ascii=False),
        "Vehicle asgn": json.dumps(vehicle_asgn, ensure_ascii=False),
    }

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)

    return path
