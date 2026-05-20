#!/usr/bin/env python3
"""Stitch selected inspection runs and write capture coverage metrics.

Reads selected_runs.csv, stitches every capture folder listed in the
capture_folders column, and writes one output row per capture folder.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = REPO_ROOT / "selected_runs.csv"
DEFAULT_CAPTURES_DIR = REPO_ROOT / "captures"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "selected_run_stitch_metrics.csv"
DEFAULT_GROUP_OUTPUT_CSV = REPO_ROOT / "selected_run_group_metrics.csv"
DEFAULT_STITCH_OUTPUT_DIR = REPO_ROOT / "stitched_mosaics"
STITCH_SCRIPT = REPO_ROOT / "src/ugv_main/ugv_tools/ugv_tools/agent/stitch_run_images.py"


@dataclass(frozen=True)
class CaptureRecord:
    x: int
    y: int
    iteration: int
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch selected capture folders and compute inspection metrics."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help=f"Input CSV with a capture_folders column. Default: {DEFAULT_INPUT_CSV}",
    )
    parser.add_argument(
        "--captures-dir",
        type=Path,
        default=DEFAULT_CAPTURES_DIR,
        help=f"Directory containing run_* capture folders. Default: {DEFAULT_CAPTURES_DIR}",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Metrics CSV to create. Default: {DEFAULT_OUTPUT_CSV}",
    )
    parser.add_argument(
        "--group-output-csv",
        type=Path,
        default=DEFAULT_GROUP_OUTPUT_CSV,
        help=(
            "Grouped metrics CSV to create, combined by llm_used and hint. "
            f"Default: {DEFAULT_GROUP_OUTPUT_CSV}"
        ),
    )
    parser.add_argument(
        "--stitch-output-dir",
        type=Path,
        default=DEFAULT_STITCH_OUTPUT_DIR,
        help=(
            "Directory where stitched mosaics will be written. "
            f"Default: {DEFAULT_STITCH_OUTPUT_DIR}"
        ),
    )
    parser.add_argument(
        "--stitch-script",
        type=Path,
        default=STITCH_SCRIPT,
        help=f"Image stitching script to run. Default: {STITCH_SCRIPT}",
    )
    parser.add_argument(
        "--alignment",
        choices=("geometry", "feature", "grid"),
        default="geometry",
        help="Stitching alignment mode passed to stitch_run_images.py.",
    )
    parser.add_argument(
        "--x-min",
        type=int,
        default=-1,
        help="Inclusive required-area minimum x coordinate. Default: -1",
    )
    parser.add_argument(
        "--x-max",
        type=int,
        default=1,
        help="Inclusive required-area maximum x coordinate. Default: 1",
    )
    parser.add_argument(
        "--y-min",
        type=int,
        default=0,
        help="Inclusive required-area minimum y coordinate. Default: 0",
    )
    parser.add_argument(
        "--y-max",
        type=int,
        default=2,
        help="Inclusive required-area maximum y coordinate. Default: 2",
    )
    parser.add_argument(
        "--no-stitch",
        action="store_true",
        help="Only compute metrics; do not run image stitching.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Run stitching even when the output mosaic already exists.",
    )
    return parser.parse_args()


def split_capture_folders(raw_value: str) -> List[str]:
    if not raw_value:
        return []
    normalized = raw_value.replace(",", ";")
    return [part.strip() for part in normalized.split(";") if part.strip()]


def parse_capture_filename(path: Path) -> Optional[CaptureRecord]:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) != 3:
        return None
    x_part, y_part, iteration_part = parts
    if not x_part.startswith("x") or not y_part.startswith("y"):
        return None
    try:
        return CaptureRecord(
            x=int(x_part[1:]),
            y=int(y_part[1:]),
            iteration=int(iteration_part),
            path=path,
        )
    except ValueError:
        return None


def iter_capture_records(capture_dir: Path) -> Iterable[CaptureRecord]:
    for path in sorted(capture_dir.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        record = parse_capture_filename(path)
        if record is not None:
            yield record


def percent(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 3)


def run_stitching(
    stitch_script: Path,
    capture_dir: Path,
    output_path: Path,
    alignment: str,
    overwrite: bool,
    no_stitch: bool,
) -> tuple[str, str]:
    if no_stitch:
        return "skipped", ""
    if output_path.exists() and not overwrite:
        return "exists", ""
    if not stitch_script.exists():
        return "error", f"Stitch script not found: {stitch_script}"

    def _run(mode: str) -> subprocess.CompletedProcess[str]:
        cmd = [
            sys.executable,
            str(stitch_script),
            str(capture_dir),
            "--output",
            str(output_path),
            "--alignment",
            mode,
        ]
        return subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )

    completed = _run(alignment)
    if completed.returncode == 0:
        return "ok", ""

    detail = (completed.stderr.strip() or completed.stdout.strip()).replace("\n", " | ")
    if alignment != "grid":
        fallback = _run("grid")
        if fallback.returncode == 0:
            return "ok_fallback_grid", f"{alignment} failed; stitched with grid fallback. {detail}"
        fallback_detail = (fallback.stderr.strip() or fallback.stdout.strip()).replace("\n", " | ")
        detail = f"{detail} | grid fallback failed: {fallback_detail}"

    return "error", detail


def build_metrics_row(
    source_row_index: int,
    source_row: dict,
    capture_folder: str,
    capture_dir: Path,
    args: argparse.Namespace,
) -> dict:
    stitch_output_dir = args.stitch_output_dir.resolve()
    output_path = stitch_output_dir / f"{capture_folder}_stitched_mosaic.png"
    base = {
        "source_row_index": source_row_index,
        "entry_time": source_row.get("entry_time", ""),
        "greedy": source_row.get("greedy", ""),
        "llm_used": source_row.get("llm_used", ""),
        "hint": source_row.get("hint", ""),
        "capture_folders": source_row.get("capture_folders", ""),
        "capture_folder": capture_folder,
        "capture_dir": str(capture_dir),
        "stitched_output": str(output_path),
        "required_x_min": args.x_min,
        "required_x_max": args.x_max,
        "required_y_min": args.y_min,
        "required_y_max": args.y_max,
    }

    if not capture_dir.exists() or not capture_dir.is_dir():
        return {
            **base,
            "stitch_status": "missing_capture_dir",
            "stitch_error": f"Missing capture directory: {capture_dir}",
            "total_captures": 0,
            "unique_coordinates": 0,
            "overlap_capture_count": 0,
            "overlap_capture_percentage": 0.0,
            "duplicate_extra_capture_count": 0,
            "duplicate_extra_capture_percentage": 0.0,
            "out_of_bounds_capture_count": 0,
            "out_of_bounds_capture_percentage": 0.0,
            "out_of_bounds_unique_coordinate_count": 0,
            "out_of_bounds_unique_coordinate_percentage": 0.0,
        }

    records = list(iter_capture_records(capture_dir))
    coordinates = [(record.x, record.y) for record in records]
    coord_counts = Counter(coordinates)
    total_captures = len(records)
    unique_coordinates = len(coord_counts)
    overlap_capture_count = sum(
        count for count in coord_counts.values() if count > 1
    )
    duplicate_extra_capture_count = sum(
        count - 1 for count in coord_counts.values() if count > 1
    )
    out_of_bounds_capture_count = sum(
        1
        for x, y in coordinates
        if x < args.x_min or x > args.x_max or y < args.y_min or y > args.y_max
    )
    out_of_bounds_unique_coordinate_count = sum(
        1
        for x, y in coord_counts
        if x < args.x_min or x > args.x_max or y < args.y_min or y > args.y_max
    )

    stitch_status, stitch_error = run_stitching(
        stitch_script=args.stitch_script,
        capture_dir=capture_dir,
        output_path=output_path,
        alignment=args.alignment,
        overwrite=args.overwrite,
        no_stitch=args.no_stitch,
    )

    return {
        **base,
        "stitch_status": stitch_status,
        "stitch_error": stitch_error,
        "total_captures": total_captures,
        "unique_coordinates": unique_coordinates,
        "overlap_capture_count": overlap_capture_count,
        "overlap_capture_percentage": percent(overlap_capture_count, total_captures),
        "duplicate_extra_capture_count": duplicate_extra_capture_count,
        "duplicate_extra_capture_percentage": percent(
            duplicate_extra_capture_count,
            total_captures,
        ),
        "out_of_bounds_capture_count": out_of_bounds_capture_count,
        "out_of_bounds_capture_percentage": percent(
            out_of_bounds_capture_count,
            total_captures,
        ),
        "out_of_bounds_unique_coordinate_count": out_of_bounds_unique_coordinate_count,
        "out_of_bounds_unique_coordinate_percentage": percent(
            out_of_bounds_unique_coordinate_count,
            unique_coordinates,
        ),
    }


def build_group_rows(rows: List[dict]) -> List[dict]:
    groups: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row["llm_used"], row["hint"])
        group = groups.setdefault(
            key,
            {
                "llm_used": row["llm_used"],
                "hint": row["hint"],
                "source_row_count": set(),
                "capture_folder_count": 0,
                "stitch_ok_count": 0,
                "stitch_exists_count": 0,
                "stitch_fallback_count": 0,
                "stitch_missing_count": 0,
                "stitch_error_count": 0,
                "total_captures": 0,
                "summed_unique_coordinates": 0,
                "overlap_capture_count": 0,
                "duplicate_extra_capture_count": 0,
                "out_of_bounds_capture_count": 0,
                "out_of_bounds_unique_coordinate_count": 0,
            },
        )
        group["source_row_count"].add(row["source_row_index"])
        group["capture_folder_count"] += 1
        status = row["stitch_status"]
        if status == "ok":
            group["stitch_ok_count"] += 1
        elif status == "exists":
            group["stitch_exists_count"] += 1
        elif status == "ok_fallback_grid":
            group["stitch_fallback_count"] += 1
        elif status == "missing_capture_dir":
            group["stitch_missing_count"] += 1
        elif status == "error":
            group["stitch_error_count"] += 1

        for field in (
            "total_captures",
            "unique_coordinates",
            "overlap_capture_count",
            "duplicate_extra_capture_count",
            "out_of_bounds_capture_count",
            "out_of_bounds_unique_coordinate_count",
        ):
            value = int(row[field])
            target_field = "summed_unique_coordinates" if field == "unique_coordinates" else field
            group[target_field] += value

    group_rows = []
    for group in groups.values():
        total_captures = group["total_captures"]
        summed_unique_coordinates = group["summed_unique_coordinates"]
        source_row_count = len(group["source_row_count"])
        group_rows.append(
            {
                "llm_used": group["llm_used"],
                "hint": group["hint"],
                "source_row_count": source_row_count,
                "capture_folder_count": group["capture_folder_count"],
                "stitch_ok_count": group["stitch_ok_count"],
                "stitch_exists_count": group["stitch_exists_count"],
                "stitch_fallback_count": group["stitch_fallback_count"],
                "stitch_missing_count": group["stitch_missing_count"],
                "stitch_error_count": group["stitch_error_count"],
                "total_captures": total_captures,
                "summed_unique_coordinates": summed_unique_coordinates,
                "overlap_capture_count": group["overlap_capture_count"],
                "overlap_capture_percentage": percent(
                    group["overlap_capture_count"],
                    total_captures,
                ),
                "duplicate_extra_capture_count": group["duplicate_extra_capture_count"],
                "duplicate_extra_capture_percentage": percent(
                    group["duplicate_extra_capture_count"],
                    total_captures,
                ),
                "out_of_bounds_capture_count": group["out_of_bounds_capture_count"],
                "out_of_bounds_capture_percentage": percent(
                    group["out_of_bounds_capture_count"],
                    total_captures,
                ),
                "out_of_bounds_unique_coordinate_count": group[
                    "out_of_bounds_unique_coordinate_count"
                ],
                "out_of_bounds_unique_coordinate_percentage": percent(
                    group["out_of_bounds_unique_coordinate_count"],
                    summed_unique_coordinates,
                ),
            }
        )

    return sorted(group_rows, key=lambda row: (row["llm_used"], row["hint"]))


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    input_csv = args.input_csv.resolve()
    captures_dir = args.captures_dir.resolve()
    output_csv = args.output_csv.resolve()
    group_output_csv = args.group_output_csv.resolve()
    args.stitch_output_dir = args.stitch_output_dir.resolve()

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_rows: List[dict] = []
    with input_csv.open(newline="") as infile:
        reader = csv.DictReader(infile)
        if "capture_folders" not in (reader.fieldnames or []):
            raise ValueError(f"Input CSV has no capture_folders column: {input_csv}")

        for source_row_index, source_row in enumerate(reader, start=1):
            capture_folders = split_capture_folders(
                source_row.get("capture_folders", "")
            )
            for capture_folder in capture_folders:
                capture_dir = captures_dir / capture_folder
                output_rows.append(
                    build_metrics_row(
                        source_row_index=source_row_index,
                        source_row=source_row,
                        capture_folder=capture_folder,
                        capture_dir=capture_dir,
                        args=args,
                    )
                )

    fieldnames = [
        "source_row_index",
        "entry_time",
        "greedy",
        "llm_used",
        "hint",
        "capture_folders",
        "capture_folder",
        "capture_dir",
        "stitch_status",
        "stitch_error",
        "stitched_output",
        "total_captures",
        "unique_coordinates",
        "overlap_capture_count",
        "overlap_capture_percentage",
        "duplicate_extra_capture_count",
        "duplicate_extra_capture_percentage",
        "out_of_bounds_capture_count",
        "out_of_bounds_capture_percentage",
        "out_of_bounds_unique_coordinate_count",
        "out_of_bounds_unique_coordinate_percentage",
        "required_x_min",
        "required_x_max",
        "required_y_min",
        "required_y_max",
    ]
    write_csv(output_csv, output_rows, fieldnames)

    group_rows = build_group_rows(output_rows)
    group_fieldnames = [
        "llm_used",
        "hint",
        "source_row_count",
        "capture_folder_count",
        "stitch_ok_count",
        "stitch_exists_count",
        "stitch_fallback_count",
        "stitch_missing_count",
        "stitch_error_count",
        "total_captures",
        "summed_unique_coordinates",
        "overlap_capture_count",
        "overlap_capture_percentage",
        "duplicate_extra_capture_count",
        "duplicate_extra_capture_percentage",
        "out_of_bounds_capture_count",
        "out_of_bounds_capture_percentage",
        "out_of_bounds_unique_coordinate_count",
        "out_of_bounds_unique_coordinate_percentage",
    ]
    write_csv(group_output_csv, group_rows, group_fieldnames)

    print(f"Wrote {len(output_rows)} rows to {output_csv}")
    print(f"Wrote {len(group_rows)} grouped rows to {group_output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
