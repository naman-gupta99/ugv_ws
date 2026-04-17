#!/usr/bin/env python3
"""Stitch inspection run images into a single mosaic based on grid coordinates.

Filename format expected:
    x{x_coordinate}_y{y_coordinate}_{iteration}.png

Example:
    x-1_y2_17.png

Default layout uses Cartesian view:
- x increases left to right
- y increases bottom to top
"""

from __future__ import annotations

import argparse
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

FILENAME_RE = re.compile(r"^x(-?\d+)_y(-?\d+)_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)


@dataclass(frozen=True)
class TileRecord:
    x: int
    y: int
    iteration: int
    path: Path


@dataclass(frozen=True)
class OverlapEstimate:
    pixels: int
    confidence: float
    samples: int


def parse_tile_record(path: Path) -> Optional[TileRecord]:
    match = FILENAME_RE.match(path.name)
    if not match:
        return None
    x_str, y_str, iteration_str, _ext = match.groups()
    return TileRecord(x=int(x_str), y=int(y_str), iteration=int(iteration_str), path=path)


def collect_tile_records(run_dir: Path) -> Tuple[Dict[Tuple[int, int], TileRecord], List[Path], List[TileRecord]]:
    coord_to_record: Dict[Tuple[int, int], TileRecord] = {}
    skipped: List[Path] = []
    discarded: List[TileRecord] = []

    for path in sorted(run_dir.iterdir()):
        if not path.is_file():
            continue
        parsed = parse_tile_record(path)
        if parsed is None:
            skipped.append(path)
            continue

        key = (parsed.x, parsed.y)
        existing = coord_to_record.get(key)
        if existing is None or parsed.iteration > existing.iteration:
            if existing is not None:
                discarded.append(existing)
            coord_to_record[key] = parsed
        else:
            discarded.append(parsed)

    return coord_to_record, skipped, discarded


def read_tiles(coord_to_record: Dict[Tuple[int, int], TileRecord]) -> Tuple[Dict[Tuple[int, int], np.ndarray], Tuple[int, int, int]]:
    images: Dict[Tuple[int, int], np.ndarray] = {}
    expected_shape: Optional[Tuple[int, int, int]] = None

    for coord, record in coord_to_record.items():
        image = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {record.path}")

        if expected_shape is None:
            expected_shape = image.shape
        elif image.shape != expected_shape:
            raise ValueError(
                "Inconsistent image size/channels. "
                f"Expected {expected_shape}, got {image.shape} at {record.path}"
            )

        images[coord] = image

    if expected_shape is None:
        raise ValueError("No valid images found to stitch.")

    return images, expected_shape


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("NCC inputs must have identical shape")

    a_f = a.astype(np.float32).reshape(-1)
    b_f = b.astype(np.float32).reshape(-1)

    a_f -= float(np.mean(a_f))
    b_f -= float(np.mean(b_f))

    denom = float(np.linalg.norm(a_f) * np.linalg.norm(b_f))
    if denom <= 1e-9:
        return -1.0
    return float(np.dot(a_f, b_f) / denom)


def estimate_axis_overlap(
    coord_to_gray: Dict[Tuple[int, int], np.ndarray],
    max_candidate: int,
    horizontal: bool,
    min_fraction: float = 0.05,
    max_fraction: float = 0.6,
    confidence_threshold: float = 0.20,
) -> OverlapEstimate:
    candidates: List[Tuple[int, float]] = []

    min_overlap = max(4, int(round(max_candidate * min_fraction)))
    max_overlap = max(min_overlap, int(round(max_candidate * max_fraction)))
    max_overlap = min(max_overlap, max_candidate - 1)

    if max_overlap < min_overlap:
        return OverlapEstimate(pixels=0, confidence=0.0, samples=0)

    for (x, y), base in coord_to_gray.items():
        neighbor_coord = (x + 1, y) if horizontal else (x, y + 1)
        neighbor = coord_to_gray.get(neighbor_coord)
        if neighbor is None:
            continue

        best_score = -1.0
        best_overlap = 0

        for overlap in range(min_overlap, max_overlap + 1):
            if horizontal:
                strip_a = base[:, -overlap:]
                strip_b = neighbor[:, :overlap]
            else:
                # Positive y means tile is above in Cartesian layout.
                strip_a = base[:overlap, :]
                strip_b = neighbor[-overlap:, :]

            score = normalized_cross_correlation(strip_a, strip_b)
            if score > best_score:
                best_score = score
                best_overlap = overlap

        if best_overlap > 0:
            candidates.append((best_overlap, best_score))

    if not candidates:
        return OverlapEstimate(pixels=0, confidence=0.0, samples=0)

    overlaps = [ov for ov, _score in candidates]
    scores = [score for _ov, score in candidates]
    median_overlap = int(round(statistics.median(overlaps)))
    median_confidence = float(statistics.median(scores))

    if median_confidence < confidence_threshold:
        return OverlapEstimate(pixels=0, confidence=median_confidence, samples=len(candidates))

    return OverlapEstimate(
        pixels=max(0, median_overlap),
        confidence=median_confidence,
        samples=len(candidates),
    )


def stitch_tiles(
    images: Dict[Tuple[int, int], np.ndarray],
    records: Dict[Tuple[int, int], TileRecord],
    width: int,
    height: int,
    x_overlap: int,
    y_overlap: int,
) -> np.ndarray:
    if x_overlap >= width or y_overlap >= height:
        raise ValueError("Overlap must be smaller than tile dimensions")

    x_vals = [coord[0] for coord in images]
    y_vals = [coord[1] for coord in images]
    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = min(y_vals), max(y_vals)

    stride_x = max(1, width - x_overlap)
    stride_y = max(1, height - y_overlap)

    canvas_width = (max_x - min_x) * stride_x + width
    canvas_height = (max_y - min_y) * stride_y + height

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    priority = np.full((canvas_height, canvas_width), fill_value=-1, dtype=np.int32)

    for (x, y), image in images.items():
        record = records[(x, y)]

        px = (x - min_x) * stride_x
        py = (max_y - y) * stride_y

        y_slice = slice(py, py + height)
        x_slice = slice(px, px + width)

        region_priority = priority[y_slice, x_slice]
        region_canvas = canvas[y_slice, x_slice]

        update_mask = record.iteration >= region_priority
        region_canvas[update_mask] = image[update_mask]
        region_priority[update_mask] = record.iteration

    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stitch inspection images into a single mosaic")
    parser.add_argument("run_dir", type=Path, help="Directory containing capture images")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <run_dir>/stitched_mosaic.png)",
    )
    parser.add_argument(
        "--x-overlap",
        type=int,
        default=None,
        help="Manual horizontal overlap in pixels. If omitted, auto-estimated.",
    )
    parser.add_argument(
        "--y-overlap",
        type=int,
        default=None,
        help="Manual vertical overlap in pixels. If omitted, auto-estimated.",
    )
    parser.add_argument(
        "--no-auto-overlap",
        action="store_true",
        help="Disable auto-estimation for any overlap not manually provided (uses 0).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir: Path = args.run_dir

    if not run_dir.exists() or not run_dir.is_dir():
        raise ValueError(f"Run directory does not exist or is not a directory: {run_dir}")

    output_path = args.output if args.output is not None else run_dir / "stitched_mosaic.png"

    records, skipped, discarded = collect_tile_records(run_dir)
    if not records:
        raise ValueError(
            "No valid image filenames found. Expected format: x{x}_y{y}_{iteration}.png"
        )

    images, shape = read_tiles(records)
    height, width, channels = shape

    if channels != 3:
        raise ValueError(f"Expected 3-channel color images, got {channels} channels")

    gray_images = {coord: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for coord, img in images.items()}

    x_estimate = OverlapEstimate(pixels=0, confidence=0.0, samples=0)
    y_estimate = OverlapEstimate(pixels=0, confidence=0.0, samples=0)

    x_overlap = args.x_overlap
    y_overlap = args.y_overlap

    if not args.no_auto_overlap:
        if x_overlap is None:
            x_estimate = estimate_axis_overlap(gray_images, max_candidate=width, horizontal=True)
            x_overlap = x_estimate.pixels
        if y_overlap is None:
            y_estimate = estimate_axis_overlap(gray_images, max_candidate=height, horizontal=False)
            y_overlap = y_estimate.pixels

    if x_overlap is None:
        x_overlap = 0
    if y_overlap is None:
        y_overlap = 0

    if x_overlap < 0 or y_overlap < 0:
        raise ValueError("Overlap values cannot be negative")

    stitched = stitch_tiles(
        images=images,
        records=records,
        width=width,
        height=height,
        x_overlap=x_overlap,
        y_overlap=y_overlap,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), stitched)
    if not success:
        raise ValueError(f"Failed to write output image to: {output_path}")

    print("Stitch complete")
    print(f"run_dir: {run_dir}")
    print(f"output: {output_path}")
    print(f"scanned_files: {len(list(run_dir.iterdir()))}")
    print(f"tiles_used: {len(records)}")
    print(f"skipped_nonmatching: {len(skipped)}")
    print(f"discarded_duplicates: {len(discarded)}")
    print(f"tile_size: {width}x{height}")
    print(f"x_overlap_px: {x_overlap}")
    print(f"y_overlap_px: {y_overlap}")

    if args.x_overlap is None and not args.no_auto_overlap:
        print(
            "x_overlap_estimate_confidence: "
            f"{x_estimate.confidence:.3f} (samples={x_estimate.samples})"
        )
    if args.y_overlap is None and not args.no_auto_overlap:
        print(
            "y_overlap_estimate_confidence: "
            f"{y_estimate.confidence:.3f} (samples={y_estimate.samples})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
