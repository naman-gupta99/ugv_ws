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
import math
import re
import statistics
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

FILENAME_RE = re.compile(r"^x(-?\d+)_y(-?\d+)_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)
DEFAULT_CAMERA_MODEL_PATHS = (
    Path("src/ugv_main/ugv_gazebo/models/ugv_rover/model.sdf"),
    Path("src/ugv_main/ugv_description/urdf/ugv_rover.urdf"),
)
X_M_PER_UNIT = 1.1
Y_M_PER_UNIT = 0.9
WALL_DISTANCE_M = 1.2


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


@dataclass(frozen=True)
class AlignmentEdge:
    base_coord: Tuple[int, int]
    neighbor_coord: Tuple[int, int]
    dx: float
    dy: float
    inliers: int
    matches: int


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    horizontal_fov: float
    source: str


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


def estimate_feature_edge(
    base_gray: np.ndarray,
    neighbor_gray: np.ndarray,
    base_coord: Tuple[int, int],
    neighbor_coord: Tuple[int, int],
    expected_dx: float,
    expected_dy: float,
    min_inliers: int,
) -> Optional[AlignmentEdge]:
    orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8)
    base_keypoints, base_desc = orb.detectAndCompute(base_gray, None)
    neighbor_keypoints, neighbor_desc = orb.detectAndCompute(neighbor_gray, None)

    if base_desc is None or neighbor_desc is None:
        return None
    if len(base_keypoints) < min_inliers or len(neighbor_keypoints) < min_inliers:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(neighbor_desc, base_desc, k=2)

    good_matches = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        best, second = pair
        if best.distance < 0.75 * second.distance:
            good_matches.append(best)

    if len(good_matches) < min_inliers:
        return None

    height, width = base_gray.shape
    max_delta_error = max(width, height) * 0.35
    direction_filtered = []
    for match in good_matches:
        neighbor_pt = neighbor_keypoints[match.queryIdx].pt
        base_pt = base_keypoints[match.trainIdx].pt
        dx = base_pt[0] - neighbor_pt[0]
        dy = base_pt[1] - neighbor_pt[1]
        if abs(dx - expected_dx) <= max_delta_error and abs(dy - expected_dy) <= max_delta_error:
            direction_filtered.append(match)

    if len(direction_filtered) < min_inliers:
        return None

    src = np.float32([neighbor_keypoints[m.queryIdx].pt for m in direction_filtered]).reshape(-1, 1, 2)
    dst = np.float32([base_keypoints[m.trainIdx].pt for m in direction_filtered]).reshape(-1, 1, 2)

    _affine, inlier_mask = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=4.0,
        maxIters=3000,
        confidence=0.995,
    )
    if inlier_mask is None:
        return None

    inliers = inlier_mask.ravel().astype(bool)
    if int(np.count_nonzero(inliers)) < min_inliers:
        return None

    inlier_matches = [match for match, keep in zip(direction_filtered, inliers) if keep]
    dx_values = []
    dy_values = []
    for match in inlier_matches:
        neighbor_pt = neighbor_keypoints[match.queryIdx].pt
        base_pt = base_keypoints[match.trainIdx].pt
        dx_values.append(base_pt[0] - neighbor_pt[0])
        dy_values.append(base_pt[1] - neighbor_pt[1])

    dx = float(statistics.median(dx_values))
    dy = float(statistics.median(dy_values))

    # Reject alignments that are geometrically implausible for adjacent grid cells.
    if abs(dx - expected_dx) > max_delta_error or abs(dy - expected_dy) > max_delta_error:
        return None

    return AlignmentEdge(
        base_coord=base_coord,
        neighbor_coord=neighbor_coord,
        dx=dx,
        dy=dy,
        inliers=len(inlier_matches),
        matches=len(good_matches),
    )


def estimate_feature_edges(
    coord_to_gray: Dict[Tuple[int, int], np.ndarray],
    stride_x: int,
    stride_y: int,
    x_overlap: int,
    y_overlap: int,
    min_inliers: int,
) -> List[AlignmentEdge]:
    edges: List[AlignmentEdge] = []

    for coord, base in coord_to_gray.items():
        x, y = coord
        horizontal_neighbor = (x + 1, y)
        if horizontal_neighbor in coord_to_gray:
            edge = estimate_feature_edge(
                base_gray=base,
                neighbor_gray=coord_to_gray[horizontal_neighbor],
                base_coord=coord,
                neighbor_coord=horizontal_neighbor,
                expected_dx=float(stride_x),
                expected_dy=0.0,
                min_inliers=min_inliers,
            )
            if edge is None:
                edge = estimate_strip_edge(
                    base_gray=base,
                    neighbor_gray=coord_to_gray[horizontal_neighbor],
                    base_coord=coord,
                    neighbor_coord=horizontal_neighbor,
                    horizontal=True,
                    stride_x=stride_x,
                    stride_y=stride_y,
                    x_overlap=x_overlap,
                    y_overlap=y_overlap,
                )
            if edge is not None:
                edges.append(edge)

        vertical_neighbor = (x, y + 1)
        if vertical_neighbor in coord_to_gray:
            edge = estimate_feature_edge(
                base_gray=base,
                neighbor_gray=coord_to_gray[vertical_neighbor],
                base_coord=coord,
                neighbor_coord=vertical_neighbor,
                expected_dx=0.0,
                expected_dy=float(-stride_y),
                min_inliers=min_inliers,
            )
            if edge is None:
                edge = estimate_strip_edge(
                    base_gray=base,
                    neighbor_gray=coord_to_gray[vertical_neighbor],
                    base_coord=coord,
                    neighbor_coord=vertical_neighbor,
                    horizontal=False,
                    stride_x=stride_x,
                    stride_y=stride_y,
                    x_overlap=x_overlap,
                    y_overlap=y_overlap,
                )
            if edge is not None:
                edges.append(edge)

    return edges


def estimate_strip_edge(
    base_gray: np.ndarray,
    neighbor_gray: np.ndarray,
    base_coord: Tuple[int, int],
    neighbor_coord: Tuple[int, int],
    horizontal: bool,
    stride_x: int,
    stride_y: int,
    x_overlap: int,
    y_overlap: int,
    confidence_threshold: float = 0.35,
) -> Optional[AlignmentEdge]:
    height, width = base_gray.shape

    if horizontal:
        if x_overlap <= 4:
            return None
        search_limit = max(8, int(round(height * 0.15)))
        best_score = -1.0
        best_dy = 0
        base_strip = base_gray[:, -x_overlap:]
        neighbor_strip = neighbor_gray[:, :x_overlap]

        for dy in range(-search_limit, search_limit + 1):
            if dy >= 0:
                base_crop = base_strip[dy:, :]
                neighbor_crop = neighbor_strip[: height - dy, :]
            else:
                base_crop = base_strip[: height + dy, :]
                neighbor_crop = neighbor_strip[-dy:, :]
            if base_crop.shape[0] < 16:
                continue
            score = normalized_cross_correlation(base_crop, neighbor_crop)
            if score > best_score:
                best_score = score
                best_dy = dy

        if best_score < confidence_threshold:
            return None
        return AlignmentEdge(
            base_coord=base_coord,
            neighbor_coord=neighbor_coord,
            dx=float(stride_x),
            dy=float(best_dy),
            inliers=max(6, int(round(best_score * 20))),
            matches=1,
        )

    if y_overlap <= 4:
        return None

    search_limit = max(8, int(round(width * 0.15)))
    best_score = -1.0
    best_dx = 0
    base_strip = base_gray[:y_overlap, :]
    neighbor_strip = neighbor_gray[-y_overlap:, :]

    for dx in range(-search_limit, search_limit + 1):
        if dx >= 0:
            base_crop = base_strip[:, dx:]
            neighbor_crop = neighbor_strip[:, : width - dx]
        else:
            base_crop = base_strip[:, : width + dx]
            neighbor_crop = neighbor_strip[:, -dx:]
        if base_crop.shape[1] < 16:
            continue
        score = normalized_cross_correlation(base_crop, neighbor_crop)
        if score > best_score:
            best_score = score
            best_dx = dx

    if best_score < confidence_threshold:
        return None
    return AlignmentEdge(
        base_coord=base_coord,
        neighbor_coord=neighbor_coord,
        dx=float(best_dx),
        dy=float(-stride_y),
        inliers=max(6, int(round(best_score * 20))),
        matches=1,
    )


def nominal_tile_positions(
    coords: List[Tuple[int, int]],
    width: int,
    height: int,
    x_overlap: int,
    y_overlap: int,
) -> Dict[Tuple[int, int], Tuple[float, float]]:
    x_vals = [coord[0] for coord in coords]
    y_vals = [coord[1] for coord in coords]
    min_x, max_y = min(x_vals), max(y_vals)
    stride_x = max(1, width - x_overlap)
    stride_y = max(1, height - y_overlap)

    return {
        (x, y): (float((x - min_x) * stride_x), float((max_y - y) * stride_y))
        for x, y in coords
    }


def solve_feature_positions(
    coords: List[Tuple[int, int]],
    nominal_positions: Dict[Tuple[int, int], Tuple[float, float]],
    edges: List[AlignmentEdge],
) -> Dict[Tuple[int, int], Tuple[float, float]]:
    index = {coord: idx for idx, coord in enumerate(sorted(coords))}
    row_x: List[np.ndarray] = []
    row_y: List[np.ndarray] = []
    rhs_x: List[float] = []
    rhs_y: List[float] = []

    def add_row(coord_weights: Dict[Tuple[int, int], float], x_value: float, y_value: float, weight: float) -> None:
        scale = weight ** 0.5
        x_row = np.zeros(len(index), dtype=np.float64)
        y_row = np.zeros(len(index), dtype=np.float64)
        for coord, coeff in coord_weights.items():
            x_row[index[coord]] = coeff * scale
            y_row[index[coord]] = coeff * scale
        row_x.append(x_row)
        row_y.append(y_row)
        rhs_x.append(x_value * scale)
        rhs_y.append(y_value * scale)

    for edge in edges:
        # More inliers make the image-derived constraint more trusted, but cap it
        # so one busy pair does not overpower the whole grid.
        edge_weight = float(min(30, max(6, edge.inliers)))
        add_row(
            {edge.neighbor_coord: 1.0, edge.base_coord: -1.0},
            edge.dx,
            edge.dy,
            edge_weight,
        )

    for coord, (px, py) in nominal_positions.items():
        add_row({coord: 1.0}, px, py, 0.25)

    anchor = min(index, key=lambda coord: (coord[0], -coord[1]))
    anchor_x, anchor_y = nominal_positions[anchor]
    add_row({anchor: 1.0}, anchor_x, anchor_y, 100.0)

    solved_x, *_ = np.linalg.lstsq(np.vstack(row_x), np.array(rhs_x), rcond=None)
    solved_y, *_ = np.linalg.lstsq(np.vstack(row_y), np.array(rhs_y), rcond=None)

    return {
        coord: (float(solved_x[idx]), float(solved_y[idx]))
        for coord, idx in index.items()
    }


def feather_weights(height: int, width: int, feather_px: int = 48) -> np.ndarray:
    y_dist = np.minimum(np.arange(height), np.arange(height)[::-1]).astype(np.float32)
    x_dist = np.minimum(np.arange(width), np.arange(width)[::-1]).astype(np.float32)
    dist = np.minimum(y_dist[:, None], x_dist[None, :])
    weights = np.clip(dist / max(1, feather_px), 0.05, 1.0)
    return weights


def composite_tiles_blended(
    images: Dict[Tuple[int, int], np.ndarray],
    positions: Dict[Tuple[int, int], Tuple[float, float]],
) -> np.ndarray:
    first_image = next(iter(images.values()))
    height, width = first_image.shape[:2]
    rounded_positions = {
        coord: (int(round(px)), int(round(py)))
        for coord, (px, py) in positions.items()
    }

    min_px = min(px for px, _py in rounded_positions.values())
    min_py = min(py for _px, py in rounded_positions.values())
    shifted_positions = {
        coord: (px - min_px, py - min_py)
        for coord, (px, py) in rounded_positions.items()
    }

    canvas_width = max(px + width for px, _py in shifted_positions.values())
    canvas_height = max(py + height for _px, py in shifted_positions.values())

    accum = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    weights = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    tile_weights = feather_weights(height, width)

    for coord, image in images.items():
        px, py = shifted_positions[coord]
        y_slice = slice(py, py + height)
        x_slice = slice(px, px + width)
        accum[y_slice, x_slice] += image.astype(np.float32) * tile_weights[:, :, None]
        weights[y_slice, x_slice] += tile_weights

    safe_weights = np.maximum(weights, 1e-6)
    blended = accum / safe_weights[:, :, None]
    empty_mask = weights <= 1e-6
    blended[empty_mask] = 0
    return np.clip(blended, 0, 255).astype(np.uint8)


def camera_intrinsics_from_hfov(
    width: int,
    height: int,
    horizontal_fov: float,
    source: str,
) -> CameraIntrinsics:
    fx = width / (2.0 * math.tan(horizontal_fov / 2.0))
    fy = fx
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        horizontal_fov=horizontal_fov,
        source=source,
    )


def extract_camera_intrinsics_from_xml(path: Path, image_width: int, image_height: int) -> Optional[CameraIntrinsics]:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return None

    candidates: List[CameraIntrinsics] = []
    for camera in root.iter("camera"):
        hfov_node = camera.find("horizontal_fov")
        image_node = camera.find("image")
        if hfov_node is None or image_node is None or hfov_node.text is None:
            continue

        width_node = image_node.find("width")
        height_node = image_node.find("height")
        if width_node is None or height_node is None or width_node.text is None or height_node.text is None:
            continue

        try:
            width = int(width_node.text.strip())
            height = int(height_node.text.strip())
            hfov = float(hfov_node.text.strip())
        except ValueError:
            continue

        camera_name = camera.attrib.get("name", "camera")
        candidates.append(camera_intrinsics_from_hfov(width, height, hfov, f"{path}:{camera_name}"))

    if not candidates:
        return None

    shape_matches = [
        intrinsics
        for intrinsics in candidates
        if intrinsics.width == image_width and intrinsics.height == image_height
    ]
    if shape_matches:
        for intrinsics in shape_matches:
            if "overhead_camera" in intrinsics.source:
                return intrinsics
        return shape_matches[0]

    return candidates[0]


def load_camera_intrinsics(
    image_width: int,
    image_height: int,
    camera_model_path: Optional[Path],
    horizontal_fov: Optional[float],
    fx: Optional[float],
    fy: Optional[float],
    cx: Optional[float],
    cy: Optional[float],
) -> CameraIntrinsics:
    if fx is not None or fy is not None or cx is not None or cy is not None:
        if fx is None or fy is None:
            raise ValueError("--fx and --fy must be provided together")
        return CameraIntrinsics(
            width=image_width,
            height=image_height,
            fx=float(fx),
            fy=float(fy),
            cx=float(cx) if cx is not None else (image_width - 1) / 2.0,
            cy=float(cy) if cy is not None else (image_height - 1) / 2.0,
            horizontal_fov=0.0,
            source="manual-focal-length",
        )

    if horizontal_fov is not None:
        return camera_intrinsics_from_hfov(
            width=image_width,
            height=image_height,
            horizontal_fov=float(horizontal_fov),
            source="manual-horizontal-fov",
        )

    paths = [camera_model_path] if camera_model_path is not None else list(DEFAULT_CAMERA_MODEL_PATHS)
    for path in paths:
        if path is None:
            continue
        intrinsics = extract_camera_intrinsics_from_xml(path, image_width, image_height)
        if intrinsics is not None:
            return intrinsics

    # Last-resort default from the ugv_rover overhead camera definition.
    return camera_intrinsics_from_hfov(
        width=image_width,
        height=image_height,
        horizontal_fov=0.9425,
        source="fallback-overhead-camera-hfov",
    )


def project_pixel_to_wall(
    u: float,
    v: float,
    record: TileRecord,
    intrinsics: CameraIntrinsics,
    wall_distance_m: float,
    x_m_per_unit: float,
    y_m_per_unit: float,
) -> Tuple[float, float]:
    x_ray = (u - intrinsics.cx) / intrinsics.fx
    y_ray = -(v - intrinsics.cy) / intrinsics.fy
    z_ray = 1.0

    tilt_rad = math.atan2(record.y * y_m_per_unit, wall_distance_m)
    cos_t = math.cos(tilt_rad)
    sin_t = math.sin(tilt_rad)

    world_x_ray = x_ray
    world_y_ray = cos_t * y_ray + sin_t * z_ray
    world_z_ray = -sin_t * y_ray + cos_t * z_ray
    if world_z_ray <= 1e-6:
        raise ValueError(
            f"Image ray does not intersect forward wall plane for {record.path.name}; "
            f"check tilt sign/model."
        )

    scale = wall_distance_m / world_z_ray
    wall_x = record.x * x_m_per_unit + scale * world_x_ray
    wall_y = scale * world_y_ray
    return wall_x, wall_y


def geometry_tile_corners(
    record: TileRecord,
    intrinsics: CameraIntrinsics,
    wall_distance_m: float,
    x_m_per_unit: float,
    y_m_per_unit: float,
) -> np.ndarray:
    width = intrinsics.width
    height = intrinsics.height
    image_corners = ((0.0, 0.0), (width - 1.0, 0.0), (width - 1.0, height - 1.0), (0.0, height - 1.0))
    return np.float32([
        project_pixel_to_wall(u, v, record, intrinsics, wall_distance_m, x_m_per_unit, y_m_per_unit)
        for u, v in image_corners
    ])


def stitch_tiles_geometry(
    images: Dict[Tuple[int, int], np.ndarray],
    records: Dict[Tuple[int, int], TileRecord],
    intrinsics: CameraIntrinsics,
    wall_distance_m: float,
    x_m_per_unit: float,
    y_m_per_unit: float,
    px_per_meter: Optional[float],
) -> Tuple[np.ndarray, float, Tuple[float, float, float, float]]:
    if wall_distance_m <= 0:
        raise ValueError("--wall-distance-m must be positive")
    if x_m_per_unit <= 0 or y_m_per_unit <= 0:
        raise ValueError("Grid unit sizes must be positive")

    all_corners = [
        geometry_tile_corners(record, intrinsics, wall_distance_m, x_m_per_unit, y_m_per_unit)
        for record in records.values()
    ]
    stacked = np.vstack(all_corners)
    min_x = float(np.min(stacked[:, 0]))
    max_x = float(np.max(stacked[:, 0]))
    min_y = float(np.min(stacked[:, 1]))
    max_y = float(np.max(stacked[:, 1]))

    scale = float(px_per_meter) if px_per_meter is not None else intrinsics.fx / wall_distance_m
    if scale <= 0:
        raise ValueError("--geometry-px-per-meter must be positive")

    margin_px = 8
    canvas_width = int(math.ceil((max_x - min_x) * scale)) + 1 + margin_px * 2
    canvas_height = int(math.ceil((max_y - min_y) * scale)) + 1 + margin_px * 2

    accum = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
    weights = np.zeros((canvas_height, canvas_width), dtype=np.float32)

    source_corners = np.float32([
        [0.0, 0.0],
        [intrinsics.width - 1.0, 0.0],
        [intrinsics.width - 1.0, intrinsics.height - 1.0],
        [0.0, intrinsics.height - 1.0],
    ])
    tile_weights = feather_weights(intrinsics.height, intrinsics.width)

    for coord, image in images.items():
        record = records[coord]
        wall_corners = geometry_tile_corners(
            record=record,
            intrinsics=intrinsics,
            wall_distance_m=wall_distance_m,
            x_m_per_unit=x_m_per_unit,
            y_m_per_unit=y_m_per_unit,
        )
        dest_corners = np.float32([
            [(x - min_x) * scale + margin_px, (max_y - y) * scale + margin_px]
            for x, y in wall_corners
        ])
        homography = cv2.getPerspectiveTransform(source_corners, dest_corners)
        warped_image = cv2.warpPerspective(
            image.astype(np.float32),
            homography,
            (canvas_width, canvas_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_weights = cv2.warpPerspective(
            tile_weights,
            homography,
            (canvas_width, canvas_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        accum += warped_image * warped_weights[:, :, None]
        weights += warped_weights

    safe_weights = np.maximum(weights, 1e-6)
    mosaic = accum / safe_weights[:, :, None]
    mosaic[weights <= 1e-6] = 0
    bounds = (min_x, max_x, min_y, max_y)
    return np.clip(mosaic, 0, 255).astype(np.uint8), scale, bounds


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
    parser.add_argument(
        "--alignment",
        choices=("geometry", "feature", "grid"),
        default="geometry",
        help=(
            "Tile placement strategy. 'geometry' projects images onto the flat wall plane; "
            "'feature' uses ORB/RANSAC neighbor alignment; 'grid' uses fixed overlap only."
        ),
    )
    parser.add_argument(
        "--feature-min-inliers",
        type=int,
        default=10,
        help="Minimum RANSAC inlier matches required to trust a feature alignment.",
    )
    parser.add_argument(
        "--camera-model",
        type=Path,
        default=None,
        help="SDF/URDF path to read camera horizontal_fov/image size from.",
    )
    parser.add_argument(
        "--horizontal-fov",
        type=float,
        default=None,
        help="Manual horizontal camera FOV in radians for geometry mode.",
    )
    parser.add_argument("--fx", type=float, default=None, help="Manual camera fx in pixels for geometry mode.")
    parser.add_argument("--fy", type=float, default=None, help="Manual camera fy in pixels for geometry mode.")
    parser.add_argument("--cx", type=float, default=None, help="Manual camera cx in pixels for geometry mode.")
    parser.add_argument("--cy", type=float, default=None, help="Manual camera cy in pixels for geometry mode.")
    parser.add_argument(
        "--wall-distance-m",
        type=float,
        default=WALL_DISTANCE_M,
        help="Distance from x0_y0 camera center to the flat window/wall plane.",
    )
    parser.add_argument(
        "--x-m-per-unit",
        type=float,
        default=X_M_PER_UNIT,
        help="Physical rover lateral motion per filename x step.",
    )
    parser.add_argument(
        "--y-m-per-unit",
        type=float,
        default=Y_M_PER_UNIT,
        help="Physical vertical target offset per filename y step, converted to tilt angle.",
    )
    parser.add_argument(
        "--geometry-px-per-meter",
        type=float,
        default=None,
        help="Output mosaic scale for geometry mode. Default is fx / wall_distance.",
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

    uses_overlap = args.alignment != "geometry"

    if uses_overlap and not args.no_auto_overlap:
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

    stride_x = max(1, width - x_overlap)
    stride_y = max(1, height - y_overlap)
    feature_edges: List[AlignmentEdge] = []

    geometry_info: Optional[Tuple[CameraIntrinsics, float, Tuple[float, float, float, float]]] = None

    if args.alignment == "geometry":
        intrinsics = load_camera_intrinsics(
            image_width=width,
            image_height=height,
            camera_model_path=args.camera_model,
            horizontal_fov=args.horizontal_fov,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy,
        )
        stitched, geometry_scale, geometry_bounds = stitch_tiles_geometry(
            images=images,
            records=records,
            intrinsics=intrinsics,
            wall_distance_m=args.wall_distance_m,
            x_m_per_unit=args.x_m_per_unit,
            y_m_per_unit=args.y_m_per_unit,
            px_per_meter=args.geometry_px_per_meter,
        )
        geometry_info = (intrinsics, geometry_scale, geometry_bounds)
    elif args.alignment == "feature":
        feature_edges = estimate_feature_edges(
            coord_to_gray=gray_images,
            stride_x=stride_x,
            stride_y=stride_y,
            x_overlap=x_overlap,
            y_overlap=y_overlap,
            min_inliers=args.feature_min_inliers,
        )
        nominal_positions = nominal_tile_positions(
            coords=list(images.keys()),
            width=width,
            height=height,
            x_overlap=x_overlap,
            y_overlap=y_overlap,
        )
        solved_positions = solve_feature_positions(
            coords=list(images.keys()),
            nominal_positions=nominal_positions,
            edges=feature_edges,
        )
        stitched = composite_tiles_blended(images=images, positions=solved_positions)
    else:
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
    print(f"alignment: {args.alignment}")
    if uses_overlap:
        print(f"x_overlap_px: {x_overlap}")
        print(f"y_overlap_px: {y_overlap}")
    if geometry_info is not None:
        intrinsics, geometry_scale, geometry_bounds = geometry_info
        min_wall_x, max_wall_x, min_wall_y, max_wall_y = geometry_bounds
        print(f"camera_intrinsics_source: {intrinsics.source}")
        print(f"camera_fx_fy_px: {intrinsics.fx:.3f}, {intrinsics.fy:.3f}")
        print(f"wall_distance_m: {args.wall_distance_m:.3f}")
        print(f"geometry_px_per_meter: {geometry_scale:.3f}")
        print(
            "geometry_wall_bounds_m: "
            f"x=[{min_wall_x:.3f},{max_wall_x:.3f}] "
            f"y=[{min_wall_y:.3f},{max_wall_y:.3f}]"
        )
    if args.alignment == "feature":
        print(f"alignment_edges_used: {len(feature_edges)}")
        if feature_edges:
            median_inliers = statistics.median(edge.inliers for edge in feature_edges)
            print(f"alignment_edge_median_weight: {median_inliers:.1f}")

    if uses_overlap and args.x_overlap is None and not args.no_auto_overlap:
        print(
            "x_overlap_estimate_confidence: "
            f"{x_estimate.confidence:.3f} (samples={x_estimate.samples})"
        )
    if uses_overlap and args.y_overlap is None and not args.no_auto_overlap:
        print(
            "y_overlap_estimate_confidence: "
            f"{y_estimate.confidence:.3f} (samples={y_estimate.samples})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
