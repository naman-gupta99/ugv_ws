import copy
import math
from typing import Optional
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    # Fallback in case dotenv is not installed
    def load_dotenv(path=None):
        pass

from sensor_msgs.msg import LaserScan

# Load environment variables from .env file in agent directory
_env_path = Path(__file__).parent / "agent" / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv()

# Get PLATFORM from environment (SIM or ROVER)
PLATFORM = os.getenv("PLATFORM", "SIM")

# LiDAR optical axis is mounted 90 degrees to the rover's left.
LIDAR_MOUNT_YAW_RAD = math.pi / 2.0


def _is_full_scan(scan: LaserScan) -> bool:
    if not scan.ranges:
        return False
    covered = abs(scan.angle_increment) * len(scan.ranges)
    return abs(covered - 2.0 * math.pi) < 0.25


def normalize_scan_for_rover(
    scan: Optional[LaserScan],
    mount_yaw_rad: float = LIDAR_MOUNT_YAW_RAD,
) -> Optional[LaserScan]:
    """Return a scan re-indexed so angles align with rover-forward frame."""
    if scan is None or not scan.ranges:
        return scan

    normalized = copy.deepcopy(scan)
    n = len(scan.ranges)
    inc = scan.angle_increment

    if inc == 0.0:
        return normalized

    shift_bins = int(round(mount_yaw_rad / inc))

    normalized.ranges = [scan.ranges[(i - shift_bins) % n] for i in range(n)]

    if scan.intensities:
        normalized.intensities = [scan.intensities[(i - shift_bins) % n] for i in range(len(scan.intensities))]

    return normalized


def index_for_angle(scan: LaserScan, angle_rad: float) -> int:
    """Return nearest index for a target angle in radians."""
    n = len(scan.ranges)
    if n == 0:
        raise ValueError("LaserScan has no ranges")

    inc = scan.angle_increment
    if inc == 0.0:
        return 0

    idx = int(round((angle_rad - scan.angle_min) / inc))

    if _is_full_scan(scan):
        return idx % n

    return max(0, min(n - 1, idx))


def finite_range_at_angle(
    scan: LaserScan,
    angle_rad: float,
    neighbor_search: int = 3,
) -> float:
    """Return finite range near angle_rad, falling back to range_max."""
    n = len(scan.ranges)
    if n == 0:
        return float(scan.range_max)

    center = index_for_angle(scan, angle_rad)
    periodic = _is_full_scan(scan)

    for step in range(neighbor_search + 1):
        candidates = [center] if step == 0 else [center - step, center + step]
        for idx in candidates:
            if periodic:
                test_idx = idx % n
            elif idx < 0 or idx >= n:
                continue
            else:
                test_idx = idx

            r = scan.ranges[test_idx]
            if math.isfinite(r) and scan.range_min <= r <= scan.range_max:
                return float(r)

    return float(scan.range_max)


def process_scan_for_rover(scan: Optional[LaserScan]) -> Optional[LaserScan]:
    """
    Process scan based on platform type.
    
    For ROVER platform: normalize scan to rover-forward frame.
    For SIM platform: return raw scan unchanged.
    
    This is the main entry point that all nodes should use.
    """
    if PLATFORM == "ROVER":
        return normalize_scan_for_rover(scan)
    else:
        # SIM mode: return raw scan without manipulation
        return scan
