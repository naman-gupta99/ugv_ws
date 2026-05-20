#!/usr/bin/env python
# encoding: utf-8

import json
import math
import subprocess
import threading
import time

import numpy as np
from PIL import Image as PILImage
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import JointState
from sensor_msgs.msg import LaserScan
from ugv_interface.action import Behavior

from .agent.audit_toolset import audit_state_instance
from .lidar_scan_utils import process_scan_for_rover

# Search range boundaries
IDEAL_DISTANCE = 1.2   # m — target distance from wall
MIN_DISTANCE   = 0.6   # m — inner boundary (too close)
MAX_DISTANCE   = 2.5   # m — outer boundary (too far)

# Accessibility thresholds
SIDE_CLEARANCE  = 1.0  # m — minimum lateral clearance on each side
FRONT_HALF_DEG  = 10.0 # ° — half-width for front distance measurement
SIDE_HALF_DEG   = 10.0 # ° — half-width for left/right clearance sectors

# Exploration
PROBE_STEP      = 0.1   # m — spacing between probe points during sweep
POSITION_TOL    = 0.10  # m — acceptable error when moving to final position
SAFETY_MARGIN   = 0.3  # m — buffer kept from any obstacle when moving
MAX_FINAL_ADJUST_ATTEMPTS = 8

# Midpoint of search range: rover goes to the nearer endpoint first
_SWEEP_MIDPOINT = (MIN_DISTANCE + MAX_DISTANCE) / 2.0  # 1.55 m

# Window detection via pan-tilt scan at MAX_DISTANCE
IMAGE_SAVE_PATH       = '/tmp/dist_wall_capture.png'
DETECT_API_URL        = 'http://127.0.0.1:8000/detect'
PT_TILT_MIN_RAD       = -0.5   # tilt down limit
PT_TILT_MAX_RAD       =  1.0   # tilt up limit
PT_TILT_STEP_RAD      =  0.1
PT_TILT_TOL_RAD       =  0.03
PT_VEL_TOL_RAD_S      =  0.03
PT_SETTLE_HOLD_S      =  0.20
PT_SETTLE_TIMEOUT_S   =  3.0
WINDOW_MIN_CONFIDENCE =  0.70
EDGE_MARGIN_PX        =  20    # px clearance on all sides = "fully visible"
X_M_PER_UNIT          =  1.2   # metres per llmptctrl grid unit (x axis)
Y_M_PER_UNIT          =  0.9   # metres per llmptctrl grid unit (y axis)


class DistanceCtrl(Node):

    def __init__(self):
        super().__init__('distance_ctrl')
        self._scan = None
        self._scan_event = threading.Event()
        self._joint_event = threading.Event()
        self._pt_pan = None
        self._pt_tilt = None
        self._pt_pan_vel = 0.0
        self._pt_tilt_vel = 0.0
        self._last_joint_stamp = None
        self._last_pan = None
        self._last_tilt = None
        self.create_subscription(LaserScan, 'scan', self._scan_cb, 10)
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)
        self._behavior_client = ActionClient(self, Behavior, 'behavior')
        self._side_clearance_m = SIDE_CLEARANCE

        # Camera image
        self._image: RosImage | None = None
        self._image_event = threading.Event()
        self.create_subscription(RosImage, '/pt_camera/image_raw', self._image_cb, 1)

        # Camera intrinsics
        self._camera_info: CameraInfo | None = None
        self._camera_info_event = threading.Event()
        self.create_subscription(CameraInfo, '/pt_camera/camera_info', self._camera_info_cb, 1)

        # Pan-tilt joint publisher
        self._joint_pub = self.create_publisher(JointState, '/ugv/joint_states', 10)

    # ------------------------------------------------------------------
    # Scan callbacks & helpers
    # ------------------------------------------------------------------

    def _scan_cb(self, msg):
        self._scan = process_scan_for_rover(msg)
        self._scan_event.set()

    def _image_cb(self, msg: RosImage) -> None:
        self._image = msg
        self._image_event.set()

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        if self._camera_info is None:
            self._camera_info = msg
            self._camera_info_event.set()

    def _joint_cb(self, msg: JointState) -> None:
        try:
            pan_idx = msg.name.index('pt_base_link_to_pt_link1')
            tilt_idx = msg.name.index('pt_link1_to_pt_link2')
        except ValueError:
            return

        pan = msg.position[pan_idx]
        tilt = msg.position[tilt_idx]

        pan_vel = None
        tilt_vel = None
        if len(msg.velocity) > pan_idx:
            pan_vel = msg.velocity[pan_idx]
        if len(msg.velocity) > tilt_idx:
            tilt_vel = msg.velocity[tilt_idx]

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if (pan_vel is None or tilt_vel is None) and self._last_joint_stamp is not None:
            dt = stamp - self._last_joint_stamp
            if dt > 1e-6 and self._last_pan is not None and self._last_tilt is not None:
                if pan_vel is None:
                    pan_vel = (pan - self._last_pan) / dt
                if tilt_vel is None:
                    tilt_vel = (tilt - self._last_tilt) / dt

        self._pt_pan = pan
        self._pt_tilt = tilt
        self._pt_pan_vel = float(pan_vel) if pan_vel is not None else 0.0
        self._pt_tilt_vel = float(tilt_vel) if tilt_vel is not None else 0.0
        self._last_joint_stamp = stamp
        self._last_pan = pan
        self._last_tilt = tilt
        self._joint_event.set()

    def _fresh_scan(self):
        """Wait for the next LiDAR scan and return it."""
        self._scan_event.clear()
        self._scan_event.wait()
        return self._scan

    def _sector_median(self, scan, center_deg: float, half_deg: float):
        """
        Median range of valid returns in the angular sector
        [center_deg ± half_deg].  Returns None if the sector is empty.
        """
        center_rad = math.radians(center_deg)
        half_rad   = math.radians(half_deg)
        dists = []
        for i, r in enumerate(scan.ranges):
            if not math.isfinite(r) or not (scan.range_min <= r <= scan.range_max):
                continue
            angle = scan.angle_min + i * scan.angle_increment
            # Normalize difference to [-π, π] to handle wrap-around
            diff = (angle - center_rad + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) <= half_rad:
                dists.append(r)
        return float(np.median(dists)) if dists else None

    def front_distance(self):
        """Median distance straight ahead (0° ± FRONT_HALF_DEG)."""
        return self._sector_median(self._fresh_scan(), 0.0, FRONT_HALF_DEG)

    def rear_distance(self):
        """Median distance directly behind (180° ± FRONT_HALF_DEG)."""
        return self._sector_median(self._fresh_scan(), 180.0, FRONT_HALF_DEG)

    def is_accessible(self):
        """
        Read one fresh scan and check both lateral clearances.

        Accessible = left clearance (90° ± 10°) ≥ 1 m
                   AND right clearance (−90° ± 10°) ≥ 1 m
        Returns True/False; logs the measured clearances.
        """
        scan = self._fresh_scan()
        lc = self._sector_median(scan,  90.0, SIDE_HALF_DEG)
        rc = self._sector_median(scan, -90.0, SIDE_HALF_DEG)

        if lc is None or rc is None:
            self.get_logger().warn('  Side sector returned no valid readings — marking inaccessible.')
            return False

        threshold = self._side_clearance_m
        self.get_logger().info(
            f'    left: {lc:.2f} m   right: {rc:.2f} m   threshold: {threshold:.2f} m'
        )
        return lc >= threshold and rc >= threshold

    # ------------------------------------------------------------------
    # Pan-tilt camera helpers (same pattern as WallCenteringCtrl)
    # ------------------------------------------------------------------

    def _capture_image(self) -> str:
        """Block until a fresh camera frame arrives, save as PNG, return path."""
        self._image_event.clear()
        if not self._image_event.wait(timeout=10.0) or self._image is None:
            raise RuntimeError('No camera frame received within 10 s.')

        msg = self._image
        encoding = msg.encoding.lower()

        if 'mono' in encoding:
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            rgb = np.stack([raw] * 3, axis=-1)
        else:
            channels = 4 if encoding in ('rgba8', 'bgra8') else 3
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
            if 'bgr' in encoding:
                rgb = raw[:, :, 2::-1]
            else:
                rgb = raw[:, :, :3]

        PILImage.fromarray(rgb, 'RGB').save(IMAGE_SAVE_PATH)
        return IMAGE_SAVE_PATH

    def _call_detection_api(self, image_path: str) -> dict | None:
        """POST image to detection API and return parsed JSON."""
        try:
            proc = subprocess.run(
                ['curl', '-s', '-X', 'POST', DETECT_API_URL,
                 '-F', f'file=@{image_path}'],
                capture_output=True, text=True, timeout=15,
            )
            return json.loads(proc.stdout)
        except Exception as exc:
            self.get_logger().error(f'Detection API error: {exc}')
            return None

    def _best_detection(self, response: dict) -> dict | None:
        """Return the detection with the highest confidence, or None."""
        detections = response.get('detections', [])
        if not detections:
            return None
        return max(detections, key=lambda d: d['confidence'])

    def _set_pt_tilt(self, tilt_rad: float) -> None:
        """Publish a JointState that sets tilt to tilt_rad with pan=0."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'ugv_joint_state'
        msg.name = [
            'left_up_wheel_link_joint',
            'left_down_wheel_link_joint',
            'right_up_wheel_link_joint',
            'right_down_wheel_link_joint',
            'pt_base_link_to_pt_link1',   # pan
            'pt_link1_to_pt_link2',       # tilt
        ]
        msg.position = [0.0, 0.0, 0.0, 0.0, 0.0, tilt_rad]
        self._joint_pub.publish(msg)

    def _wait_pt_settle(self, target_tilt: float, timeout_s: float = PT_SETTLE_TIMEOUT_S) -> bool:
        if not self._joint_event.wait(timeout=timeout_s):
            self.get_logger().warn('No /joint_states data while waiting for pan-tilt settle.')
            return False

        start = time.monotonic()
        settled_since = None
        while time.monotonic() - start < timeout_s:
            if self._pt_pan is None or self._pt_tilt is None:
                time.sleep(0.02)
                continue

            target_ok = abs(self._pt_tilt - target_tilt) <= PT_TILT_TOL_RAD and abs(self._pt_pan) <= PT_TILT_TOL_RAD
            vel_ok = abs(self._pt_pan_vel) <= PT_VEL_TOL_RAD_S and abs(self._pt_tilt_vel) <= PT_VEL_TOL_RAD_S
            if target_ok and vel_ok:
                if settled_since is None:
                    settled_since = time.monotonic()
                if time.monotonic() - settled_since >= PT_SETTLE_HOLD_S:
                    return True
            else:
                settled_since = None

            time.sleep(0.02)

        self.get_logger().warn(f'Pan-tilt settle timeout at target tilt {target_tilt:+.2f} rad.')
        return False

    def _is_window_fully_visible(self, detection: dict,
                                  image_width: int, image_height: int) -> bool:
        """True when the bounding box does not clip any image edge."""
        box = detection['box']
        return (
            box['x1'] > EDGE_MARGIN_PX and
            box['x2'] < image_width  - EDGE_MARGIN_PX and
            box['y1'] > EDGE_MARGIN_PX and
            box['y2'] < image_height - EDGE_MARGIN_PX
        )

    def _pixel_to_wall(self, px: float, py: float,
                        fx: float, fy: float, cx: float, cy: float,
                        tilt_rad: float, wall_distance: float) -> tuple[float, float]:
        """
        Project pixel (px, py) onto the wall plane at wall_distance,
        compensating for camera tilt_rad (positive = tilt down).

        Returns (x_world, y_world):
          x_world: positive = right
          y_world: positive = up, NEGATIVE = DOWN
        """
        dy_cam   = (py - cy) / fy
        dz_world = -math.sin(tilt_rad) * dy_cam + math.cos(tilt_rad)
        t        = wall_distance / dz_world
        x_world  = ((px - cx) / fx) * t
        y_world  = (-math.cos(tilt_rad) * dy_cam - math.sin(tilt_rad)) * t
        return x_world, y_world

    def _scan_for_window_coords(self) -> dict | None:
        """
        Sweep the pan-tilt camera vertically (PT_TILT_MIN_RAD → PT_TILT_MAX_RAD).
        At each tilt, capture an image and call the detection API.
        When a window with confidence ≥ WINDOW_MIN_CONFIDENCE is fully visible
        (bounding box does not clip any edge), project its corners onto the wall
        plane accounting for the current tilt angle, then convert physical extents
        to llmptctrl grid coordinates.

        Grid convention: +x right, +y up, -y down.
        Origin = wall point aligned with pan=0, tilt=0 optical axis.
        Rates: 1.3 m/unit (x), 0.9 m/unit (y).

        Updates audit_state_instance.target_area and regenerates remaining_coordinates.
        Returns target_area dict or None if no fully-visible window was found.
        """
        if not self._camera_info_event.wait(timeout=10.0) or self._camera_info is None:
            self.get_logger().error('No camera info — skipping window coord scan.')
            return None

        fx           = self._camera_info.k[0]
        fy           = self._camera_info.k[4]
        cx           = self._camera_info.k[2]
        cy           = self._camera_info.k[5]
        image_width  = self._camera_info.width
        image_height = self._camera_info.height

        wall_distance = self.front_distance()
        if wall_distance is None:
            self.get_logger().error('Cannot read wall distance — aborting coord calc.')
            return None

        tilt_positions = list(np.arange(
            PT_TILT_MIN_RAD,
            PT_TILT_MAX_RAD + PT_TILT_STEP_RAD / 2,
            PT_TILT_STEP_RAD,
        ))
        self.get_logger().info(
            f'Window coord scan at {wall_distance:.2f} m: '
            f'{len(tilt_positions)} tilt positions '
            f'[{PT_TILT_MIN_RAD:.2f} … {PT_TILT_MAX_RAD:.2f} rad]'
        )

        def _round_min_threshold(value: float) -> int:
            frac, int_part = math.modf(value)
            if abs(frac) > 0.5:
                return math.floor(value)
            return int(int_part)

        def _round_max_threshold(value: float) -> int:
            frac, int_part = math.modf(value)
            if abs(frac) > 0.5:
                return math.ceil(value)
            return int(int_part)

        for tilt in tilt_positions:
            self.get_logger().info(f'  Tilt → {tilt:+.2f} rad')
            self._set_pt_tilt(tilt)
            self._wait_pt_settle(tilt)

            try:
                image_path = self._capture_image()
            except RuntimeError as exc:
                self.get_logger().warn(f'  Frame capture failed: {exc}')
                continue

            response = self._call_detection_api(image_path)
            if response is None:
                continue

            best = self._best_detection(response)
            if best is None or best['confidence'] < WINDOW_MIN_CONFIDENCE:
                continue

            if not self._is_window_fully_visible(best, image_width, image_height):
                self.get_logger().info(
                    f'  Detected (conf={best["confidence"]:.3f}) but clipped — continuing.'
                )
                continue

            # Project bounding box corners onto wall plane, compensating for tilt
            box = best['box']
            x_left,  y_top    = self._pixel_to_wall(
                box['x1'], box['y1'], fx, fy, cx, cy, tilt, wall_distance)
            x_right, y_bottom = self._pixel_to_wall(
                box['x2'], box['y2'], fx, fy, cx, cy, tilt, wall_distance)
            # y_top >= 0   : top of box is above or at optical-axis height
            # y_bottom <= 0: bottom of box is below optical-axis height (-y = down)

            window_width_m = abs(x_right - x_left)
            self._side_clearance_m = (window_width_m + X_M_PER_UNIT) / 2.0

            y_compensation = math.tan(tilt) * wall_distance

            x_min = _round_min_threshold(x_left / X_M_PER_UNIT)
            x_max = _round_max_threshold(x_right / X_M_PER_UNIT)
            y_min = _round_min_threshold((y_bottom + y_compensation) / Y_M_PER_UNIT)
            y_max = _round_max_threshold((y_top + y_compensation) / Y_M_PER_UNIT)

            target_area = {'x_min': x_min, 'x_max': x_max,
                           'y_min': y_min, 'y_max': y_max}

            self.get_logger().info(
                f'  Fully-visible window  tilt={tilt:+.2f} rad  '
                f'conf={best["confidence"]:.3f}\n'
                f'  wall coords: x=[{x_left:.2f}, {x_right:.2f}] m  '
                f'y=[{y_bottom:.2f}, {y_top:.2f}] m\n'
                f'  side clearance threshold: {self._side_clearance_m:.2f} m\n'
                f'  grid target_area: {target_area}'
            )

            audit_state_instance.target_area = target_area
            audit_state_instance.remaining_coordinates = \
                audit_state_instance._State__generate_goal_coordinates()

            self._set_pt_tilt(0.0)
            return target_area

        self.get_logger().warn('Window coord scan complete — no fully-visible window found.')
        self._set_pt_tilt(0.0)
        return None

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def _send_move(self, toward_wall_m: float) -> float:
        """
        Move toward the wall by *toward_wall_m* metres (negative = away).

        Before moving, reads the relevant clearance sector and clamps the
        requested distance to (clearance - SAFETY_MARGIN) so the rover
        never drives into an obstacle.

        Returns the actual distance moved (≥ 0).  Returns 0.0 if the path
        is fully blocked or the requested distance is negligible.
        """
        if abs(toward_wall_m) < 1e-3:
            return 0.0

        requested = abs(toward_wall_m)

        if toward_wall_m > 0:
            # Moving toward wall — check front clearance
            clearance = self.front_distance()
            cmd_type, label = 'drive_on_heading', 'toward wall'
        else:
            # Moving away from wall — check rear clearance
            clearance = self.rear_distance()
            cmd_type, label = 'back_up', 'away from wall'

        if clearance is None:
            self.get_logger().warn(
                f'No {label.split()[0]} distance reading — aborting move.')
            return 0.0

        available = clearance - SAFETY_MARGIN
        if available <= 1e-3:
            self.get_logger().warn(
                f'Path {label} fully blocked (clearance {clearance:.2f} m, '
                f'margin {SAFETY_MARGIN} m) — skipping move.')
            return 0.0

        actual = min(requested, available)
        if actual < requested - 1e-3:
            self.get_logger().warn(
                f'Clamping move {label}: requested {requested:.2f} m, '
                f'available {available:.2f} m (clearance {clearance:.2f} m).')

        goal = Behavior.Goal()
        goal.command = json.dumps([{'type': cmd_type, 'data': actual}])
        self.get_logger().info(f'Moving {label} by {actual:.2f} m')

        done = threading.Event()
        outcome = {'success': False, 'message': ''}

        def _result_cb(_):
            result = _.result()
            behavior_result = result.result
            outcome['success'] = bool(getattr(behavior_result, 'result', False))
            outcome['message'] = getattr(behavior_result, 'message', '')
            done.set()

        def _goal_cb(future):
            handle = future.result()
            if not handle.accepted:
                outcome['message'] = f'Behavior goal for {cmd_type} was rejected.'
                done.set()
                return
            handle.get_result_async().add_done_callback(_result_cb)

        self._behavior_client.send_goal_async(goal).add_done_callback(_goal_cb)
        done.wait()
        if not outcome['success']:
            self.get_logger().warn(f'Behavior move failed: {outcome["message"]}')
            return None
        return actual

    def _measured_front_distance(self, context: str) -> float | None:
        """Read and log the actual measured wall distance."""
        distance = self.front_distance()
        if distance is None:
            self.get_logger().warn(f'Cannot read measured wall distance after {context}.')
            return None
        self.get_logger().info(f'Measured wall distance after {context}: {distance:.2f} m')
        return distance

    def _move_then_measure(self, toward_wall_m: float, context: str) -> float | None:
        """Send a move command, then use LiDAR rather than the command as truth."""
        commanded = self._send_move(toward_wall_m)
        if commanded is None:
            return None
        self.get_logger().info(
            f'Commanded {commanded:.2f} m for {context}; waiting for measured distance.'
        )
        return self._measured_front_distance(context)

    def _move_to_measured_distance(
        self,
        target_distance: float,
        tolerance: float,
        reason: str,
    ) -> float | None:
        """
        Iteratively command moves until the measured wall distance is close
        enough to target_distance.  The behavior command is never treated as
        the final position estimate.
        """
        current = self.front_distance()
        if current is None:
            self.get_logger().warn(f'Cannot read wall distance before {reason}.')
            return None

        for attempt in range(1, MAX_FINAL_ADJUST_ATTEMPTS + 1):
            error = current - target_distance
            self.get_logger().info(
                f'{reason}: measured {current:.2f} m, target {target_distance:.2f} m, '
                f'error {error:+.2f} m (tol {tolerance:.2f} m).'
            )
            if abs(error) <= tolerance:
                return current

            # Positive command drives toward the wall and reduces front distance.
            current = self._move_then_measure(error, f'{reason} attempt {attempt}')
            if current is None:
                return None

        self.get_logger().warn(
            f'{reason}: stopped after {MAX_FINAL_ADJUST_ATTEMPTS} attempts at '
            f'{current:.2f} m from wall (target {target_distance:.2f} m).'
        )
        return current

    # ------------------------------------------------------------------
    # Main exploration controller
    # ------------------------------------------------------------------

    def find_accessible_distance(self):
        """
        Explore measured wall distances in [MIN_DISTANCE, MAX_DISTANCE].

        The rover still commands PROBE_STEP-sized moves, but the sweep never
        assumes those moves are exact.  Each probe is recorded at the fresh
        LiDAR distance measured at that stop, and the sweep ends only when the
        measured distance crosses the min/max boundary.

        Returns the selected wall distance (float), or None if no
        accessible position was found.
        """
        # ── 1. Read initial distance ───────────────────────────────────
        d = self.front_distance()
        if d is None:
            self.get_logger().error('Cannot read front distance — aborting.')
            return None
        self.get_logger().info(f'Initial wall distance: {d:.2f} m')

        # ── 2. Enter the search range [MIN, MAX] using measured feedback ─
        if d > MAX_DISTANCE:
            # Case 1 — too far: drive toward wall to MAX_DISTANCE
            self.get_logger().info(
                f'[Case 1] d={d:.2f} m > {MAX_DISTANCE} m — approaching wall.')
            d = self._move_then_measure(d - MAX_DISTANCE, 'enter search range at max boundary')
            if d is None:
                return None

        elif d < MIN_DISTANCE:
            # Case 4 — too close: back away to MIN_DISTANCE
            self.get_logger().info(
                f'[Case 4] d={d:.2f} m < {MIN_DISTANCE} m — backing away.')
            d = self._move_then_measure(d - MIN_DISTANCE, 'enter search range at min boundary')
            if d is None:
                return None

        elif d > IDEAL_DISTANCE:
            self.get_logger().info(
                f'[Case 2] d={d:.2f} m in ({IDEAL_DISTANCE}, {MAX_DISTANCE}] — in range, above ideal.')
        else:
            self.get_logger().info(
                f'[Case 3] d={d:.2f} m in [{MIN_DISTANCE}, {IDEAL_DISTANCE}] — in range, below ideal.')

        if d < MIN_DISTANCE or d > MAX_DISTANCE:
            self.get_logger().warn(
                f'Could not enter search range exactly; measured {d:.2f} m. '
                'Continuing sweep from measured position and recording only in-range probes.'
            )

        # ── 3. Choose sweep direction for minimum-travel, zero-overlap ─
        #
        #   Rover goes to the nearer endpoint first, then sweeps in one
        #   continuous pass to the far endpoint.  The midpoint (1.55 m)
        #   is the crossover: below it we go to 0.6 first, above it to 2.5.
        #
        #   Travel cost  = |d - near_end| + (MAX - MIN)
        #   Choosing the nearer end minimises that first segment.
        #
        if d <= _SWEEP_MIDPOINT:
            sweep_start, sweep_end = MIN_DISTANCE, MAX_DISTANCE
            sweep_dir = +1   # moving away from wall (+distance) each step
        else:
            sweep_start, sweep_end = MAX_DISTANCE, MIN_DISTANCE
            sweep_dir = -1   # moving toward wall (−distance) each step

        self.get_logger().info(
            f'Sweep: {sweep_start:.1f} m → {sweep_end:.1f} m '
            f'(step {PROBE_STEP} m, {sweep_dir:+d} direction)')

        # Move once toward the sweep start, then trust only the measured
        # distance.  Exploration does not fine-position at probe points.
        if abs(d - sweep_start) > 1e-3:
            current_d = self._move_then_measure(d - sweep_start, 'move to sweep start')
            if current_d is None:
                return None
        else:
            current_d = d

        # ── 4. Single-pass sweep — command 10 cm, record measured stops ─
        accessible_positions = []   # wall distances where rover is accessible
        _window_scan_done = False
        probe_count = 0
        max_probe_count = int(math.ceil((MAX_DISTANCE - MIN_DISTANCE) / PROBE_STEP)) + 10

        while True:
            current_d = self.front_distance()
            if current_d is None:
                self.get_logger().warn('Cannot read front distance during sweep — stopping.')
                break

            if current_d < MIN_DISTANCE or current_d > MAX_DISTANCE:
                self.get_logger().info(
                    f'Measured sweep boundary crossed at {current_d:.2f} m '
                    f'outside [{MIN_DISTANCE:.2f}, {MAX_DISTANCE:.2f}] m — stopping.'
                )
                break

            probe_d = current_d

            # At the farthest measured probe point, scan for window grid coordinates
            # before checking accessibility so is_accessible() uses the
            # window-derived side-clearance threshold.
            if not _window_scan_done and sweep_dir == -1 and probe_count == 0:
                self.get_logger().info(
                    f'At farthest measured probe point ({probe_d:.2f} m) — scanning for window coordinates.'
                )
                self._scan_for_window_coords()
                _window_scan_done = True

            self.get_logger().info(f'Probing measured wall distance {probe_d:.2f} m ...')
            if self.is_accessible():
                accessible_positions.append(probe_d)
                self.get_logger().info(f'  -> ACCESSIBLE recorded at measured {probe_d:.2f} m')
            else:
                self.get_logger().info(f'  -> blocked recorded at measured {probe_d:.2f} m')

            if probe_count >= max_probe_count:
                self.get_logger().warn('Sweep probe limit reached — stopping.')
                break
            probe_count += 1

            # sweep_dir +1 -> distance increases -> move away from wall -> _send_move < 0
            requested_move = -sweep_dir * PROBE_STEP
            next_d = self._move_then_measure(
                requested_move,
                f'sweep step {probe_count} from measured {probe_d:.2f} m',
            )
            if next_d is None:
                return None
            if (
                not _window_scan_done and
                sweep_dir == 1 and
                (next_d < MIN_DISTANCE or next_d > MAX_DISTANCE)
            ):
                self.get_logger().info(
                    f'Last in-range measured probe was {probe_d:.2f} m before boundary '
                    'crossing — scanning for window coordinates.'
                )
                self._scan_for_window_coords()
                _window_scan_done = True
            if abs(next_d - probe_d) < 1e-3:
                self.get_logger().warn(
                    f'Sweep command produced no measured distance change at {next_d:.2f} m — stopping.'
                )
                break

        if not _window_scan_done and accessible_positions:
            self.get_logger().info(
                'Window coordinate scan was not triggered by a boundary condition; '
                'scanning from current measured position.'
            )
            self._scan_for_window_coords()

        # ── 5. Select the accessible position closest to IDEAL_DISTANCE ─
        if not accessible_positions:
            self.get_logger().warn(
                f'No accessible position found in [{MIN_DISTANCE}, {MAX_DISTANCE}] m. '
                'Inspection cannot proceed.')
            return None

        best_d = min(accessible_positions, key=lambda x: abs(x - IDEAL_DISTANCE))
        delta  = best_d - IDEAL_DISTANCE
        self.get_logger().info(
            f'Best accessible distance: {best_d:.2f} m '
            f'(Δ from ideal: {delta:+.2f} m)  '
            f'[{len(accessible_positions)} accessible positions found]')

        # ── 6. Move to the chosen position ───────────────────────────
        final = self._move_to_measured_distance(
            best_d,
            POSITION_TOL,
            'return to selected inspection distance',
        )
        if final is None:
            return None
        self.get_logger().info(
            f'Positioned at {final:.2f} m from wall '
            f'(target {best_d:.2f} m, tolerance {POSITION_TOL:.2f} m).')
        return best_d


def main(args=None):
    rclpy.init(args=args)
    node = DistanceCtrl()
    result = [None]
    done_event = threading.Event()

    def _run():
        time.sleep(1.0)  # Allow subscriptions / action server to connect
        result[0] = node.find_accessible_distance()
        done_event.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
