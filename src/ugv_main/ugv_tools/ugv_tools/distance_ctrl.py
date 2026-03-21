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
POSITION_TOL    = 0.05  # m — acceptable error when moving to final position
SAFETY_MARGIN   = 0.3  # m — buffer kept from any obstacle when moving

# behavior_ctrl drives at exactly this speed (must match behavior_ctrl.py)
MOVE_SPEED_M_S  = 0.2   # m/s
MOVE_SETTLE_S   = 1.0   # extra settle time after movement completes

# Midpoint of search range: rover goes to the nearer endpoint first
_SWEEP_MIDPOINT = (MIN_DISTANCE + MAX_DISTANCE) / 2.0  # 1.55 m

# Window detection via pan-tilt scan at MAX_DISTANCE
IMAGE_SAVE_PATH       = '/tmp/dist_wall_capture.png'
DETECT_API_URL        = 'http://127.0.0.1:8000/detect'
PT_TILT_MIN_RAD       = -0.5   # tilt down limit
PT_TILT_MAX_RAD       =  1.0   # tilt up limit
PT_TILT_STEP_RAD      =  0.1
PT_TILT_SETTLE_S      =  0.5
WINDOW_MIN_CONFIDENCE =  0.70
EDGE_MARGIN_PX        =  20    # px clearance on all sides = "fully visible"
X_M_PER_UNIT          =  1.3   # metres per llmptctrl grid unit (x axis)
Y_M_PER_UNIT          =  0.9   # metres per llmptctrl grid unit (y axis)


class DistanceCtrl(Node):

    def __init__(self):
        super().__init__('distance_ctrl')
        self._scan = None
        self._scan_event = threading.Event()
        self.create_subscription(LaserScan, 'scan', self._scan_cb, 10)
        self._behavior_client = ActionClient(self, Behavior, 'behavior')

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
        self._scan = msg
        self._scan_event.set()

    def _image_cb(self, msg: RosImage) -> None:
        self._image = msg
        self._image_event.set()

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        if self._camera_info is None:
            self._camera_info = msg
            self._camera_info_event.set()

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

        self.get_logger().info(f'    left: {lc:.2f} m   right: {rc:.2f} m')
        return lc >= SIDE_CLEARANCE and rc >= SIDE_CLEARANCE

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

        for tilt in tilt_positions:
            self.get_logger().info(f'  Tilt → {tilt:+.2f} rad')
            self._set_pt_tilt(tilt)
            time.sleep(PT_TILT_SETTLE_S)

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

            y_compensation = math.tan(tilt) * wall_distance

            x_min = math.floor(x_left                      / X_M_PER_UNIT)
            x_max = math.ceil( x_right                     / X_M_PER_UNIT)
            y_min = math.floor((y_bottom + y_compensation) / Y_M_PER_UNIT)
            y_max = math.ceil( (y_top    + y_compensation) / Y_M_PER_UNIT)

            target_area = {'x_min': x_min, 'x_max': x_max,
                           'y_min': y_min, 'y_max': y_max}

            self.get_logger().info(
                f'  Fully-visible window  tilt={tilt:+.2f} rad  '
                f'conf={best["confidence"]:.3f}\n'
                f'  wall coords: x=[{x_left:.2f}, {x_right:.2f}] m  '
                f'y=[{y_bottom:.2f}, {y_top:.2f}] m\n'
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

        def _result_cb(_):
            done.set()

        def _goal_cb(future):
            future.result().get_result_async().add_done_callback(_result_cb)

        self._behavior_client.send_goal_async(goal).add_done_callback(_goal_cb)
        done.wait()
        # behavior_ctrl queues the command and returns the action result
        # immediately (before movement starts).  Sleep long enough for the
        # physical move to complete, then add a settle buffer.
        time.sleep(actual / MOVE_SPEED_M_S + MOVE_SETTLE_S)
        return actual

    # ------------------------------------------------------------------
    # Main exploration controller
    # ------------------------------------------------------------------

    def find_accessible_distance(self):
        """
        Explore [MIN_DISTANCE, MAX_DISTANCE] in a single continuous pass
        (zero overlap) to find every accessible position, then move to
        the one closest to IDEAL_DISTANCE.

        Cases handled
        -------------
        1. d > MAX_DISTANCE (2.5 m) : approach wall until d == 2.5 m
        2. MAX_DISTANCE ≥ d > IDEAL  : already in range, above ideal
        3. IDEAL ≥ d > MIN_DISTANCE  : already in range, below ideal
        4. d < MIN_DISTANCE (0.6 m)  : back away until d == 0.6 m

        For cases 2–3 the sweep direction is chosen so the rover reaches
        the nearer range endpoint first, giving the shortest total travel
        with no revisited positions.

        Returns the selected wall distance (float), or None if no
        accessible position was found.
        """
        # ── 1. Read initial distance ───────────────────────────────────
        d = self.front_distance()
        if d is None:
            self.get_logger().error('Cannot read front distance — aborting.')
            return None
        self.get_logger().info(f'Initial wall distance: {d:.2f} m')

        # ── 2. Enter the search range [MIN, MAX] ───────────────────────
        if d > MAX_DISTANCE:
            # Case 1 — too far: drive toward wall to MAX_DISTANCE
            self.get_logger().info(
                f'[Case 1] d={d:.2f} m > {MAX_DISTANCE} m — approaching wall.')
            self._send_move(d - MAX_DISTANCE)
            d = MAX_DISTANCE

        elif d < MIN_DISTANCE:
            # Case 4 — too close: back away to MIN_DISTANCE
            self.get_logger().info(
                f'[Case 4] d={d:.2f} m < {MIN_DISTANCE} m — backing away.')
            self._send_move(-(MIN_DISTANCE - d))
            d = MIN_DISTANCE

        elif d > IDEAL_DISTANCE:
            self.get_logger().info(
                f'[Case 2] d={d:.2f} m in ({IDEAL_DISTANCE}, {MAX_DISTANCE}] — in range, above ideal.')
        else:
            self.get_logger().info(
                f'[Case 3] d={d:.2f} m in [{MIN_DISTANCE}, {IDEAL_DISTANCE}] — in range, below ideal.')

        # d is now guaranteed ∈ [MIN_DISTANCE, MAX_DISTANCE]

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

        # Move from current d to sweep start, checking clearance first.
        # _send_move also clamps internally; this pre-check additionally
        # caps sweep_start so _farthest_point is accurate.
        delta_to_start = d - sweep_start   # >0 → toward wall, <0 → away

        if delta_to_start > 0:
            # Moving toward wall — verify front clearance.
            front = self.front_distance()
            if front is None:
                self.get_logger().warn('Cannot read front distance — skipping forward move.')
                delta_to_start = 0.0
            else:
                available_fwd = front - SAFETY_MARGIN
                if available_fwd <= 0:
                    self.get_logger().warn(
                        f'Front blocked (clearance {front:.2f} m) — skipping forward move.')
                    delta_to_start = 0.0
                elif available_fwd < delta_to_start:
                    # Can't reach sweep_start; go as far as safely possible
                    sweep_start    = d - available_fwd
                    delta_to_start = available_fwd
                    self.get_logger().warn(
                        f'Front clearance {front:.2f} m limits forward move to '
                        f'{available_fwd:.2f} m — sweep starts at {sweep_start:.2f} m.')

        elif delta_to_start < 0:
            # Backing away from wall — verify rear clearance.
            rear = self.rear_distance()
            if rear is None:
                self.get_logger().warn('Cannot read rear distance — skipping back-up.')
                delta_to_start = 0.0
            else:
                available_back = rear - SAFETY_MARGIN
                if available_back <= 0:
                    self.get_logger().warn(
                        f'Rear blocked (clearance {rear:.2f} m) — skipping back-up.')
                    delta_to_start = 0.0
                elif available_back < abs(delta_to_start):
                    # Can't reach sweep_start; back up as far as safely possible
                    sweep_start    = d + available_back
                    delta_to_start = -available_back
                    self.get_logger().warn(
                        f'Rear clearance {rear:.2f} m limits back-up to '
                        f'{available_back:.2f} m — sweep starts at {sweep_start:.2f} m.')

        actual = self._send_move(delta_to_start)
        current_d = d - math.copysign(actual, delta_to_start)

        # ── 4. Single-pass sweep — probe every PROBE_STEP ─────────────
        # For backward sweeps (sweep_dir=+1) also cap sweep_end so the total
        # planned travel never exceeds the current rear clearance.
        if sweep_dir == 1:
            rear = self.rear_distance()
            if rear is None or rear - SAFETY_MARGIN <= 0:
                self.get_logger().warn(
                    f'Rear clearance {rear} m too small — limiting sweep to current position.')
                sweep_end = current_d
            else:
                available_rear = rear - SAFETY_MARGIN
                max_safe_end   = current_d + available_rear
                if max_safe_end < sweep_end:
                    self.get_logger().warn(
                        f'Rear clearance {rear:.2f} m caps backward sweep '
                        f'to {max_safe_end:.2f} m (was {sweep_end:.2f} m).')
                    sweep_end = max_safe_end

        _farthest_point = max(sweep_start, sweep_end)
        n_steps = int(round(abs(sweep_end - sweep_start) / PROBE_STEP))
        accessible_positions = []   # wall distances where rover is accessible
        _window_scan_done = False

        for step_idx in range(n_steps + 1):
            # Clamp to avoid floating-point overshoot at boundaries
            probe_d = max(MIN_DISTANCE, min(MAX_DISTANCE, current_d))

            self.get_logger().info(f'Probing {probe_d:.2f} m ...')
            if self.is_accessible():
                accessible_positions.append(probe_d)
                self.get_logger().info(f'  -> ACCESSIBLE at {probe_d:.2f} m')
            else:
                self.get_logger().info(f'  -> blocked at {probe_d:.2f} m')

            # At the farthest probe point, scan for window grid coordinates.
            if not _window_scan_done and abs(probe_d - _farthest_point) < PROBE_STEP / 2:
                self.get_logger().info(
                    f'At farthest probe point ({probe_d:.2f} m) — scanning for window coordinates.'
                )
                self._scan_for_window_coords()
                _window_scan_done = True

            # Advance to next probe point (skip after last step)
            if step_idx < n_steps:
                # sweep_dir +1 → distance increases → move away from wall → _send_move < 0
                actual = self._send_move(-sweep_dir * PROBE_STEP)
                if actual < PROBE_STEP - 1e-3:
                    # Obstacle blocked the step — record where we stopped and abort sweep
                    current_d += sweep_dir * actual
                    self.get_logger().warn(
                        f'Sweep blocked at {current_d:.2f} m after {step_idx + 1} probe(s) '
                        f'— stopping early.')
                    break
                current_d += sweep_dir * actual

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
        # Re-read actual position before final move to absorb any clamping drift
        current_d = self.front_distance() or current_d
        move_needed = current_d - best_d
        if abs(move_needed) > POSITION_TOL:
            self._send_move(move_needed)

        final = self.front_distance()
        self.get_logger().info(
            f'Positioned at {final:.2f} m from wall '
            f'(target {best_d:.2f} m).')
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
