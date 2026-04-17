#!/usr/bin/env python
# encoding: utf-8

"""
WallCenteringCtrl: after perpendicular wall alignment, capture a camera
image, call the object-detection REST API, and laterally shift the rover
until the highest-confidence detection is centred in frame.

Used between Phase 2 (perpendicular align) and Phase 3 (ideal distance)
of the inspection pipeline.
"""

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

from .lidar_scan_utils import process_scan_for_rover


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

IMAGE_SAVE_PATH     = '/tmp/wall_capture.png'
DETECT_API_URL      = 'http://127.0.0.1:8000/detect'

# Horizontal pixel distance from image centre that counts as "centred"
CENTERING_MARGIN_PX = 10

# Half-width of the front LiDAR sector used to estimate wall distance
FRONT_HALF_DEG      = 10.0

# Horizontal field of view of the camera in degrees
CAMERA_HFOV_DEG     = 80.0

# Physical speed constants (must match behavior_ctrl.py)
DRIVE_SPEED_M_S     = 0.2   # drive_on_heading / back_up speed
SPIN_90_SLEEP_S     = 12.0   # seconds to allow a 90° spin to complete

# Pan-tilt vertical scan constants
PT_TILT_MIN_RAD       = -0.5   # Tilt down limit
PT_TILT_MAX_RAD       =  1.0   # Tilt up limit
PT_TILT_STEP_RAD      =  0.1   # Step between scan positions
PT_TILT_SETTLE_S      =  0.5   # Seconds to wait after moving before capturing
WINDOW_MIN_CONFIDENCE =  0.70  # Detection confidence threshold for scan to stop


class WallCenteringCtrl(Node):
    """
    Captures the front camera image, detects objects via a REST API, and
    laterally shifts the rover to centre the highest-confidence detection.

    Lateral movement sequence:
        1. Spin ±90° to face along the wall.
        2. Drive forward by the required lateral offset.
        3. Spin ∓90° to face the wall again.
    """

    def __init__(self):
        super().__init__('wall_centering_ctrl')

        self._image: RosImage | None = None
        self._image_event = threading.Event()

        self._camera_info: CameraInfo | None = None
        self._camera_info_event = threading.Event()

        self._scan: LaserScan | None = None
        self._scan_event = threading.Event()

        self.create_subscription(RosImage,    '/pt_camera/image_raw',    self._image_cb,       1)
        self.create_subscription(CameraInfo,  '/pt_camera/camera_info',  self._camera_info_cb, 1)
        self.create_subscription(LaserScan,   'scan',                    self._scan_cb,        10)

        self._joint_pub       = self.create_publisher(JointState, '/ugv/joint_states', 10)
        self._behavior_client = ActionClient(self, Behavior, 'behavior')

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _image_cb(self, msg: RosImage) -> None:
        self._image = msg
        self._image_event.set()

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        # Only need the first message; intrinsics don't change.
        if self._camera_info is None:
            self._camera_info = msg
            self._camera_info_event.set()

    def _scan_cb(self, msg: LaserScan) -> None:
        self._scan = process_scan_for_rover(msg)
        self._scan_event.set()

    # ------------------------------------------------------------------
    # Image capture
    # ------------------------------------------------------------------

    def _capture_image(self) -> str:
        """
        Block until a fresh camera frame arrives, save it as a PNG, and
        return the file path.

        Raises RuntimeError if no frame arrives within the timeout.
        """
        self._image_event.clear()
        if not self._image_event.wait(timeout=10.0) or self._image is None:
            raise RuntimeError('No camera frame received within 10 s.')

        msg = self._image
        encoding = msg.encoding.lower()

        # Decode raw bytes into an H×W×C numpy array.
        if 'mono' in encoding:
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            rgb = np.stack([raw] * 3, axis=-1)
        else:
            channels = 4 if encoding in ('rgba8', 'bgra8') else 3
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
            if 'bgr' in encoding:
                rgb = raw[:, :, 2::-1]   # BGR(A) → RGB
            else:
                rgb = raw[:, :, :3]      # RGB(A) → RGB

        PILImage.fromarray(rgb, 'RGB').save(IMAGE_SAVE_PATH)
        self.get_logger().info(f'Image saved: {IMAGE_SAVE_PATH}')
        return IMAGE_SAVE_PATH

    # ------------------------------------------------------------------
    # Detection API
    # ------------------------------------------------------------------

    def _call_detection_api(self, image_path: str) -> dict | None:
        """POST the image to the detection API and return the parsed JSON."""
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

    # ------------------------------------------------------------------
    # Centering geometry
    # ------------------------------------------------------------------

    def _horizontal_pixel_offset(self, detection: dict, image_width: int) -> float:
        """
        Signed pixel offset from the image centre to the detection box centre.
        Positive → detection is to the right of centre.
        """
        box = detection['box']
        box_centre_x = (box['x1'] + box['x2']) / 2.0
        return box_centre_x - image_width / 2.0

    def _pixels_to_meters(self, pixel_offset: float, px_per_meter: float) -> float:
        """
        Convert a horizontal pixel offset to a lateral distance in metres.

        px_per_meter = image_width_px / (2 * tan(CAMERA_HFOV_DEG/2) * wall_distance)
        """
        return pixel_offset / px_per_meter

    def _front_wall_distance(self) -> float | None:
        """Return the median LiDAR distance in the front ±FRONT_HALF_DEG sector."""
        self._scan_event.clear()
        if not self._scan_event.wait(timeout=5.0) or self._scan is None:
            return None

        scan = self._scan
        half_rad = math.radians(FRONT_HALF_DEG)
        dists = [
            r for i, r in enumerate(scan.ranges)
            if math.isfinite(r) and scan.range_min <= r <= scan.range_max
            and abs(scan.angle_min + i * scan.angle_increment) <= half_rad
        ]
        return float(np.median(dists)) if dists else None

    # ------------------------------------------------------------------
    # Lateral movement
    # ------------------------------------------------------------------

    def _send_behavior(self, commands: list) -> None:
        """Send a list of behavior commands and block until accepted."""
        goal = Behavior.Goal()
        goal.command = json.dumps(commands)
        done = threading.Event()

        def _result_cb(_):
            done.set()

        def _goal_cb(future):
            future.result().get_result_async().add_done_callback(_result_cb)

        self._behavior_client.send_goal_async(goal).add_done_callback(_goal_cb)
        done.wait()

    def _lateral_move(self, offset_m: float) -> None:
        """
        Shift the rover laterally by offset_m metres parallel to the wall.

        Positive offset → move right (spin +90°, drive, spin −90°).
        Negative offset → move left  (spin −90°, drive, spin +90°).
        """
        spin_towards  = -90.0 if offset_m > 0 else  90.0
        spin_back     = -spin_towards
        distance      = abs(offset_m)
        drive_sleep   = distance / DRIVE_SPEED_M_S + 2.0   # buffer

        self.get_logger().info(
            f'Lateral shift: {offset_m:+.3f} m  '
            f'(spin {spin_towards:+.0f}° → drive {distance:.3f} m → spin {spin_back:+.0f}°)'
        )

        self._send_behavior([{'type': 'spin', 'data': spin_towards}])
        time.sleep(SPIN_90_SLEEP_S)

        self._send_behavior([{'type': 'drive_on_heading', 'data': distance}])
        time.sleep(drive_sleep)

        self._send_behavior([{'type': 'spin', 'data': spin_back}])
        time.sleep(SPIN_90_SLEEP_S)

    # ------------------------------------------------------------------
    # Pan-tilt control
    # ------------------------------------------------------------------

    def _set_pt_tilt(self, tilt_rad: float) -> None:
        """Publish a joint state that sets the pan-tilt tilt to *tilt_rad*.
        Pan is always reset to 0 (facing the wall straight on).
        """
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

    def _scan_for_window(self) -> tuple[dict | None, float]:
        """
        Sweep the pan-tilt camera vertically from PT_TILT_MIN_RAD to
        PT_TILT_MAX_RAD in steps of PT_TILT_STEP_RAD.

        At each position: capture an image, call the detection API, and
        check whether any detection has confidence ≥ WINDOW_MIN_CONFIDENCE.
        Stops at the first position that meets the threshold.

        Returns:
            (detection, tilt_rad) — the best detection found and the tilt
            angle at which it was found.  detection is None if no qualifying
            frame was seen during the entire sweep; tilt_rad will be 0.0.
        """
        tilt_positions = list(np.arange(PT_TILT_MIN_RAD, PT_TILT_MAX_RAD + PT_TILT_STEP_RAD / 2,
                                        PT_TILT_STEP_RAD))

        self.get_logger().info(
            f'Scanning vertically: {len(tilt_positions)} positions '
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
            if best is not None and best['confidence'] >= WINDOW_MIN_CONFIDENCE:
                self.get_logger().info(
                    f'  Window found at tilt={tilt:+.2f} rad  '
                    f'label={best["label"]}  confidence={best["confidence"]:.3f}'
                )
                return best, tilt

        self.get_logger().warn('Vertical scan complete — no qualifying window found.')
        return None, 0.0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> bool:
        """
        Full centering sequence:
          1. Wait for camera intrinsics.
          2. Scan the pan-tilt camera vertically until a window is detected
             with confidence ≥ WINDOW_MIN_CONFIDENCE.
          3. Use the detection from the scan to compute the horizontal
             pixel offset.
          4. If the detection is off-centre beyond the margin, compute the
             lateral offset in metres and shift the rover.
          5. Reset the pan-tilt to 0° after the shift.

        Returns:
            True  — rover is centred (or was already, or no detections).
            False — a required sensor/API was unavailable.
        """
        self.get_logger().info('Waiting for camera info...')
        if not self._camera_info_event.wait(timeout=10.0) or self._camera_info is None:
            self.get_logger().error('No camera info — skipping wall centering.')
            return False

        image_width = self._camera_info.width

        self.get_logger().info('Scanning vertically for a window...')
        detection, found_tilt = self._scan_for_window()

        if detection is None:
            self.get_logger().warn('No qualifying window found during scan — skipping lateral shift.')
            self._set_pt_tilt(0.0)
            return True   # Not a hard failure; proceed to Phase 3.

        self.get_logger().info(
            f'Using detection: label={detection["label"]}  '
            f'confidence={detection["confidence"]:.3f}  box={detection["box"]}  '
            f'tilt={found_tilt:+.2f} rad'
        )

        pixel_offset = self._horizontal_pixel_offset(detection, image_width)
        self.get_logger().info(f'Pixel offset from image centre: {pixel_offset:+.1f} px')

        if abs(pixel_offset) <= CENTERING_MARGIN_PX:
            self.get_logger().info('Detection is within margin — no shift needed.')
            self._set_pt_tilt(0.0)
            return True

        wall_distance = self._front_wall_distance()
        if wall_distance is None:
            self.get_logger().error('Cannot read wall distance — skipping lateral shift.')
            self._set_pt_tilt(0.0)
            return False

        self.get_logger().info(f'Estimated wall distance: {wall_distance:.3f} m')

        image_width_m = 2 * math.tan(math.radians(CAMERA_HFOV_DEG / 2)) * wall_distance
        px_per_meter  = image_width / image_width_m

        lateral_m = self._pixels_to_meters(pixel_offset, px_per_meter)
        self.get_logger().info(f'Required lateral shift: {lateral_m:+.3f} m')

        self._lateral_move(lateral_m)
        self._set_pt_tilt(0.0)
        return True
