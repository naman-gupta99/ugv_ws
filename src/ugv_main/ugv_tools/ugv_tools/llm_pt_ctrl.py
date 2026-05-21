#!/usr/bin/env python
# encoding: utf-8

import cv2
from cv_bridge import CvBridge
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image, LaserScan
from rclpy.action import ActionClient
from ugv_interface.action import Behavior
import json
import threading
import queue
import time

from .agent.audit_toolset import audit_state_instance
from .agent.code_execution_agent import execute_code_audit
from .agent.greedy_baseline import execute_greedy_audit
from .agent.models import Models
from .agent.validation_react_agent import ValidationReactAgent
import os
import pathlib
import tempfile
import traceback

PT_TILT_TOL_RAD = 0.03
PT_VEL_TOL_RAD_S = 0.03
PT_SETTLE_HOLD_S = 0.20
PT_SETTLE_TIMEOUT_S = 4.0
JOINT_FEEDBACK_TOPIC = os.environ.get("UGV_JOINT_FEEDBACK_TOPIC", "/joint_states")

PLATFORM = os.getenv("UGV_PLATFORM", os.getenv("PLATFORM", "SIM")).upper()
TOPICS = {
    "SIM": {
        "image_raw": "/overhead_camera/image_raw",
        "joint_states": "ugv/joint_states"
    },
    "ROVER": {
        "image_raw": "/image_raw",
        "joint_states": "/ugv/joint_states"
    }
}
if PLATFORM not in TOPICS:
    raise ValueError(f"Unsupported UGV platform {PLATFORM!r}; expected one of {sorted(TOPICS)}")

class LlmPtCtrl(Node):

    def set_new_angles(self, dx, y_rad):
        if dx != 0:
            command_type = 'drive_on_heading' if dx > 0 else 'back_up'
            goal = Behavior.Goal()
            goal.command = json.dumps([{'type': command_type, 'data': abs(dx)}])

            done_event = threading.Event()
            outcome = {'success': False, 'message': ''}

            def _result_cb(_):
                result = _.result()
                behavior_result = result.result
                outcome['success'] = bool(getattr(behavior_result, 'result', False))
                outcome['message'] = getattr(behavior_result, 'message', '')
                done_event.set()

            def _goal_cb(future):
                handle = future.result()
                if not handle.accepted:
                    outcome['message'] = f'{command_type} goal was rejected.'
                    done_event.set()
                    return
                handle.get_result_async().add_done_callback(_result_cb)

            self._behavior_client.send_goal_async(goal).add_done_callback(_goal_cb)
            done_event.wait()
            if not outcome['success']:
                self.motion_failed_reason = outcome['message'] or 'Behavior command failed.'
                self.motion_failed_event.set()
                self.get_logger().warn(self.motion_failed_reason)
                return False
        self.y_rad = y_rad
            
        self.publish_joint_state()

        self._wait_pt_settle(self.x_rad, self.y_rad)
        # Allow capture only after a new frame arrives post movement/tilt update.
        self._capture_not_before = time.monotonic()

        return True

    def _wait_for_fresh_image(self, timeout_s: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self.image_lock:
                has_fresh_frame = (
                    self.curr_image_raw is not None
                    and self._last_image_rx_time is not None
                    and self._last_image_rx_time >= self._capture_not_before
                )
            if has_fresh_frame:
                return True
            time.sleep(0.02)
        return False
    
    def get_laser_scan(self):
        return self.curr_lidar_scan
    
    def get_current_angles(self):
        return 0.0, self.y_rad

    def __init__(self, name):
        super().__init__(name)
        self.get_logger().info(
            f"LlmPtCtrl started with model={os.environ.get('UGV_AGENT_MODEL', Models.DEFAULT_MODEL)} "
            f"hint={os.environ.get('UGV_AGENT_HINT', '') or 'none'}"
        )

        # Subscriptions
        self.image_raw_sub = self.create_subscription(
            Image, TOPICS[PLATFORM]["image_raw"], self.image_raw_callback, 10
        )
        self.lidar_sub = self.create_subscription(
            LaserScan, "scan", self.lidar_callback, 10
        )

        # Publisher
        self.pub_cmdJoint = self.create_publisher(JointState, TOPICS[PLATFORM]["joint_states"], 10)

        # Pan-Tilt Camera State
        self.x_rad = math.pi / 2
        self.y_rad = 0.0

        # Image saving state
        self.curr = 0
        self.pictures_taken = 0
        self.bridge = CvBridge()
        self.curr_image_raw = None
        self.image_lock = threading.Lock()
        self._last_image_rx_time = None
        self._capture_not_before = 0.0
        
        # Laser Scan state
        self.curr_lidar_scan = None

        # Pan-tilt feedback state from Gazebo joint states
        self._pt_feedback_event = threading.Event()
        self._pt_cmd_feedback_event = threading.Event()
        self._pt_pan = None
        self._pt_tilt = None
        self._pt_cmd_pan = None
        self._pt_cmd_tilt = None
        self._pt_pan_vel = 0.0
        self._pt_tilt_vel = 0.0
        self._last_joint_stamp = None
        self._last_pan = None
        self._last_tilt = None
        self.motion_failed_event = threading.Event()
        self.motion_failed_reason = None
        self.joint_feedback_sub = self.create_subscription(
            JointState, JOINT_FEEDBACK_TOPIC, self.joint_feedback_callback, 10
        )
        self.joint_cmd_feedback_sub = self.create_subscription(
            JointState, TOPICS[PLATFORM]["joint_states"], self.joint_command_callback, 10
        )

        # Action client for rover drive commands
        self._behavior_client = ActionClient(self, Behavior, 'behavior')

        # Capture directory (configurable via env var). Prefer mounted workspace if present,
        # otherwise fall back to user's home directory under ~/ugv_ws/captures.
        env_dir = os.environ.get("UGV_CAPTURE_DIR")
        default_workspace_dir = "/home/ws/ugv_ws/captures"
        home_fallback = os.path.expanduser("~/ugv_ws/captures")
        if env_dir:
            self.capture_base_dir = env_dir
        elif os.path.isdir(os.path.dirname(default_workspace_dir)):
            # If /home/ws/ugv_ws exists on the system, use it (common for your setup)
            self.capture_base_dir = default_workspace_dir
        else:
            self.capture_base_dir = home_fallback

        run_ts = time.strftime("%Y%m%d%H%M%S")
        self._run_dir = os.path.join(self.capture_base_dir, f"run_{run_ts}")

        # Timer to periodically call publish_joint_state
        # self.timer = self.create_timer(0.1, self.publish_joint_state)

        # Agent Controller
        audit_state_instance.update_rover_state_func = self.set_new_angles
        audit_state_instance.get_laser_scan_func = self.get_laser_scan
        audit_state_instance.get_current_angles_func = self.get_current_angles
        audit_state_instance.capture_image_func = self.capture_image
        self.validation_agent = None
        self.validation_agent_metrics = None
        self.validation_agent_model_name = None
        self.validation_agent_thread_id = None
        self.validation_agent_hint = os.environ.get("UGV_AGENT_HINT", "")
        
        self.validation_agent_thread = threading.Thread(
            target=self._run_validation_agent, daemon=True
        )
        self.validation_agent_thread.start()

    # Callback for image_raw
    def image_raw_callback(self, msg):
        with self.image_lock:
            self.curr_image_raw = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
            self._last_image_rx_time = time.monotonic()
            
    def lidar_callback(self, msg):
        self.curr_lidar_scan = msg

    def joint_feedback_callback(self, msg: JointState):
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
        self._pt_feedback_event.set()

    def joint_command_callback(self, msg: JointState):
        if msg.header.frame_id != 'ugv_joint_state':
            return

        try:
            pan_idx = msg.name.index('pt_base_link_to_pt_link1')
            tilt_idx = msg.name.index('pt_link1_to_pt_link2')
        except ValueError:
            return

        self._pt_cmd_pan = msg.position[pan_idx]
        self._pt_cmd_tilt = msg.position[tilt_idx]
        self._pt_cmd_feedback_event.set()

    def _wait_pt_settle(self, target_pan: float, target_tilt: float, timeout_s: float = PT_SETTLE_TIMEOUT_S) -> bool:
        if not self._pt_feedback_event.wait(timeout=timeout_s):
            if self._pt_cmd_feedback_event.is_set():
                cmd_target_ok = (
                    self._pt_cmd_pan is not None and
                    self._pt_cmd_tilt is not None and
                    abs(self._pt_cmd_pan - target_pan) <= PT_TILT_TOL_RAD and
                    abs(self._pt_cmd_tilt - target_tilt) <= PT_TILT_TOL_RAD
                )
                if cmd_target_ok:
                    self.get_logger().warn(
                        f'No {JOINT_FEEDBACK_TOPIC} feedback while waiting for pan-tilt settle; '
                        f'using commanded state from {TOPICS[PLATFORM]["joint_states"]}.'
                    )
                    time.sleep(PT_SETTLE_HOLD_S)
                    return True

            self.get_logger().warn(
                f'No {JOINT_FEEDBACK_TOPIC} data while waiting for pan-tilt settle.'
            )
            return False

        start = time.monotonic()
        settled_since = None
        while time.monotonic() - start < timeout_s:
            if self._pt_pan is None or self._pt_tilt is None:
                time.sleep(0.02)
                continue

            target_ok = (
                abs(self._pt_pan - target_pan) <= PT_TILT_TOL_RAD and
                abs(self._pt_tilt - target_tilt) <= PT_TILT_TOL_RAD
            )
            vel_ok = (
                abs(self._pt_pan_vel) <= PT_VEL_TOL_RAD_S and
                abs(self._pt_tilt_vel) <= PT_VEL_TOL_RAD_S
            )

            if target_ok and vel_ok:
                if settled_since is None:
                    settled_since = time.monotonic()
                if time.monotonic() - settled_since >= PT_SETTLE_HOLD_S:
                    return True
            else:
                settled_since = None

            time.sleep(0.02)

        self.get_logger().warn(
            f'Pan-tilt settle timeout at pan={target_pan:+.2f}, tilt={target_tilt:+.2f} rad.'
        )
        return False

    def capture_image(self):
        if not self._wait_for_fresh_image(timeout_s=1.0):
            self.get_logger().warn(
                "Timed out waiting for a fresh post-move image frame; using latest available frame."
            )

        # Process image if available
        image_to_save = None
        with self.image_lock:
            if self.curr_image_raw is not None:
                image_to_save = self.curr_image_raw.copy()

        if image_to_save is not None:
            coord = audit_state_instance.current_coordinates
            fname = f"x{coord['x']}_y{coord['y']}_{self.curr}.png"
            try:
                pathlib.Path(self._run_dir).mkdir(parents=True, exist_ok=True)
                filename = os.path.join(self._run_dir, fname)
                ok = cv2.imwrite(filename, image_to_save)
                if ok:
                    self.get_logger().info(
                        f"Image saved successfully as {filename}"
                    )
                else:
                    # cv2 can fail silently; log and fallback to temp
                    raise IOError("cv2.imwrite returned False")
            except Exception as exc:
                # Log full traceback to help debugging permission or filesystem issues
                tb = traceback.format_exc()
                self.get_logger().error(
                    f"Failed to save image to {self._run_dir}: {exc}\n{tb}"
                )
                # Fallback: create a temp directory we can write to
                try:
                    tmpdir = tempfile.mkdtemp(prefix="ugv_captures_")
                    tmpfile = os.path.join(tmpdir, fname)
                    cv2.imwrite(tmpfile, image_to_save)
                    self.get_logger().info(
                        f"Image saved to fallback location {tmpfile}"
                    )
                except Exception as exc2:
                    self.get_logger().error(
                        f"Fallback save also failed: {exc2}\n{traceback.format_exc()}"
                    )
            self.curr += 1
            self._record_picture_taken()
        elif image_to_save is None:
            self.get_logger().warn(
                "No image_raw received yet, delaying image capture but publishing joint state."
            )

    def _record_picture_taken(self):
        self.pictures_taken += 1
        if isinstance(self.validation_agent_metrics, dict):
            self.validation_agent_metrics["pictures_taken"] = self.pictures_taken
        if self.validation_agent is not None and hasattr(self.validation_agent, "metrics"):
            self.validation_agent.metrics["pictures_taken"] = self.pictures_taken

    def publish_joint_state(self):
        try:

            x_rad = self.x_rad
            y_rad = self.y_rad
            print(f"Publishing joint state with x_rad: {x_rad}, y_rad: {y_rad}")

            # Always publish the joint state with current values
            joint_state = JointState()
            joint_state.header.frame_id = "ugv_joint_state"
            joint_state.header.stamp = self.get_clock().now().to_msg()

            # Assuming the robot has two joints for x and y rotation
            joint_state.name = [
                "left_up_wheel_link_joint",
                "left_down_wheel_link_joint",
                "right_up_wheel_link_joint",
                "right_down_wheel_link_joint",
                "pt_base_link_to_pt_link1",
                "pt_link1_to_pt_link2",
            ]
            joint_state.position = [0.0, 0.0, 0.0, 0.0, x_rad, y_rad]

            self.pub_cmdJoint.publish(joint_state)
        except Exception as exc:
            self.get_logger().error(f"Failed to publish joint state: {exc}")

    def _run_validation_agent(self):
        try:
            model_name = os.environ.get("UGV_AGENT_MODEL", Models.DEFAULT_MODEL)
            self.validation_agent_model_name = model_name
            code_model_prefix = f"{Models.CODE_AGENT_MODEL}-"

            if model_name == Models.GREEDY_MODEL:
                metrics = self._new_agent_metrics()
                self.validation_agent_metrics = metrics
                self.validation_agent_thread_id = "greedy_baseline"
                execute_greedy_audit(
                    metrics,
                    lambda: self._print_agent_metrics(metrics),
                )
                self.validation_agent_metrics = dict(metrics)
                return

            if model_name == Models.CODE_AGENT_MODEL or model_name.startswith(code_model_prefix):
                metrics = self._new_agent_metrics()
                self.validation_agent_metrics = metrics
                selected_llm = None
                if model_name.startswith(code_model_prefix):
                    selected_llm = model_name[len(code_model_prefix):]
                execute_code_audit(
                    metrics,
                    lambda: self._print_agent_metrics(metrics),
                    llm_model_name=selected_llm,
                    hint=self.validation_agent_hint,
                )
                # Update model name to show which actual LLM was used for code generation
                if metrics.get("used_model"):
                    self.validation_agent_model_name = f"code-{metrics['used_model']}"
                self.validation_agent_thread_id = metrics.get("langsmith_thread_id") or "code_execution_agent"
                self.validation_agent_metrics = dict(metrics)
                return

            agent = ValidationReactAgent(hint=self.validation_agent_hint)
            self.validation_agent = agent
            self.validation_agent_thread_id = agent.thread_id
            agent.execute()
            self.validation_agent_metrics = dict(agent.metrics)
        except Exception as exc:
            self.get_logger().error(f"Inspection agent failed: {exc}")

    def _new_agent_metrics(self):
        return {
            # Turn/iteration counts
            "agent_turns": 0,
            "validation_turns": 0,
            "tools_transitions": 0,
            "graph_loops": 0,
            "graph_recursion_errors": 0,

            # Tool usage quality
            "tool_calls_present": 0,
            "tool_calls_missing_fields": 0,
            "tool_calls_nonexistent": 0,
            "tool_calls_with_args": 0,
            "tool_calls_invalid_json": 0,
            "mentioned_tool_but_not_called": 0,
            "no_tool_calls_made": 0,

            # Outcomes
            "missions_completed": 0,
            "pictures_taken": 0,

            # Time
            "start_time": None,
            "end_time": None,
            "duration_sec": None,
        }

    def _print_agent_metrics(self, metrics):
        print("\n=== Agent Run Metrics ===")
        if metrics.get("duration_sec") is not None:
            print(f"Runtime: {metrics['duration_sec']} sec")
        print(f"Graph loops (validation passes): {metrics['graph_loops']}")
        print(
            f"Agent turns: {metrics['agent_turns']} | "
            f"Validation turns: {metrics['validation_turns']}"
        )
        print(f"Transitions to tools (tool attempts): {metrics['tools_transitions']}")
        print(f"Graph recursion errors: {metrics['graph_recursion_errors']}")

        invalid_total = (
            metrics["tool_calls_missing_fields"]
            + metrics["tool_calls_nonexistent"]
            + metrics["tool_calls_with_args"]
            + metrics["tool_calls_invalid_json"]
        )
        print(f"Tool calls present: {metrics['tool_calls_present']}")
        print(f"  - Invalid tool call attempts: {invalid_total}")
        if invalid_total:
            print(f"    - Missing fields: {metrics['tool_calls_missing_fields']}")
            print(f"    - Nonexistent tool: {metrics['tool_calls_nonexistent']}")
            print(f"    - Arguments not allowed: {metrics['tool_calls_with_args']}")
            print(f"    - Invalid JSON args: {metrics['tool_calls_invalid_json']}")
        print(f"Mentioned tool but not called: {metrics['mentioned_tool_but_not_called']}")
        print(f"No tool calls made: {metrics['no_tool_calls_made']}")
        print(f"Missions completed (this run): {metrics['missions_completed']}")
        print(f"Pictures taken: {metrics.get('pictures_taken', 0)}")

    def on_shutdown(self):
        self.y_rad = 0.0
        self.x_rad = 0.0
        self.publish_joint_state()
        self.get_logger().info("Shutting down LlmPtCtrl node.")


def main(args=None):
    rclpy.init(args=args)

    pt_ctrl = LlmPtCtrl("llm_pt_ctrl")

    try:
        rclpy.spin(pt_ctrl)
    except KeyboardInterrupt:
        pass
    finally:
        pt_ctrl.on_shutdown()
        pt_ctrl.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
