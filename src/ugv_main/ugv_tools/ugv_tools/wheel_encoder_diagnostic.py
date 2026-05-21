#!/usr/bin/env python3
import argparse
import csv
import math
import os
import statistics
import time
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


@dataclass
class Sample:
    stamp: float
    phase: str
    left: float
    right: float


@dataclass
class Phase:
    name: str
    linear_x: float
    angular_z: float
    duration: float


class WheelEncoderDiagnostic(Node):
    def __init__(self, args):
        super().__init__("wheel_encoder_diagnostic")
        self.args = args
        self.samples = []
        self.last_sample_time = None
        self.current_phase = "idle"
        self.cmd_pub = self.create_publisher(Twist, args.cmd_topic, 10)
        self.create_subscription(
            Float32MultiArray,
            args.odom_topic,
            self.odom_callback,
            100,
        )

    def odom_callback(self, msg):
        if len(msg.data) < 2:
            self.get_logger().warn(
                f"Ignoring {self.args.odom_topic}: expected at least 2 values, got {len(msg.data)}"
            )
            return
        stamp = time.monotonic()
        self.last_sample_time = stamp
        self.samples.append(
            Sample(
                stamp=stamp,
                phase=self.current_phase,
                left=float(msg.data[0]),
                right=float(msg.data[1]),
            )
        )

    def publish_cmd(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def stop(self):
        for _ in range(5):
            self.publish_cmd(0.0, 0.0)
            rclpy.spin_once(self, timeout_sec=0.02)

    def wait_for_odom(self):
        deadline = time.monotonic() + self.args.wait_timeout
        while rclpy.ok() and self.last_sample_time is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.last_sample_time is not None

    def run_phase(self, phase):
        self.current_phase = phase.name
        start = time.monotonic()
        next_log = start
        while rclpy.ok() and time.monotonic() - start < phase.duration:
            if not self.args.listen_only:
                self.publish_cmd(phase.linear_x, phase.angular_z)
            rclpy.spin_once(self, timeout_sec=0.02)
            if time.monotonic() >= next_log:
                self.get_logger().info(
                    f"{phase.name}: cmd linear.x={phase.linear_x:.3f}, angular.z={phase.angular_z:.3f}"
                )
                next_log += 1.0

        self.stop()
        self.current_phase = "settle"
        settle_end = time.monotonic() + self.args.settle
        while rclpy.ok() and time.monotonic() < settle_end:
            rclpy.spin_once(self, timeout_sec=0.05)


def phase_delta(samples):
    if len(samples) < 2:
        return None
    return samples[-1].left - samples[0].left, samples[-1].right - samples[0].right


def finite_ratio(numerator, denominator):
    if abs(denominator) < 1e-9:
        return math.inf
    return numerator / denominator


def summarize_phase(name, samples, asymmetry_threshold):
    delta = phase_delta(samples)
    if delta is None:
        return {
            "phase": name,
            "samples": len(samples),
            "status": "insufficient samples",
        }

    left_delta, right_delta = delta
    left_abs = abs(left_delta)
    right_abs = abs(right_delta)
    mean_abs = (left_abs + right_abs) / 2.0
    difference = left_abs - right_abs
    difference_pct = 0.0 if mean_abs < 1e-9 else 100.0 * difference / mean_abs
    ratio = finite_ratio(left_abs, right_abs)

    intervals = [
        samples[i].stamp - samples[i - 1].stamp
        for i in range(1, len(samples))
        if samples[i].stamp > samples[i - 1].stamp
    ]
    rate_hz = 0.0
    if intervals:
        rate_hz = 1.0 / statistics.mean(intervals)

    status = "ok"
    if mean_abs < 1e-5:
        status = "no encoder movement"
    elif abs(difference_pct) > asymmetry_threshold:
        slower = "left" if left_abs < right_abs else "right"
        status = f"{slower} side lower by {abs(difference_pct):.1f}%"

    return {
        "phase": name,
        "samples": len(samples),
        "left_delta": left_delta,
        "right_delta": right_delta,
        "abs_left_delta": left_abs,
        "abs_right_delta": right_abs,
        "abs_left_over_abs_right": ratio,
        "abs_difference_pct": difference_pct,
        "sample_rate_hz": rate_hz,
        "status": status,
    }


def write_csv(path, samples):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["monotonic_time_s", "phase", "left_odom", "right_odom"])
        for sample in samples:
            writer.writerow([f"{sample.stamp:.6f}", sample.phase, sample.left, sample.right])


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Drive short test phases and compare left/right /odom/odom_raw encoder deltas."
    )
    parser.add_argument("--cmd-topic", default="/cmd_vel")
    parser.add_argument("--odom-topic", default="/odom/odom_raw")
    parser.add_argument("--linear-speed", type=float, default=0.12)
    parser.add_argument("--angular-speed", type=float, default=0.35)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--settle", type=float, default=1.0)
    parser.add_argument("--wait-timeout", type=float, default=5.0)
    parser.add_argument("--asymmetry-threshold-pct", type=float, default=12.0)
    parser.add_argument(
        "--listen-only",
        action="store_true",
        help="Do not publish /cmd_vel; only log encoder data while you move/command the rover separately.",
    )
    parser.add_argument(
        "--skip-spin",
        action="store_true",
        help="Only run forward/backward phases.",
    )
    parser.add_argument(
        "--csv",
        default="wheel_encoder_diagnostic.csv",
        help="Output CSV path.",
    )
    return parser


def main(args=None):
    parsed = build_arg_parser().parse_args(args)
    rclpy.init(args=None)
    node = WheelEncoderDiagnostic(parsed)

    phases = [
        Phase("forward", parsed.linear_speed, 0.0, parsed.duration),
        Phase("backward", -parsed.linear_speed, 0.0, parsed.duration),
    ]
    if not parsed.skip_spin:
        phases.extend(
            [
                Phase("spin_left", 0.0, parsed.angular_speed, parsed.duration),
                Phase("spin_right", 0.0, -parsed.angular_speed, parsed.duration),
            ]
        )

    try:
        node.get_logger().info(f"Waiting for encoder samples on {parsed.odom_topic}...")
        if not node.wait_for_odom():
            node.get_logger().error(
                f"No samples received on {parsed.odom_topic} within {parsed.wait_timeout:.1f}s."
            )
            return 2

        if parsed.listen_only:
            node.get_logger().info("listen-only mode: logging until Ctrl+C.")
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)
        else:
            node.get_logger().info(
                "Starting motion test. Keep the rover lifted or in a clear, open area."
            )
            for phase in phases:
                node.run_phase(phase)

        write_csv(parsed.csv, node.samples)
        print(f"\nSaved raw samples to {os.path.abspath(parsed.csv)}")
        print("\nEncoder summary:")
        for phase in phases:
            phase_samples = [sample for sample in node.samples if sample.phase == phase.name]
            summary = summarize_phase(
                phase.name,
                phase_samples,
                parsed.asymmetry_threshold_pct,
            )
            print(
                "  {phase}: samples={samples}, left_delta={left_delta}, "
                "right_delta={right_delta}, abs_L/R={ratio}, status={status}".format(
                    phase=summary["phase"],
                    samples=summary["samples"],
                    left_delta=f"{summary.get('left_delta', 0.0):.4f}"
                    if "left_delta" in summary
                    else "n/a",
                    right_delta=f"{summary.get('right_delta', 0.0):.4f}"
                    if "right_delta" in summary
                    else "n/a",
                    ratio=f"{summary.get('abs_left_over_abs_right', 0.0):.3f}"
                    if "abs_left_over_abs_right" in summary
                    else "n/a",
                    status=summary["status"],
                )
            )
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted; stopping rover and writing collected samples.")
        node.stop()
        write_csv(parsed.csv, node.samples)
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
