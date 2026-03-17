#!/usr/bin/env python
# encoding: utf-8

"""
Controller that first aligns the rover parallel to a wall using AlignCtrl,
then starts the LLM-based pan-tilt inspection agent (LlmPtCtrl).
"""

import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor

from .align_ctrl import AlignCtrl
from .distance_ctrl import DistanceCtrl
from .llm_pt_ctrl import LlmPtCtrl


def main(args=None):
    rclpy.init(args=args)
    
    # ------------------------------------------------------------------
    # Phase 1: Align rover perpendicular to wall
    # ------------------------------------------------------------------
    align_node = AlignCtrl('perpendicular')
    align_done = threading.Event()

    def _run_align():
        time.sleep(1.0)  # Allow subscriptions / action server to connect
        align_node.align()
        align_done.set()

    align_thread = threading.Thread(target=_run_align, daemon=True)
    align_thread.start()

    executor = SingleThreadedExecutor()
    executor.add_node(align_node)
    try:
        while not align_done.is_set():
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        executor.remove_node(align_node)
        align_node.destroy_node()
        rclpy.shutdown()
        return

    align_node.get_logger().info('Perpendicular Alignment complete. Finding ideal distance.')
    executor.remove_node(align_node)
    align_node.destroy_node()
    
    time.sleep(5)

    # ------------------------------------------------------------------
    # Phase 2: Find ideal distance from the wall
    # ------------------------------------------------------------------
    dist_node = DistanceCtrl()
    dist_done = threading.Event()

    def _run_dist():
        time.sleep(0.5)
        best = dist_node.find_accessible_distance()
        if best is None:
            dist_node.get_logger().error('No accessible distance found — pipeline aborted.')
        dist_done.set()

    dist_thread = threading.Thread(target=_run_dist, daemon=True)
    dist_thread.start()

    executor = SingleThreadedExecutor()
    executor.add_node(dist_node)
    try:
        while not dist_done.is_set():
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        executor.remove_node(dist_node)
        dist_node.destroy_node()
        rclpy.shutdown()
        return

    executor.remove_node(dist_node)
    dist_node.destroy_node()
    
    time.sleep(5)

    # ------------------------------------------------------------------
    # Phase 3: Align rover parallel to wall
    # ------------------------------------------------------------------
    align_node = AlignCtrl('parallel')
    align_done = threading.Event()

    def _run_align():
        time.sleep(1.0)  # Allow subscriptions / action server to connect
        align_node.align()
        align_done.set()

    align_thread = threading.Thread(target=_run_align, daemon=True)
    align_thread.start()

    executor = SingleThreadedExecutor()
    executor.add_node(align_node)
    try:
        while not align_done.is_set():
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        executor.remove_node(align_node)
        align_node.destroy_node()
        rclpy.shutdown()
        return

    align_node.get_logger().info('Alignment complete. Starting inspection agent.')
    executor.remove_node(align_node)
    align_node.destroy_node()
    
    time.sleep(10)

    # ------------------------------------------------------------------
    # Phase 4: Run LLM pan-tilt inspection controller
    # ------------------------------------------------------------------
    pt_ctrl = LlmPtCtrl('llm_pt_ctrl')
    try:
        rclpy.spin(pt_ctrl)
    except KeyboardInterrupt:
        pass
    finally:
        pt_ctrl.on_shutdown()
        pt_ctrl.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
