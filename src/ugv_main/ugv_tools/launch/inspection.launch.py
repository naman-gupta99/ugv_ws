#!/usr/bin/env python3

"""
Inspection launch file.

Start order:
    1. bringup.launch.py     (Gazebo + robot spawn)
    2. nav.launch.py        (AMCL localisation + DWA local planner)
    3. behavior_ctrl        (after NAV_READY_DELAY seconds)
    4. inspection_pipeline  (after PIPELINE_DELAY seconds)

Tune the delays below if the nav stack or behavior server
need more time to become available on your machine.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

# Seconds to wait after launching nav before starting behavior_ctrl
NAV_READY_DELAY = 10.0

# Seconds to wait after launching nav before starting inspection_pipeline
# (must be > NAV_READY_DELAY so behavior_ctrl is up first)
PIPELINE_DELAY = 15.0

# Enable debugpy attach for inspection_pipeline so you can break inside
# coord_convert.py while running the full inspection pipeline.
DEBUGPY_PORT = '5678'


def generate_launch_description():
    debug_inspection_pipeline = LaunchConfiguration('debug_inspection_pipeline')
    bringup_launch_file = os.path.join(
        get_package_share_directory('ugv_gazebo'), 'launch', 'bringup', 'bringup.launch.py'
    )
    nav_launch_dir = os.path.join(
        get_package_share_directory('ugv_gazebo'), 'launch', 'nav'
    )

    # ------------------------------------------------------------------
    # 1. Gazebo bringup (world + robot spawn)
    # ------------------------------------------------------------------
    bringup_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(bringup_launch_file),
    )

    # ------------------------------------------------------------------
    # 2. Navigation stack (AMCL + DWA)
    # ------------------------------------------------------------------
    nav_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav_launch_dir, 'nav.launch.py')
        ),
        launch_arguments={
            'use_localization': 'amcl',
            'use_localplan': 'dwa',
        }.items(),
    )

    # ------------------------------------------------------------------
    # 2. Behavior controller (waits for nav stack to be ready)
    # ------------------------------------------------------------------
    behavior_ctrl_cmd = TimerAction(
        period=NAV_READY_DELAY,
        actions=[
            Node(
                package='ugv_tools',
                executable='behavior_ctrl',
                output='screen',
            )
        ],
    )

    # ------------------------------------------------------------------
    # 3. Inspection pipeline (waits for behavior_ctrl to be ready)
    # ------------------------------------------------------------------
    inspection_pipeline_cmd = TimerAction(
        period=PIPELINE_DELAY,
        actions=[
            Node(
                package='ugv_tools',
                executable='inspection_pipeline',
                output='screen',
                condition=UnlessCondition(debug_inspection_pipeline),
            ),
            Node(
                package='ugv_tools',
                executable='inspection_pipeline',
                output='screen',
                additional_env={'UGV_DEBUGPY_PORT': DEBUGPY_PORT},
                condition=IfCondition(debug_inspection_pipeline),
            )
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'debug_inspection_pipeline',
            default_value='false',
            description='Attach debugpy to inspection_pipeline for debugging coord_convert.py',
        ),
        bringup_cmd,
        nav_cmd,
        behavior_ctrl_cmd,
        inspection_pipeline_cmd,
    ])
