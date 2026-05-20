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
from launch.actions import DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

# Seconds to wait after launching nav before starting behavior_ctrl
NAV_READY_DELAY = 20.0

# Seconds to wait after launching nav before starting inspection_pipeline
# (must be > NAV_READY_DELAY so behavior_ctrl is up first)
PIPELINE_DELAY = 30.0

# Enable debugpy attach for inspection_pipeline so you can break inside
# coord_convert.py while running the full inspection pipeline.
DEBUGPY_PORT = '5678'


def generate_launch_description():
    debug_inspection_pipeline = LaunchConfiguration('debug_inspection_pipeline')
    nav_ready_delay = LaunchConfiguration('nav_ready_delay')
    pipeline_delay = LaunchConfiguration('pipeline_delay')
    use_sim_time = LaunchConfiguration('use_sim_time')
    agent_model = LaunchConfiguration('agent_model')
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
            'use_sim_time': use_sim_time,
        }.items(),
    )

    # ------------------------------------------------------------------
    # 2. Behavior controller (waits for nav stack to be ready)
    # ------------------------------------------------------------------
    behavior_ctrl_cmd = TimerAction(
        period=nav_ready_delay,
        actions=[
            Node(
                package='ugv_tools',
                executable='behavior_ctrl',
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}],
            )
        ],
    )

    # ------------------------------------------------------------------
    # 3. Inspection pipeline (waits for behavior_ctrl to be ready)
    # ------------------------------------------------------------------
    inspection_pipeline_node = Node(
        package='ugv_tools',
        executable='inspection_pipeline',
        output='screen',
        additional_env={'UGV_AGENT_MODEL': agent_model},
        parameters=[{'use_sim_time': use_sim_time}],
        condition=UnlessCondition(debug_inspection_pipeline),
        on_exit=[
            EmitEvent(event=Shutdown(reason='inspection_pipeline completed'))
        ],
    )
    debug_inspection_pipeline_node = Node(
        package='ugv_tools',
        executable='inspection_pipeline',
        output='screen',
        additional_env={
            'UGV_AGENT_MODEL': agent_model,
            'UGV_DEBUGPY_PORT': DEBUGPY_PORT,
        },
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(debug_inspection_pipeline),
        on_exit=[
            EmitEvent(event=Shutdown(reason='inspection_pipeline completed'))
        ],
    )
    inspection_pipeline_cmd = TimerAction(
        period=pipeline_delay,
        actions=[
            inspection_pipeline_node,
            debug_inspection_pipeline_node,
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'debug_inspection_pipeline',
            default_value='false',
            description='Attach debugpy to inspection_pipeline for debugging coord_convert.py',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock for all inspection pipeline nodes.',
        ),
        DeclareLaunchArgument(
            'agent_model',
            default_value=os.environ.get('UGV_AGENT_MODEL', 'gemini-2.5-pro'),
            description='Agent mode or model name passed via UGV_AGENT_MODEL. Use greedy or code for controller-level agents.',
        ),
        DeclareLaunchArgument(
            'nav_ready_delay',
            default_value=str(NAV_READY_DELAY),
            description='Seconds to wait after nav launch before starting behavior_ctrl.',
        ),
        DeclareLaunchArgument(
            'pipeline_delay',
            default_value=str(PIPELINE_DELAY),
            description='Seconds to wait after nav launch before starting inspection_pipeline.',
        ),
        bringup_cmd,
        nav_cmd,
        behavior_ctrl_cmd,
        inspection_pipeline_cmd,
    ])
