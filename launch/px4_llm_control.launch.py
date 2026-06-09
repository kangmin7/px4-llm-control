"""
px4_llm_control.launch.py

Launches the natural-language mission executor and an interactive CLI for typing
plain-English instructions.

Prerequisites (start separately before this launch):
  • PX4 SITL:   make px4_sitl gz_x500          (or any other PX4 Gazebo target)
  • uXRCE-DDS:  MicroXRCEAgent udp4 -p 8888
  • ANTHROPIC_API_KEY set in the environment (the planner calls the Claude API to
    turn instructions into mission steps — `pip install anthropic` if missing)
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    executor = Node(
        package='px4_llm_control',
        executable='mission_executor',
        name='nl_mission_executor',
        output='screen',
    )
    gui = Node(
        package='px4_llm_control',
        executable='command_gui',
        name='nl_command_gui',
        output='screen',
    )
    return LaunchDescription([executor, gui])
