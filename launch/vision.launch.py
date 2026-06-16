"""
vision.launch.py

Bridges the x500_mono_cam Gazebo camera into ROS 2 and runs the ultralytics_ros
YOLO tracker on it, publishing /yolo_result for mission_executor's "follow" steps.

Prerequisites: PX4 SITL running the mono-camera model
  (cd ~/PX4-Autopilot && make px4_sitl gz_x500_mono_cam)
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    image_bridge = Node(
        package='ros_gz_image',
        executable='image_bridge',
        arguments=['/camera/color/image_raw'],
        output='screen',
    )
    tracker = Node(
        package='ultralytics_ros',
        executable='tracker_node.py',
        parameters=[{
            'input_topic': '/camera/color/image_raw',
            'result_topic': '/yolo_result',
            'result_image_topic': '/yolo_image',
            'yolo_model': 'yolo26n.pt',
            'classes': [0, 2, 9],  # COCO: person, car, traffic light
            'conf_thres': 0.15,  # lowered from the 0.25 default to catch harder
                                  # angles (e.g. a car seen from behind)
            'device': 'cpu',
            'tracker': 'bytetrack.yaml',
        }],
        output='screen',
    )
    return LaunchDescription([image_bridge, tracker])
