"""ros_gz_bridge + RuntimeLink in ONE component container with intra-process comms (design live-runtime.md D150).

  ros2 launch ariadne_bridge live_link.launch.py config_file:=<bridge_live.yaml> socket_path:=<run>/link.sock \
      ready_file:=<run>/link.ready

The bridge's own launch action composes it the same way (ros_gz_bridge.actions.RosGzBridge, use_composition), so
this is the ROS-native composition with one more component. use_sim_time: every node follows /clock (arch §5.4).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    cfg = LaunchConfiguration("config_file")
    intra = [{"use_intra_process_comms": True}]
    container = ComposableNodeContainer(
        name="ariadne_live_link",
        namespace="",
        package="rclcpp_components",
        executable="component_container",   # single-threaded: callbacks never race on the socket or counters
        composable_node_descriptions=[
            ComposableNode(package="ros_gz_bridge", plugin="ros_gz_bridge::RosGzBridge", name="ros_gz_bridge",
                           parameters=[{"config_file": cfg, "use_sim_time": True}], extra_arguments=intra),
            ComposableNode(package="ariadne_bridge", plugin="ariadne_bridge::RuntimeLink", name="ariadne_runtime_link",
                           parameters=[{"bridge_config": cfg, "socket_path": LaunchConfiguration("socket_path"),
                                        "ready_file": LaunchConfiguration("ready_file"), "use_sim_time": True}],
                           extra_arguments=intra),
        ],
        output="screen",
    )
    return LaunchDescription([
        DeclareLaunchArgument("config_file", description="the mission's bridge_live.yaml"),
        DeclareLaunchArgument("socket_path", description="the runtime's Unix socket (ariadne_live --link)"),
        DeclareLaunchArgument("ready_file", default_value="", description="written when every topic is matched"),
        container,
    ])
