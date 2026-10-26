"""The live world: gz sim server + the runtime link (design live-runtime.md D150/D155).

  ros2 launch ariadne_sim live_world.launch.py scene_dir:=<scene> run_dir:=<run>

- gz starts PAUSED (no -r): infra/scripts/live.sh unpauses it through /world/<w>/control only after
  <run>/link.ready exists, so no frame is rendered before the runtime can take it (the D40 recorder's rule).
- Headless EGL rendering (arch §5.10); the render device is chosen by the container (sim.sh: NVIDIA or llvmpipe).
- gz runs in run_dir, so each mover's executed-script log (mover_agent_<k>.jsonl, relative in the SDF) lands there.
- The plugin path is the sim image's built mover (GZ_SIM_SYSTEM_PLUGIN_PATH, set by the caller).
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare


def _world(context):
    scene = LaunchConfiguration("scene_dir").perform(context)
    run = LaunchConfiguration("run_dir").perform(context)
    name = LaunchConfiguration("mission").perform(context)
    world = os.path.join(scene, f"mission_{name}.sdf")
    bridge = os.path.join(scene, f"bridge_{name}.yaml")
    for p in (world, bridge):
        if not os.path.isfile(p):
            raise RuntimeError(f"live_world: missing {p} (compose it: worldgen generate --rig ... --live-rtf R)")
    gz = ExecuteProcess(cmd=["gz", "sim", "-s", "--headless-rendering", world], cwd=run, output="screen",
                        name="gz_sim")
    link = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([FindPackageShare("ariadne_bridge"), "/launch/live_link.launch.py"]),
        launch_arguments={"config_file": bridge, "socket_path": os.path.join(run, "link.sock"),
                          "ready_file": os.path.join(run, "link.ready")}.items())
    return [gz, link]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("scene_dir"),
        DeclareLaunchArgument("run_dir"),
        DeclareLaunchArgument("mission", default_value="live"),
        OpaqueFunction(function=_world),
    ])
