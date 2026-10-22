# SMOKE §4w (D165): gz paused + a bridge with /clock, one camera and the world-control service; probe.py asks
# run_to_sim_time 2 s then 4 s. Run from the repo root:
#   docker run --rm --memory=4g -u "$(id -u):$(id -g)" -v "$PWD":/work:ro ariadne/sim:dev bash /work/infra/scripts/tests/lockstep_smoke/run.sh
# Needs build/sim-plugins and build/live/scenes/ci_tiny_0 (infra/scripts/live.sh build; live.sh compose ci_tiny 0 0.15).
set +u
source /opt/ros/jazzy/setup.bash
export GZ_SIM_SYSTEM_PLUGIN_PATH=/work/build/sim-plugins HOME=/tmp GZ_TRANSPORT_RCVHWM=0 GZ_TRANSPORT_SNDHWM=0
SC=/work/build/live/scenes/ci_tiny_0; W=scene_aadffb8df5790ca0
cat > /tmp/b.yaml <<Y
- {ros_topic_name: /clock, gz_topic_name: /clock, ros_type_name: rosgraph_msgs/msg/Clock, gz_type_name: gz.msgs.Clock, direction: GZ_TO_ROS}
- {ros_topic_name: /ariadne/agent_0/left/image_raw, gz_topic_name: /ariadne/agent_0/left/image_raw, ros_type_name: sensor_msgs/msg/Image, gz_type_name: gz.msgs.Image, direction: GZ_TO_ROS}
- {service_name: /world/$W/control, ros_type_name: ros_gz_interfaces/srv/ControlWorld, gz_req_type_name: gz.msgs.WorldControl, gz_rep_type_name: gz.msgs.Boolean}
Y
mkdir -p /tmp/run && cd /tmp/run
gz sim -s --headless-rendering $SC/mission_live.sdf > /tmp/gz.log 2>&1 &
ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=/tmp/b.yaml > /tmp/bridge.log 2>&1 &
sleep 10
python3 /work/infra/scripts/tests/lockstep_smoke/probe.py $W
echo "--- bridge log"; grep -i "service\|error\|warn" /tmp/bridge.log | head -5
