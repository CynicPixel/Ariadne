#!/usr/bin/env python3
import os, signal, subprocess, time, json, shutil
import numpy as np
HERE = "/work"
TOPICS = {"/probe/cam": ("sensor_msgs/msg/Image", "gz.msgs.Image"),
          "/probe/depth": ("sensor_msgs/msg/Image", "gz.msgs.Image"),
          "/probe/imu": ("sensor_msgs/msg/Imu", "gz.msgs.IMU")}
PHYS = '<plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>'

def build():
    subprocess.run("mkdir -p /tmp/b && cd /tmp/b && cmake -S /work -B . -DCMAKE_BUILD_TYPE=Release >/dev/null && make -s", shell=True, check=True)

def run(tag, static, physics):
    world = f"/tmp/{tag}.sdf"
    s = open(f"{HERE}/probe.sdf.in").read().replace("@STATIC@", static).replace("@PHYSICS@", PHYS if physics else "")
    open(world, "w").write(s)
    bag = f"/tmp/bag_{tag}"; shutil.rmtree(bag, ignore_errors=True)
    args = [f"{t}@{r}[{g}" for t, (r, g) in TOPICS.items()]
    procs = [subprocess.Popen(["ros2", "run", "ros_gz_bridge", "parameter_bridge", *args], start_new_session=True,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
             subprocess.Popen(["ros2", "bag", "record", "-s", "mcap", "-o", bag, *TOPICS], start_new_session=True,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
    time.sleep(4)
    env = dict(os.environ, GZ_SIM_SYSTEM_PLUGIN_PATH="/tmp/b")
    r = subprocess.run(["gz", "sim", "-s", "-r", "--headless-rendering", "--iterations", "2000", world],
                       env=env, capture_output=True, text=True, timeout=600)
    time.sleep(3)
    for p in procs: os.killpg(p.pid, signal.SIGINT)
    for p in procs: p.wait(timeout=20)
    errs = [l for l in (r.stderr or "").splitlines() if "rror" in l][:3]
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    rd = rosbag2_py.SequentialReader()
    rd.open(rosbag2_py.StorageOptions(uri=bag, storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in rd.get_all_topics_and_types()}
    depth, cam, imu = [], [], []
    while rd.has_next():
        topic, data, _ = rd.read_next()
        m = deserialize_message(data, get_message(types[topic]))
        st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if topic == "/probe/depth":
            d = np.frombuffer(bytes(m.data), dtype=np.float32).reshape(m.height, m.width)
            depth.append((st, float(d[m.height // 2, m.width // 2])))
        elif topic == "/probe/cam":
            im = np.frombuffer(bytes(m.data), dtype=np.uint8).reshape(m.height, m.width).astype(float)
            w = im.sum()
            col = (im.sum(0) * np.arange(m.width)).sum() / w if w > 0 else float("nan")
            cam.append((st, im.mean(), col))
        else:
            imu.append((st, m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z,
                        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z))
    out = {"tag": tag, "gz_rc": r.returncode, "errs": errs, "n_depth": len(depth), "n_cam": len(cam), "n_imu": len(imu)}
    # Phase A: implied pose time from depth. wall face at x=8.0, camera at link origin.
    offs = [((8.0 - d) / 5.0) - st for st, d in depth if 0.05 < st < 0.95 and np.isfinite(d)]
    out["phaseA_offset_ms"] = [round(o * 1e3, 2) for o in offs[:6]]
    out["phaseA_offset_ms_median"] = round(float(np.median(offs)) * 1e3, 3) if offs else None
    out["phaseB_depth"] = [round(d, 4) for st, d in depth if 1.05 < st < 1.95][:4]
    out["cam_mean_first_last"] = [round(cam[0][1], 2), round(cam[-1][1], 2)] if cam else None
    out["phaseB_centroid_col"] = [round(c, 1) for st, m_, c in cam if 1.05 < st < 1.95][:6]
    out["phaseA_centroid_col"] = [round(c, 1) for st, m_, c in cam if 0.05 < st < 0.95][:6]
    if imu:
        a = np.array(imu)
        out["imu_mean_acc"] = np.round(a[:, 1:4].mean(0), 3).tolist()
        out["imu_std_acc"] = np.round(a[:, 1:4].std(0), 3).tolist()
        out["imu_mean_gyro"] = np.round(a[:, 4:7].mean(0), 4).tolist()
    return out

build()
for tag, static, phys in [("static_nophys", "true", False), ("dyn_nophys", "false", False), ("dyn_phys", "false", True)]:
    print(json.dumps(run(tag, static, phys)), flush=True)
