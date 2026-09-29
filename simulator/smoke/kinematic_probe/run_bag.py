import os, signal, subprocess, time, json, shutil, glob
SIM_S = 5.0
topics = {}
for k in (0, 1):
    for n in ("left", "right", "depth"):
        topics[f"/a{k}/{n}"] = ("sensor_msgs/msg/Image", "gz.msgs.Image")
subprocess.run("mkdir -p /tmp/b && cd /tmp/b && cmake -S /work -B . >/dev/null && make -s", shell=True, check=True)
subprocess.run("mkdir -p /tmp/w && cp /work/bagsize.sdf /work/tex_*.png /tmp/w/", shell=True, check=True)
res = {}
for profile in ("none", "zstd_fast"):
    bag = f"/tmp/bag_{profile}"; shutil.rmtree(bag, ignore_errors=True)
    rec = ["ros2", "bag", "record", "-s", "mcap", "-o", bag, *topics]
    if profile != "none": rec[5:5] = ["--storage-preset-profile", profile]
    procs = [subprocess.Popen(["ros2", "run", "ros_gz_bridge", "parameter_bridge", *[f"{t}@{r}[{g}" for t, (r, g) in topics.items()]],
                              start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
             subprocess.Popen(rec, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
    time.sleep(5)
    env = dict(os.environ, GZ_SIM_SYSTEM_PLUGIN_PATH="/tmp/b", GZ_SIM_RESOURCE_PATH="/tmp/w")
    t0 = time.monotonic()
    r = subprocess.run(["gz", "sim", "-s", "-r", "--headless-rendering", "--iterations", str(int(SIM_S * 1000)), "/tmp/w/bagsize.sdf"],
                       env=env, capture_output=True, text=True, timeout=3000)
    wall = time.monotonic() - t0
    time.sleep(8)
    for p in procs: os.killpg(p.pid, signal.SIGINT)
    for p in procs: p.wait(timeout=60)
    size = sum(os.path.getsize(f) for f in glob.glob(f"{bag}/*.mcap"))
    import rosbag2_py
    rd = rosbag2_py.SequentialReader()
    rd.open(rosbag2_py.StorageOptions(uri=bag, storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    counts, raw = {}, {}
    while rd.has_next():
        t, d, _ = rd.read_next(); counts[t] = counts.get(t, 0) + 1; raw[t] = raw.get(t, 0) + len(d)
    errs = [l for l in r.stderr.splitlines() if "rror" in l or "arn" in l][:4]
    res[profile] = {"rtf": round(SIM_S / wall, 3), "file_MB": round(size / 1e6, 1), "MB_per_sim_s": round(size / 1e6 / SIM_S, 1),
                    "raw_MB_per_topic": {t: round(v / 1e6, 1) for t, v in raw.items()}, "counts": counts, "gz_msgs": errs}
    # keep one frame for visual check
    if profile == "none":
        rd = rosbag2_py.SequentialReader()
        rd.open(rosbag2_py.StorageOptions(uri=bag, storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import Image
        import numpy as np
        while rd.has_next():
            t, d, _ = rd.read_next()
            if t == "/a0/left":
                m = deserialize_message(d, Image)
                im = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, 3)
                res["a0_left_mean_std"] = [round(float(im.mean()), 1), round(float(im.std()), 1)]
                np.save("/out/a0_left.npy", im)
                break
print(json.dumps(res, indent=1))
