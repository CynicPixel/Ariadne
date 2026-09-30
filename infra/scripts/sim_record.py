#!/usr/bin/env python3
"""Record a generated mission and verify it (plan 1.3, arch §5.9, DECISIONS D39).

Runs INSIDE the ariadne-sim image (see infra/scripts/sim_record.sh), repo at /work:
    python3 infra/scripts/sim_record.py --scene DIR [--mission default] --out RECDIR
        [--seconds S]   (default: the whole mission + 1 s)
    python3 infra/scripts/sim_record.py --scene DIR --out RECDIR --seconds S --verify-only

A recording directory holds two rosbag2 MCAP bags and a manifest:
  RECDIR/canonical/   IMU + GT pose of every agent, written OFFLINE from the mission's
                      scripts by ariadne_canonical_streams (same code as the mover
                      plugin) on the exact stamp grid k*period, k = 0..floor(S/period).
                      Complete and byte-reproducible by construction: the SOURCE OF
                      TRUTH for these topics. Tagged "analytic IMU/GT from script".
  RECDIR/sensors/     everything gz rendered, via ros_gz_bridge + ros2 bag record:
                      stereo images, depth, camera_info, /clock, /tf_static, and the
                      plugin's LIVE IMU/GT pose (kept only as a cross-check).
  RECDIR/recording.json

Steps: build the plugin + tool; write canonical/ (twice, to prove reproducibility)
BEFORE the long gz run; record sensors/; VERIFY:
  canonical  exact stamp grid, log time == stamp, values == the independent Python
             evaluator (<= 1e-9)
  sensors    cameras/infos STRICT (count, period, monotone, stereo pairing 1.0);
             every live IMU/pose sample byte-identical to canonical at its stamp
             (any mismatch or off-grid stamp FAILS); live samples lost in transport
             are COUNTED and reported (live_gaps), not fatal, unless the cross-check
             covers < 99% of the stream.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, "/work/python")

PLUGIN_DIR = "/work/build/sim-plugins"
CANONICAL_TOOL = os.path.join(PLUGIN_DIR, "ariadne_canonical_streams")
STEP_NS = 1_000_000                 # the mission worlds' <physics max_step_size> (1 ms), asserted per world
LIVE_COVERAGE_MIN = 0.99            # below this the live cross-check is too thin to count


def log(m):
    print(f"[sim_record] {m}", flush=True)


def build_plugins():
    subprocess.run("cmake -S /work/simulator/plugins -B /work/build/sim-plugins >/dev/null && "
                   "cmake --build /work/build/sim-plugins >/dev/null", shell=True, check=True)


def write_canonical(world, end_ns, out):
    """Runs the offline writer; returns its one-line JSON summary."""
    r = subprocess.run([CANONICAL_TOOL, "--world", world, "--end-ns", str(end_ns), "--out", out],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"canonical stream writer failed: {r.stderr.strip()}")
    return json.loads(r.stdout)


def check_world_step(world, period_ns):
    """The recorder counts time in 1 ms gz iterations and the stamp grid assumes the
    step divides the IMU period: assert both against the world file itself."""
    import xml.etree.ElementTree as ET
    steps = [float(e.text) for e in ET.parse(world).getroot().iter("max_step_size")]
    if len(steps) != 1 or int(round(steps[0] * 1e9)) != STEP_NS:
        raise SystemExit(f"{world}: <max_step_size> {steps} != {STEP_NS} ns")
    if period_ns % STEP_NS:
        raise SystemExit(f"imu period {period_ns} ns is not a multiple of the {STEP_NS} ns step")


def content_digest(can):
    """Container-independent identity of the canonical streams: sha256 over every
    message in (stamp, topic) order of  topic NUL  stamp(int64 LE)  len(uint32 LE)
    payload. Invariant to rosbag2/MCAP/zstd versions; changes iff a stamp, a topic or
    a payload byte changes."""
    h = hashlib.sha256()
    for ns, topic in sorted((ns, t) for t, e in can.items() for ns in e["raw"]):
        b = can[topic]["raw"][ns]
        h.update(topic.encode() + b"\0" + struct.pack("<qI", ns, len(b)) + b)
    return h.hexdigest()


def dir_digests(d):
    return {f: {"bytes": os.path.getsize(os.path.join(d, f)), "sha256": sha256(os.path.join(d, f))}
            for f in sorted(os.listdir(d))}


def canonical_reproducible(world, end_ns, reference_dir):
    """Writes the canonical bag a second time (same directory NAME: rosbag2 embeds it)
    and compares every file byte-for-byte with the one the recording keeps."""
    with tempfile.TemporaryDirectory() as tmp:
        again = os.path.join(tmp, "canonical")
        write_canonical(world, end_ns, again)
        return dir_digests(again) == dir_digests(reference_dir)


def static_tf_node(mission):
    """A child process publishing /tf_static (transient-local) for every agent."""
    code = r'''
import json, sys, time, rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
m = json.load(open(sys.argv[1]))
rclpy.init()
node = rclpy.create_node("ariadne_static_tf")
qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
pub = node.create_publisher(TFMessage, "/tf_static", qos)
# R_B_optical = rows [[0,0,1],[-1,0,0],[0,-1,0]] -> quaternion (x,y,z,w) = (-0.5, 0.5, -0.5, 0.5)
msg = TFMessage()
for a in m["agents"]:
    body = a["frame_id"]
    for cam, t in m["t_B_camera_m"].items():
        ts = TransformStamped()
        ts.header.frame_id = body
        ts.child_frame_id = body.replace("base_link", cam + "_optical")
        ts.transform.translation.x, ts.transform.translation.y, ts.transform.translation.z = t
        ts.transform.rotation.x, ts.transform.rotation.y, ts.transform.rotation.z, ts.transform.rotation.w = -0.5, 0.5, -0.5, 0.5
        msg.transforms.append(ts)
pub.publish(msg)
while rclpy.ok():
    rclpy.spin_once(node, timeout_sec=0.5)
'''
    return subprocess.Popen([sys.executable, "-c", code, mission], start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--mission", default="default")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--max-cache-mb", type=int, default=100,
                    help="ros2 bag record --max-cache-size per buffer (double-buffered); rosbag2 drops what overflows")
    ap.add_argument("--verify-only", action="store_true",
                    help="re-verify an existing RECDIR (sensors/ + canonical/) without recording")
    args = ap.parse_args()
    scene_dir = os.path.abspath(args.scene)
    mfile = os.path.join(scene_dir, f"mission_{args.mission}.json")
    mission = json.load(open(mfile))
    sim_s = args.seconds if args.seconds else math.ceil(mission["duration_s"]) + 1.0
    iterations = int(round(sim_s * 1e9 / STEP_NS))
    end_ns = iterations * STEP_NS
    world = os.path.join(scene_dir, mission["world_file"])
    check_world_step(world, mission["imu"]["period_ns"])
    rec = os.path.abspath(args.out)
    sensors, canonical = os.path.join(rec, "sensors"), os.path.join(rec, "canonical")
    build_plugins()
    if args.verify_only:
        if not os.path.isdir(sensors):
            raise SystemExit(f"{sensors} missing")
        if not os.path.isdir(canonical):
            write_canonical(world, end_ns, canonical)
        result = verify(rec, mission, scene_dir, sim_s)
        result["canonical_reproducible"] = canonical_reproducible(world, end_ns, canonical)
        result["passed"] = result["passed"] and result["canonical_reproducible"]
        result["verify_only"] = True
        result["loss_reports"] = read_loss_reports(rec)
        write_manifest(rec, mission, scene_dir, result, args, end_ns)
        log("PASSED" if result["passed"] else "FAILED: " + "; ".join(result["failures"] or ["canonical"]))
        log_live_gaps(result)
        return 0 if result["passed"] else 1

    if os.path.exists(rec):
        raise SystemExit(f"{rec} exists; refusing to overwrite a recording")
    os.makedirs(rec)
    # Canonical streams first: cheap, and a script the plugin would reject fails here.
    summary = write_canonical(world, end_ns, canonical)
    log(f"canonical: {summary}")
    reproducible = canonical_reproducible(world, end_ns, canonical)
    log(f"canonical reproducible byte-for-byte: {reproducible}")
    if not reproducible:
        shutil.rmtree(rec)   # nothing was recorded; leave no directory that blocks a retry
        raise SystemExit("canonical streams are not reproducible; refusing to spend a render on them")
    import yaml
    bridge_cfg = os.path.join(scene_dir, mission["bridge_file"])
    topics = [e["ros_topic_name"] for e in yaml.safe_load(open(bridge_cfg))] + ["/tf_static"]

    env = dict(os.environ, GZ_SIM_SYSTEM_PLUGIN_PATH=PLUGIN_DIR)
    procs = []
    # Every helper's output is KEPT: rosbag2 reports its own losses there ("Cache
    # buffers lost messages per topic" on close; "Messages lost on transport layer").
    logs = os.path.join(rec, "logs")
    os.makedirs(logs)
    logf = {k: open(os.path.join(logs, f"{k}.log"), "w") for k in ("bridge", "recorder")}
    t_wall0 = time.monotonic()
    try:
        procs.append(subprocess.Popen(["ros2", "run", "ros_gz_bridge", "parameter_bridge", "--ros-args",
                                       "-p", f"config_file:={bridge_cfg}", "-p", "use_sim_time:=true"],
                                      start_new_session=True, stdout=logf["bridge"], stderr=subprocess.STDOUT))
        procs.append(static_tf_node(mfile))
        # Reliable, deep queues for the 200 Hz live streams (they are the cross-check
        # now, but a thin cross-check is a failed one: see LIVE_COVERAGE_MIN).
        qos_path = os.path.join("/tmp", "ariadne_record_qos.yaml")
        deep = {t: {"history": "keep_last", "depth": 1000, "reliability": "reliable", "durability": "volatile"}
                for t in topics if t.endswith("/imu") or t.endswith("/gt/pose")}
        with open(qos_path, "w") as f:
            yaml.safe_dump(deep, f)
        procs.append(subprocess.Popen(["ros2", "bag", "record", "-s", "mcap", "--storage-preset-profile", "zstd_fast",
                                       "--use-sim-time", "--qos-profile-overrides-path", qos_path,
                                       "--max-cache-size", str(args.max_cache_mb * 1024 * 1024), "-o", sensors,
                                       *topics],
                                      start_new_session=True, stdout=logf["recorder"], stderr=subprocess.STDOUT))
        time.sleep(6)   # discovery: bridge and recorder subscribed before the first step
        log(f"gz: {iterations} steps ({sim_s:.1f} sim-s), world {mission['world_file']}")
        t0 = time.monotonic()
        gz = subprocess.run(["gz", "sim", "-s", "-r", "--headless-rendering", "--iterations", str(iterations), world],
                            env=env, capture_output=True, text=True, timeout=6 * 3600)
        wall = time.monotonic() - t0
        log(f"gz exited {gz.returncode} after {wall:.1f} wall-s (RTF {sim_s / wall:.3f})")
        time.sleep(10)  # drain the recorder queue
    finally:
        for p in procs:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGINT)
        for p in procs:
            try:
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
        for f in logf.values():
            f.close()
    loss_reports = read_loss_reports(rec)
    gz_errors = [l for l in gz.stderr.splitlines() if "[Err]" in l or "rror" in l][:10]
    mover_lines = [l for l in gz.stdout.splitlines() + gz.stderr.splitlines() if "KinematicMover" in l][:10]
    result = verify(rec, mission, scene_dir, sim_s)
    result.update({"gz_returncode": gz.returncode, "gz_errors": gz_errors, "mover_log": mover_lines,
                   "sim_seconds": sim_s, "wall_seconds_gz": round(wall, 2), "canonical_summary": summary,
                   "canonical_reproducible": reproducible, "max_cache_mb": args.max_cache_mb,
                   "loss_reports": loss_reports,
                   "rtf": round(sim_s / wall, 4), "wall_seconds_total": round(time.monotonic() - t_wall0, 1)})
    result["passed"] = result["passed"] and gz.returncode == 0 and not gz_errors and reproducible
    write_manifest(rec, mission, scene_dir, result, args, end_ns)
    log("PASSED" if result["passed"] else "FAILED: " + "; ".join(result["failures"] or gz_errors or ["gz/canonical"]))
    log_live_gaps(result)
    for k, lines in loss_reports.items():
        for l in lines[:5]:
            log(f"{k} reports: {l}")
    return 0 if result["passed"] else 1


def read_loss_reports(rec):
    """Lines in which the bridge / recorder report lost or dropped messages."""
    out = {}
    for k in ("bridge", "recorder"):
        path = os.path.join(rec, "logs", f"{k}.log")
        if os.path.exists(path):
            out[k] = [l.strip() for l in open(path, errors="replace")
                      if "lost" in l.lower() or "dropp" in l.lower()][:50]
    return out


def log_live_gaps(result):
    for t, g in sorted(result["live_cross_check"].items()):
        if g["live_gaps"]:
            log(f"note: live {t} lost {g['live_gaps']} sample(s) in transport at {g['missing_stamps_ns'][:5]} ns "
                "(canonical is complete; reported, not fatal)")


def cdr_stamp_ns(data):
    """header.stamp (or Clock.clock) of a little-endian CDR message: every topic we
    record starts with a builtin_interfaces/Time right after the 4-byte encapsulation
    header, so the stamp is readable without deserializing a 2.7 MB image."""
    if bytes(data[0:2]) != b"\x00\x01":
        raise ValueError("not CDR little-endian")
    sec, nsec = struct.unpack_from("<iI", data, 4)
    return sec * 1_000_000_000 + nsec


def field_bits(m):
    """Every field of an Imu / PoseStamped as exact bits (stamp, frame_id, float64s)."""
    h = (m.header.stamp.sec, m.header.stamp.nanosec, m.header.frame_id)
    if hasattr(m, "angular_velocity"):
        o, w, a = m.orientation, m.angular_velocity, m.linear_acceleration
        vals = [o.x, o.y, o.z, o.w, *m.orientation_covariance, w.x, w.y, w.z, *m.angular_velocity_covariance,
                a.x, a.y, a.z, *m.linear_acceleration_covariance]
    else:
        p, o = m.pose.position, m.pose.orientation
        vals = [p.x, p.y, p.z, o.x, o.y, o.z, o.w]
    return h, struct.pack(f"<{len(vals)}d", *vals)


def read_bag(uri, keep_bytes=lambda topic: False):
    """topic -> {type, stamps[], log_ns[], bytes (sum), raw{stamp: payload} for kept topics}."""
    import rosbag2_py
    rd = rosbag2_py.SequentialReader()
    rd.open(rosbag2_py.StorageOptions(uri=uri, storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    out = {t.name: {"type": t.type, "stamps": [], "log_ns": [], "bytes": 0, "raw": {}, "dup": 0}
           for t in rd.get_all_topics_and_types()}
    while rd.has_next():
        topic, data, log_ns = rd.read_next()
        e = out[topic]
        ns = 0 if topic == "/tf_static" else cdr_stamp_ns(data)
        e["stamps"].append(ns)
        e["log_ns"].append(log_ns)
        e["bytes"] += len(data)
        if keep_bytes(topic):
            if ns in e["raw"]:
                e["dup"] += 1
            e["raw"][ns] = bytes(data)
    return out


def check_canonical(can, mission, scene_dir, grid, checks):
    """canonical/: every agent's IMU and GT topics on exactly `grid`, log time == stamp,
    frame_id right, and values equal to the INDEPENDENT Python evaluator (<= 1e-9).
    Fills checks[{"gt_pose","imu"}] = [n, worst]; returns failure strings."""
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    from dataset_tools.worldgen import routes
    from dataset_tools.worldgen.motion import Profile

    failures = []
    expected = {a["topics"][k] for a in mission["agents"] for k in ("imu", "gt_pose")}
    if set(can) != expected:
        failures.append(f"canonical topics {sorted(can)} != {sorted(expected)}")
    checks_pose, checks_imu = [0, 0.0], [0, 0.0]   # count, worst abs error vs Python
    for a in mission["agents"]:
        js = json.load(open(os.path.join(scene_dir, "routes", f"agent_{a['agent']}.json")))
        segs = [routes.Segment(d["kind"], d["t0"], Profile(d["D"], d["Ta"], d["vc"]), d["duration"],
                               tuple(d["start"]), tuple(d["end"]), d["tag"]) for d in js["segments"]]
        sc = routes.AgentScript(a["agent"], segs)
        for key in ("imu", "gt_pose"):
            topic = a["topics"][key]
            e = can.get(topic)
            if e is None:
                failures.append(f"canonical {topic}: missing")
                continue
            if e["stamps"] != grid:
                failures.append(f"canonical {topic}: stamps != the {len(grid)}-stamp grid {grid[:2]}..{grid[-1]} ns")
            if e["log_ns"] != e["stamps"]:
                failures.append(f"canonical {topic}: log time != header stamp")
            msg_t = get_message(e["type"])
            for ns in e["stamps"]:
                m = deserialize_message(e["raw"][ns], msg_t)
                if m.header.frame_id != a["frame_id"]:
                    failures.append(f"canonical {topic}: frame_id {m.header.frame_id!r}")
                    break
                t = ns * 1e-9
                if key == "gt_pose":
                    (x, y, z, h, p), _, _ = sc.pose(t)
                    qw, qx, qy, qz = (math.cos(h / 2) * math.cos(p / 2), -math.sin(h / 2) * math.sin(p / 2),
                                      math.cos(h / 2) * math.sin(p / 2), math.sin(h / 2) * math.cos(p / 2))
                    o = m.pose.orientation
                    err = max(abs(m.pose.position.x - x), abs(m.pose.position.y - y), abs(m.pose.position.z - z),
                              1 - abs(o.w * qw + o.x * qx + o.y * qy + o.z * qz))
                    checks_pose[0] += 1
                    checks_pose[1] = max(checks_pose[1], err)
                else:
                    gyro, f = routes.imu(sc, t)
                    w, la = m.angular_velocity, m.linear_acceleration
                    err = max(abs(w.x - gyro[0]), abs(w.y - gyro[1]), abs(w.z - gyro[2]),
                              abs(la.x - f[0]), abs(la.y - f[1]), abs(la.z - f[2]))
                    checks_imu[0] += 1
                    checks_imu[1] = max(checks_imu[1], err)
    if checks_pose[0] == 0 or checks_pose[1] > 1e-9:
        failures.append(f"canonical GT pose vs Python script: n={checks_pose[0]} worst {checks_pose[1]:.3g}")
    if checks_imu[0] == 0 or checks_imu[1] > 1e-9:
        failures.append(f"canonical IMU vs Python analytic: n={checks_imu[0]} worst {checks_imu[1]:.3g}")
    checks["gt_pose"], checks["imu"] = checks_pose, checks_imu
    return failures


def verify(rec, mission, scene_dir, sim_s):
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    period = mission["imu"]["period_ns"]
    end_ns = int(round(sim_s * 1e9 / STEP_NS)) * STEP_NS
    n_grid = end_ns // period                      # canonical k = 0..n_grid; the plugin publishes k >= 1
    fast = {a["topics"][k] for a in mission["agents"] for k in ("imu", "gt_pose")}
    can = read_bag(os.path.join(rec, "canonical"), keep_bytes=lambda t: True)
    live = read_bag(os.path.join(rec, "sensors"), keep_bytes=lambda t: t in fast)
    failures = []
    grid = [k * period for k in range(n_grid + 1)]

    checks = {}
    failures += check_canonical(can, mission, scene_dir, grid, checks)
    checks_pose, checks_imu = checks["gt_pose"], checks["imu"]

    # --- live IMU/pose: a bit-exact cross-check against canonical; losses counted.
    # Equal bytes settle it; otherwise the decoded fields are compared as IEEE bit
    # patterns (so -0.0 != 0.0), because CDR alignment padding carries no content
    # and rclcpp does not initialise it (D39).
    cross = {}
    live_grid = set(grid[1:])
    for topic in sorted(fast):
        e = live.get(topic)
        if e is None:
            failures.append(f"live {topic}: not in bag (no cross-check)")
            continue
        c = can.get(topic, {"raw": {}, "type": e["type"]})
        msg_t = get_message(e["type"])
        off_grid = sorted(set(e["raw"]) - live_grid)
        n_bytes_eq, mismatched = 0, []
        for ns, b in sorted(e["raw"].items()):
            if ns not in c["raw"]:
                continue
            if c["raw"][ns] == b:
                n_bytes_eq += 1
            elif field_bits(deserialize_message(b, msg_t)) != field_bits(deserialize_message(c["raw"][ns], msg_t)):
                mismatched.append(ns)
        compared = len(set(e["raw"]) & set(c["raw"]))
        missing = sorted(live_grid - set(e["raw"]))
        coverage = len(set(e["raw"]) & live_grid) / len(live_grid)
        cross[topic] = {"live": len(e["stamps"]), "expected_live": len(live_grid), "compared": compared,
                        "bytes_identical": n_bytes_eq, "fields_bit_identical": compared - len(mismatched),
                        "mismatched": len(mismatched), "off_grid": len(off_grid), "duplicates": e["dup"],
                        "live_gaps": len(missing), "missing_stamps_ns": missing[:50], "coverage": coverage,
                        "bytes": e["bytes"]}
        if mismatched:
            failures.append(f"live {topic}: {len(mismatched)} samples differ from canonical (first at {mismatched[0]} ns)")
        if off_grid:
            failures.append(f"live {topic}: {len(off_grid)} stamps off the canonical grid")
        if e["dup"]:
            failures.append(f"live {topic}: {e['dup']} duplicate stamps")
        if coverage < LIVE_COVERAGE_MIN:
            failures.append(f"live {topic}: cross-check coverage {coverage:.4f} < {LIVE_COVERAGE_MIN}")

    # --- rendered topics: STRICT (an image cannot be regenerated)
    per_topic = {}
    cam_rate = mission["rig"]["camera"]["rate_hz"]
    exp_cam = int(math.floor(sim_s * cam_rate + 1e-9)) + 1
    cam_period = int(round(1e9 / cam_rate))
    for a in mission["agents"]:
        T = a["topics"]
        for key in ("left_image", "right_image", "depth_image", "left_info", "right_info", "depth_info"):
            topic = T[key]
            if topic not in live:
                if key.startswith("depth") and not mission["rig"]["depth"]["enabled"]:
                    continue
                failures.append(f"{topic}: not in bag")
                continue
            s = live[topic]["stamps"]
            gaps = [b - a_ for a_, b in zip(s, s[1:])]
            per_topic[topic] = {"count": len(s), "expected": exp_cam, "max_gap_ms": (max(gaps) / 1e6) if gaps else None,
                                "monotone": all(g > 0 for g in gaps), "bytes": live[topic]["bytes"]}
            if len(s) != exp_cam:
                failures.append(f"{topic}: {len(s)} messages != expected {exp_cam}")
            if gaps and max(gaps) != cam_period:
                failures.append(f"{topic}: max gap {max(gaps) / 1e6:.3f} ms != period {cam_period / 1e6:.3f} ms")
            if not per_topic[topic]["monotone"]:
                failures.append(f"{topic}: stamps not strictly increasing")
        L, R = set(live.get(T["left_image"], {"stamps": []})["stamps"]), set(live.get(T["right_image"], {"stamps": []})["stamps"])
        pairing = len(L & R) / max(len(L), 1) if L else 0.0
        per_topic.setdefault(T["left_image"], {})["stereo_pairing"] = pairing
        if pairing < 1.0:
            failures.append(f"agent {a['agent']}: stereo pairing {pairing:.4f} < 1")
    if not live.get("/tf_static", {}).get("stamps"):
        failures.append("/tf_static not recorded")
    clock = live.get("/clock", {}).get("stamps", [])
    if not clock:
        failures.append("/clock not recorded")
    # /clock is not consumed by replay (header stamps are); its transport losses are
    # reported (they explain rosbag2's "lost on the transport layer" count), not fatal.
    clock_gaps = sum((b - a_) // STEP_NS - 1 for a_, b in zip(clock, clock[1:]) if b - a_ > STEP_NS)
    return {"passed": not failures, "failures": failures, "topics": per_topic,
            "canonical": {t: {"count": len(can[t]["stamps"]), "bytes": can[t]["bytes"]} for t in sorted(can)},
            "canonical_content_sha256": content_digest(can),
            "canonical_vs_python": {"gt_pose": {"n": checks_pose[0], "worst_abs": checks_pose[1]},
                                    "imu": {"n": checks_imu[0], "worst_abs": checks_imu[1]}},
            "live_cross_check": cross,
            "clock": {"count": len(clock), "missing_steps": int(clock_gaps)}}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(rec, mission, scene_dir, result, args, end_ns):
    versions = open("/opt/ariadne/sim-versions.txt").read().splitlines() \
        if os.path.exists("/opt/ariadne/sim-versions.txt") else []
    scene = json.load(open(os.path.join(scene_dir, "scene.json")))
    m = {"tool": "infra/scripts/sim_record.py", "command": " ".join(sys.argv),
         "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "git_commit": os.environ.get("ARIADNE_GIT_COMMIT", "unknown"),
         "git_dirty": os.environ.get("ARIADNE_GIT_DIRTY", "unknown"),
         "image_id": os.environ.get("ARIADNE_IMAGE_DIGEST", "unknown"),
         "render_mode": os.environ.get("ARIADNE_RENDER_MODE", "unknown"),
         "scene_id": scene["scene_id"], "generator_version": scene["generator_version"],
         "preset": scene["preset"]["name"], "preset_digest": scene["preset"]["digest"], "seed": scene["seed"],
         "mission": mission["mission"], "rig": mission["rig"]["name"], "end_ns": end_ns,
         "streams": {
             "canonical": {"dir": "canonical", "source": "analytic IMU/GT from script [DESIGN] (D39): "
                           "ariadne_canonical_streams evaluating the mission SDF's KinematicMover scripts; "
                           "not sensor output. Source of truth for the IMU and GT-pose topics.",
                           # a pure function of these inputs (tool built from git_commit in image_id):
                           "inputs": {"mission_sdf": mission["world_file"],
                                      "mission_sdf_sha256": sha256(os.path.join(scene_dir, mission["world_file"])),
                                      "end_ns": end_ns, "tool_sha256": sha256(CANONICAL_TOOL)},
                           "content_sha256": result.get("canonical_content_sha256"),
                           "reproducibility": "container bytes: re-generated and compared in the same image on "
                                              "every recording; across images/arches the content digest may move "
                                              "only by libm/codegen, guarded by the <= 1e-9 Python check",
                           "files": dir_digests(os.path.join(rec, "canonical"))},
             "sensors": {"dir": "sensors", "source": "gz render -> ros_gz_bridge -> ros2 bag record; its IMU/GT "
                         "topics are the live cross-check only", "files": dir_digests(os.path.join(rec, "sensors"))}},
         "replay_rule": "IMU + GT pose from canonical/, everything else from sensors/; merge by header stamp "
                        "(ties: topic name), not by log time (sensor log times are receive times)",
         "sim_versions": versions, "result": result}
    with open(os.path.join(rec, "recording.json"), "w") as f:
        json.dump(m, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    sys.exit(main())
