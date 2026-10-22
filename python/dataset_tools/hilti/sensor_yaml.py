"""Hilti Kalibr calibration -> EuRoC sensor.yaml files for an exported ASL tree (5.6, D141/D143).

  python -m dataset_tools.hilti.sensor_yaml ASL_ROOT CALIB_DIR [--rate-hz 10]

ASL_ROOT is to_asl's output (mav0/cam0, mav0/cam1, mav0/imu_imu_data); CALIB_DIR is calibration_robot/ from
calibration.zip. Writes mav0/cam{0,1}/sensor.yaml and moves imu_imu_data -> imu0 with its sensor.yaml, then checks:
  - the stereo extrinsic composed from the two camera-IMU transforms matches Kalibr's own T_cn_cnm1 for the pair
    (a frame-direction error shows up here first);
  - left and right image stamps pair up. The pair is hardware-triggered (same start of exposure) but each image is
    stamped at its own centre of exposure, so when the two exposure times differ by ~20 us the stamps differ by
    ~10 us (13 of 3592 frames in site2_robot_3). Such a frame is re-stamped on the right with the left's stamp, only
    when exposure.csv shows the same start_of_frame and |dt| < 100 us; any other mismatch is refused.

Kalibr's T_here_imu0 (`T_cam_imu`) maps IMU coordinates into the camera: p_C = T_C_I p_I. EuRoC's T_BS is body <-
sensor with body = IMU, so T_BS = T_C_I^-1. The cameras are matched by `rostopic` against hilti_pair.txt.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml


def _T(rows):
    T = np.array(rows, dtype=float)
    if T.shape != (4, 4) or abs(np.linalg.det(T[:3, :3]) - 1.0) > 1e-6:
        raise SystemExit("not a 4x4 rigid transform")
    return T


def _inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def _flat(T):
    return ", ".join(repr(float(v)) for v in T.ravel())


def camera_yaml(cam, T_BS, rate_hz, origin):
    if cam["camera_model"] != "pinhole" or cam["distortion_model"] not in ("equidistant", "radtan"):
        raise SystemExit(f"{origin}: unsupported {cam['camera_model']}/{cam['distortion_model']}")
    dm = "equidistant" if cam["distortion_model"] == "equidistant" else "radial-tangential"
    k, d, res = cam["intrinsics"], cam["distortion_coeffs"], cam["resolution"]
    return ("%YAML:1.0\n"
            "sensor_type: camera\n"
            f"comment: Hilti 2023 {cam['rostopic']} (Kalibr calibration_robot, converted by dataset_tools.hilti)\n"
            f"T_BS:\n  cols: 4\n  rows: 4\n  data: [{_flat(T_BS)}]\n"
            f"rate_hz: {rate_hz}\n"
            f"resolution: [{res[0]}, {res[1]}]\n"
            "camera_model: pinhole\n"
            f"intrinsics: [{', '.join(repr(float(v)) for v in k)}]\n"
            f"distortion_model: {dm}\n"
            f"distortion_coefficients: [{', '.join(repr(float(v)) for v in d)}]\n")


def imu_yaml(imu):
    return ("sensor_type: imu\n"
            f"comment: Hilti 2023 {imu['rostopic']} (Kalibr calibration_imu; body = this IMU)\n"
            f"T_BS:\n  cols: 4\n  rows: 4\n  data: [{_flat(np.eye(4))}]\n"
            f"rate_hz: {float(imu['update_rate'])}\n"
            f"gyroscope_noise_density: {imu['gyroscope_noise_density']!r}\n"
            f"gyroscope_random_walk: {imu['gyroscope_random_walk']!r}\n"
            f"accelerometer_noise_density: {imu['accelerometer_noise_density']!r}\n"
            f"accelerometer_random_walk: {imu['accelerometer_random_walk']!r}\n")


def stamps(csv_path):
    return [l.split(",")[0] for l in Path(csv_path).read_text().splitlines() if l and not l.startswith("#")]


def paired_right_csv(mav):
    """cam1/data.csv text with trigger-paired stamps, and how many rows were re-stamped (see module doc)."""
    s0, s1 = stamps(mav / "cam0" / "data.csv"), stamps(mav / "cam1" / "data.csv")
    if len(s0) != len(s1):
        raise SystemExit(f"left/right frame counts differ: {len(s0)} vs {len(s1)}")
    bad = [i for i, (a, b) in enumerate(zip(s0, s1)) if a != b]
    if not bad:
        return None, 0
    e0, e1 = (mav / "cam0" / "exposure.csv"), (mav / "cam1" / "exposure.csv")
    if not (e0.exists() and e1.exists()):
        raise SystemExit(f"left/right stamps differ on {len(bad)} rows and there is no exposure.csv to pair them by")
    sof0 = {r.split(",")[0]: r.split(",")[1] for r in e0.read_text().splitlines() if r and not r.startswith("#")}
    sof1 = {r.split(",")[0]: r.split(",")[1] for r in e1.read_text().splitlines() if r and not r.startswith("#")}
    for i in bad:
        a, b = s0[i], s1[i]
        if sof0.get(a) is None or sof0.get(a) != sof1.get(b) or abs(int(a) - int(b)) >= 100_000:
            raise SystemExit(f"left/right stamps differ at row {i} ({a} vs {b}) and are not one trigger")
    rows = [l for l in (mav / "cam1" / "data.csv").read_text().splitlines() if l and not l.startswith("#")]
    out = ["#timestamp [ns],filename"] + [f"{s0[i]},{r.split(',', 1)[1]}" for i, r in enumerate(rows)]
    return "\n".join(out) + "\n", len(bad)


def convert(root, calib, rate_hz):
    mav = Path(root) / "mav0"
    pair = (Path(root) / "hilti_pair.txt").read_text().split()   # "left T right T"
    left, right = pair[1], pair[3]
    cams = yaml.safe_load((Path(calib) / "calibration_cams.yaml").read_text())
    imus = yaml.safe_load((Path(calib) / "calibration_imu.yaml").read_text())
    by_topic = {c["rostopic"]: (n, c) for n, c in cams.items()}
    (n0, c0), (n1, c1) = by_topic[left], by_topic[right]
    T_C0_I, T_C1_I = _T(c0["T_here_imu0"]), _T(c1["T_here_imu0"])
    report = {"cam0": n0, "cam1": n1}
    right_csv, restamped = paired_right_csv(mav)   # every check before anything is moved or rewritten
    if "T_cn_cnm1" in c1:   # Kalibr's chained extrinsic: cam(n) <- cam(n-1), present when cam1 follows cam0
        T_kalibr = _T(c1["T_cn_cnm1"])
        T_ours = T_C1_I @ _inv(T_C0_I)
        report["stereo_extrinsic_max_abs_diff"] = float(np.abs(T_ours - T_kalibr).max())
        report["baseline_m"] = float(np.linalg.norm(T_ours[:3, 3]))
        if report["stereo_extrinsic_max_abs_diff"] > 1e-3:
            raise SystemExit(f"stereo extrinsic disagrees with Kalibr's T_cn_cnm1: {report}")
    (mav / "cam0" / "sensor.yaml").write_text(camera_yaml(c0, _inv(T_C0_I), rate_hz, n0))
    (mav / "cam1" / "sensor.yaml").write_text(camera_yaml(c1, _inv(T_C1_I), rate_hz, n1))
    imu_dirs = [d for d in mav.iterdir() if d.name.startswith("imu_")]
    imu_name, imu = next(iter(imus.items()))
    src = [d for d in imu_dirs if d.name == "imu_" + imu["rostopic"].strip("/").replace("/", "_")]
    if len(src) != 1:
        raise SystemExit(f"no exported IMU folder for {imu['rostopic']} among {[d.name for d in imu_dirs]}")
    if (mav / "imu0").exists():
        raise SystemExit("mav0/imu0 already exists")
    shutil.move(str(src[0]), str(mav / "imu0"))
    (mav / "imu0" / "sensor.yaml").write_text(imu_yaml(imu))
    if right_csv is not None:
        (mav / "cam1" / "data.csv").write_text(right_csv)
    report["frames"] = len(stamps(mav / "cam0" / "data.csv"))
    report["right_restamped_to_left"] = restamped
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dataset_tools.hilti.sensor_yaml")
    ap.add_argument("root"); ap.add_argument("calib"); ap.add_argument("--rate-hz", type=float, default=10.0)
    a = ap.parse_args(argv)
    for k, v in convert(a.root, a.calib, a.rate_hz).items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    sys.exit(main())
