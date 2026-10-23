"""Hilti ROS1 bag -> EuRoC export (5.6, D141): images + IMU + the exposure and control-point side files, on a
synthetic bag that carries Hilti's two custom message types the way the real one does (definitions embedded)."""
import csv

import numpy as np
import pytest

rosbags = pytest.importorskip("rosbags")
cv2 = pytest.importorskip("cv2")
from rosbags.rosbag1 import Writer  # noqa: E402
from rosbags.typesys import Stores, get_types_from_msg, get_typestore  # noqa: E402

from dataset_tools.hilti import to_asl  # noqa: E402

EXPOSURE_MSG = """std_msgs/Header header
time start_of_frame
uint32 exposure_time
uint32 iso
"""
GCP_MSG = """std_msgs/Header header
string point_name
geometry_msgs/Point point
"""


def _store():
    ts = get_typestore(Stores.ROS1_NOETIC)
    ts.register(get_types_from_msg(EXPOSURE_MSG, "hilti_slam_challenge_msgs/msg/ExposureInfo"))
    ts.register(get_types_from_msg(GCP_MSG, "hilti_slam_challenge_msgs/msg/GroundControlPoint"))
    return ts


def _write_bag(path):
    ts = _store()
    T, H, Img = ts.types["builtin_interfaces/msg/Time"], ts.types["std_msgs/msg/Header"], ts.types["sensor_msgs/msg/Image"]
    Exp = ts.types["hilti_slam_challenge_msgs/msg/ExposureInfo"]
    Gcp = ts.types["hilti_slam_challenge_msgs/msg/GroundControlPoint"]
    Pt = ts.types["geometry_msgs/msg/Point"]
    Imu, Q, V = ts.types["sensor_msgs/msg/Imu"], ts.types["geometry_msgs/msg/Quaternion"], \
        ts.types["geometry_msgs/msg/Vector3"]
    hdr = lambda ns, fid="": H(seq=0, stamp=T(sec=ns // 10**9, nanosec=ns % 10**9), frame_id=fid)  # noqa: E731
    with Writer(path) as w:
        c = {t: w.add_connection(t, ty, typestore=ts) for t, ty in [
            ("/oak_cam_front/left/image_raw", Img.__msgtype__), ("/oak_cam_front/right/image_raw", Img.__msgtype__),
            ("/oak_cam_front/left/exposure_info", Exp.__msgtype__),
            ("/oak_cam_front/right/exposure_info", Exp.__msgtype__),
            ("/ground_control_point", Gcp.__msgtype__), ("/imu/data", Imu.__msgtype__)]}
        for i in range(3):
            ns = 10**18 + i * 100_000_000
            px = np.full((4, 6), 10 * i, np.uint8)
            for side in ("left", "right"):
                m = Img(header=hdr(ns), height=4, width=6, encoding="mono8", is_bigendian=0, step=6, data=px.ravel())
                w.write(c[f"/oak_cam_front/{side}/image_raw"], ns, ts.serialize_ros1(m, Img.__msgtype__))
                sof = ns - 500_000
                e = Exp(header=hdr(ns), start_of_frame=T(sec=sof // 10**9, nanosec=sof % 10**9),
                        exposure_time=1000 + i, iso=200 + i)
                w.write(c[f"/oak_cam_front/{side}/exposure_info"], ns, ts.serialize_ros1(e, Exp.__msgtype__))
            imu = Imu(header=hdr(ns), orientation=Q(x=0, y=0, z=0, w=1), orientation_covariance=np.zeros(9),
                      angular_velocity=V(x=0.1, y=0.2, z=0.3), angular_velocity_covariance=np.zeros(9),
                      linear_acceleration=V(x=0.0, y=0.0, z=9.81), linear_acceleration_covariance=np.zeros(9))
            w.write(c["/imu/data"], ns, ts.serialize_ros1(imu, Imu.__msgtype__))
        g = Gcp(header=hdr(10**18 + 150_000_000, "rslidar"), point_name="P7", point=Pt(x=1.5, y=-0.25, z=-0.75))
        w.write(c["/ground_control_point"], 10**18 + 150_000_000, ts.serialize_ros1(g, Gcp.__msgtype__))


def _rows(p):
    with open(p) as f:
        return [r for r in csv.reader(f) if r and not r[0].startswith("#")]


@pytest.mark.parametrize("no_images", [False, True])
def test_export_writes_images_imu_exposure_and_gcp(tmp_path, no_images):
    bag = tmp_path / "s.bag"
    _write_bag(bag)
    argv = ["export", str(bag), str(tmp_path / "out"), "--left", "/oak_cam_front/left/image_raw",
            "--right", "/oak_cam_front/right/image_raw"] + (["--no-images"] if no_images else [])
    to_asl.main(argv)
    mav = tmp_path / "out" / "mav0"
    for cam in ("cam0", "cam1"):
        exp = _rows(mav / cam / "exposure.csv")
        assert [r[2:] for r in exp] == [["1000", "200"], ["1001", "201"], ["1002", "202"]]
        assert int(exp[0][0]) - int(exp[0][1]) == 500_000          # start of exposure precedes the centre stamp
        if no_images:
            assert not (mav / cam / "data.csv").exists()
        else:
            imgs = _rows(mav / cam / "data.csv")
            assert [r[0] for r in imgs] == [r[0] for r in exp]
            assert cv2.imread(str(mav / cam / "data" / imgs[2][1]), cv2.IMREAD_UNCHANGED)[0, 0] == 20
    assert _rows(tmp_path / "out" / "gcp.csv") == [[str(10**18 + 150_000_000), "rslidar", "P7", "1.5", "-0.25", "-0.75"]]
    assert len(_rows(mav / "imu_imu_data" / "data.csv")) == 3
