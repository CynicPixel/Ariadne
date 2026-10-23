"""Kalibr -> EuRoC sensor.yaml (5.6, D143), with the Hilti robot front pair's published numbers."""
import numpy as np
import pytest

yaml = pytest.importorskip("yaml")
from dataset_tools.hilti import sensor_yaml as sy  # noqa: E402

CAMS = """cam0:
  T_here_imu0:
  - [0.005591575656874202, -0.9999626443753757, 0.00659121654007, 0.028950893174162138]
  - [-0.9059045237433778, -0.002274297411819927, 0.4234758805794793, 0.09594713829195171]
  - [-0.4234450709867294, -0.00833891030574313, -0.9058834000200884, -0.03848313572199586]
  - [0.0, 0.0, 0.0, 1.0]
  camera_model: pinhole
  distortion_coeffs: [-0.0032589824594531436, 0.004329813673000139, 0.0035601008360336764, -0.002021576943252078]
  distortion_model: equidistant
  intrinsics: [567.7610239182566, 567.1293085165794, 651.55636483273, 402.9081460847209]
  resolution: [1280, 800]
  rostopic: /oak_cam_front/left/image_raw
cam1:
  T_cn_cnm1:
  - [0.9999932800052187, 0.0021415002527286177, 0.002975553909348469, -0.07514051004566691]
  - [-0.002144068885679238, 0.9999973314022884, 0.0008603237244430193, -0.0002760964662000532]
  - [-0.0029737035853187994, -0.0008666977356267781, 0.9999952029495028, 0.0002737345347534046]
  - [0.0, 0.0, 0.0, 1.0]
  T_here_imu0:
  - [0.0023915596786017214, -0.999985607917269, 0.0048025410600709695, -0.04609884924539652]
  - [-0.9062783948125973, -0.00013747671236041467, 0.42268126548979595, 0.09557560511751825]
  - [-0.4226745219884982, -0.005363306674388446, -0.9062656803627239, -0.03837846512402488]
  - [0.0, 0.0, 0.0, 1.0]
  camera_model: pinhole
  distortion_coeffs: [-0.008341888403781413, 0.04033084558312753, -0.04129189258321806, 0.01521997678462175]
  distortion_model: equidistant
  intrinsics: [565.7720712258174, 565.3132881060791, 652.281607825715, 395.68847555469665]
  resolution: [1280, 800]
  rostopic: /oak_cam_front/right/image_raw
"""
IMU = """imu0:
  T_here_imu0: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
  accelerometer_noise_density: 0.0006771784583884886
  accelerometer_random_walk: 4.855592390728982e-06
  gyroscope_noise_density: 0.00013548849713483453
  gyroscope_random_walk: 2.6867875817855036e-06
  model: calibrated
  rostopic: /imu/data
  update_rate: 200.0
"""


def _tree(tmp_path, right_stamps=("1", "2"), sof=None):
    calib = tmp_path / "calib"; calib.mkdir()
    (calib / "calibration_cams.yaml").write_text(CAMS)
    (calib / "calibration_imu.yaml").write_text(IMU)
    root = tmp_path / "asl"
    for cam, st in (("cam0", ("1", "2")), ("cam1", right_stamps)):
        (root / "mav0" / cam).mkdir(parents=True)
        (root / "mav0" / cam / "data.csv").write_text("#timestamp [ns],filename\n" + "".join(f"{s},{s}.png\n" for s in st))
    if sof:   # exposure.csv per camera: (stamp, start_of_frame) rows
        for cam, rows in sof.items():
            (root / "mav0" / cam / "exposure.csv").write_text(
                "#timestamp [ns],start_of_frame [ns],exposure_time [us],iso\n" + "".join(f"{t},{f},30000,900\n" for t, f in rows))
    (root / "mav0" / "imu_imu_data").mkdir()
    (root / "mav0" / "imu_imu_data" / "data.csv").write_text("#t\n")
    (root / "hilti_pair.txt").write_text("left /oak_cam_front/left/image_raw right /oak_cam_front/right/image_raw\n")
    return root, calib


def test_converts_and_self_checks_the_front_pair(tmp_path):
    root, calib = _tree(tmp_path)
    rep = sy.convert(root, calib, 10.0)
    assert rep["stereo_extrinsic_max_abs_diff"] < 1e-9
    assert rep["baseline_m"] == pytest.approx(0.07514, abs=1e-4)
    assert rep["right_restamped_to_left"] == 0 and rep["frames"] == 2
    c0 = yaml.safe_load((root / "mav0/cam0/sensor.yaml").read_text().split("\n", 1)[1])
    assert c0["distortion_model"] == "equidistant" and c0["rate_hz"] == 10.0
    T_BS = np.array(c0["T_BS"]["data"]).reshape(4, 4)
    T_C_I = np.array(yaml.safe_load(CAMS)["cam0"]["T_here_imu0"])
    assert np.abs(T_BS @ T_C_I - np.eye(4)).max() < 1e-12            # T_BS = T_C_I^-1 (body = IMU)
    imu = yaml.safe_load((root / "mav0/imu0/sensor.yaml").read_text())
    assert imu["rate_hz"] == 200.0 and not (root / "mav0/imu_imu_data").exists()


def test_refuses_unpaired_stamps_and_leaves_the_tree_untouched(tmp_path):
    root, calib = _tree(tmp_path, right_stamps=("1", "3"))
    with pytest.raises(SystemExit, match="stamps differ"):
        sy.convert(root, calib, 10.0)
    assert (root / "mav0/imu_imu_data").exists() and not (root / "mav0/cam0/sensor.yaml").exists()


def test_one_trigger_with_centre_stamps_10us_apart_is_restamped(tmp_path):
    big = 1674219531241094048
    root, calib = _tree(tmp_path, right_stamps=("1", str(big + 9984)),
                        sof={"cam0": [(1, 0), (big, big - 15_000_000)], "cam1": [(1, 0), (big + 9984, big - 15_000_000)]})
    root.joinpath("mav0/cam0/data.csv").write_text(f"#timestamp [ns],filename\n1,1.png\n{big},{big}.png\n")
    rep = sy.convert(root, calib, 10.0)
    assert rep["right_restamped_to_left"] == 1
    assert root.joinpath("mav0/cam1/data.csv").read_text().splitlines()[2] == f"{big},{big + 9984}.png"


def test_different_triggers_are_refused_even_when_close(tmp_path):
    big = 1674219531241094048
    root, calib = _tree(tmp_path, right_stamps=("1", str(big + 9984)),
                        sof={"cam0": [(1, 0), (big, big - 15_000_000)], "cam1": [(1, 0), (big + 9984, big - 14_990_016)]})
    root.joinpath("mav0/cam0/data.csv").write_text(f"#timestamp [ns],filename\n1,1.png\n{big},{big}.png\n")
    with pytest.raises(SystemExit, match="not one trigger"):
        sy.convert(root, calib, 10.0)
