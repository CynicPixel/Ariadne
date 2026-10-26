"""The live tee contract (link.py, D231) and the service's --listen mode, against the mock tee."""
import json
import socket
import threading
import time

import numpy as np
import pytest

from ariadne_recon.live import link, mock_tee, service
from ariadne_recon.live.placement import briefing_T_W_O
from ariadne_recon.spool import load
from test_live import _scene, _spool_v3_rgb

LIVE = ["--backend", "cpu_init", "--training-image", "rgb"]


def _start(tmp_path, extra=()):
    """The service in --listen mode on its own thread; returns (socket path, out dir, thread)."""
    sock, out = str(tmp_path / "recon.sock"), tmp_path / "live"
    th = threading.Thread(target=service.main, args=(["--listen", sock, "--out", str(out), *LIVE, *extra],))
    th.start()
    return sock, out, th


def _send(sock, frames):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    for _ in range(200):
        try:
            s.connect(sock)
            break
        except (FileNotFoundError, ConnectionRefusedError):
            time.sleep(0.05)
    with s:
        for topic, payload in frames:
            s.sendall(link.frame_bytes(topic, payload))


def _finish(th, out):
    th.join(timeout=30)
    assert not th.is_alive()
    return json.loads((out / "table.json").read_text()), json.loads((out / "manifest.json").read_text())


def _t_w_o(agent, T):
    return "#t_w_o", json.dumps({"agent": agent, "t_s": 0.0, "xyzquat": mock_tee.xyzquat(T)}).encode()


def test_half_rgb_is_bit_exact_with_opencv_inter_area():
    import cv2
    rgb = np.random.default_rng(0).integers(0, 256, (36, 64, 3), dtype=np.uint8)
    assert np.array_equal(link.half_rgb(rgb), cv2.resize(rgb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))


def test_kf_round_trip_and_a_short_payload_is_refused():
    rng = np.random.default_rng(1)
    Z = rng.uniform(0.5, 4, (3, 4)).astype(np.float16).astype(np.float32)
    Z[0, 0] = 0.0                                                     # 0 = invalid, the tee's convention
    S = np.full((3, 4), 0.01, np.float32)
    rgb = rng.integers(0, 256, (6, 8, 3), dtype=np.uint8)
    hdr = {"agent": 2, "id": 7, "t_ns": 5, "pose_version": 0, "T_O_C": list(np.eye(4)[:3].ravel()),
           "K": [8, 8, 3.5, 2.5], "Kz": [4, 4, 1.5, 1], "w": 8, "h": 6, "zw": 4, "zh": 3, "training_image": "rgb"}
    p = link.encode_kf(hdr, Z, S, rgb)
    r = link.decode_kf(p)
    assert r["agent"] == 2 and np.array_equal(r["Z"], Z) and np.array_equal(r["rgb"], rgb)
    assert np.array_equal(r["S"], S.astype(np.float16).astype(np.float32))
    with pytest.raises(link.LinkError, match="expected"):
        link.decode_kf(p[:-1])


def test_t_w_o_quaternion_is_qw_last():
    c = np.sqrt(0.5)
    k, T = link.decode_t_w_o(json.dumps({"agent": 1, "t_s": 0, "xyzquat": [1, 2, 3, 0, 0, c, c]}).encode())
    assert k == 1                                                     # (0, 0, sin 45, cos 45): +90 deg about z
    np.testing.assert_allclose(T[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], [1, 2, 3])
    np.testing.assert_allclose(link.decode_t_w_o(_t_w_o(0, T)[1])[1], T, atol=1e-12)   # the mock's encoder inverts


def test_live_ingest_trains_on_the_same_pixels_as_the_replay(tmp_path):
    import yaml
    spool = _spool_v3_rgb(tmp_path / "spool")
    y = yaml.safe_load((spool / "spool.yaml").read_text())
    y["bundles"] = [b for b in y["bundles"] if b["agent"] == 0]      # one agent: identity placement both ways
    (spool / "spool.yaml").write_text(yaml.safe_dump(y))
    assert service.main(["--spool", str(spool), "--out", str(tmp_path / "replay"), *LIVE]) == 0
    sock, out, th = _start(tmp_path)
    mock_tee.send(sock, [str(spool)], None)
    liv, m = _finish(th, out)
    rep = json.loads((tmp_path / "replay" / "table.json").read_text())
    assert len(liv["submaps"]) == len(rep["submaps"]) == 1
    for a, b in zip(rep["submaps"], liv["submaps"]):                  # byte-identical splats, same placement
        assert (tmp_path / "replay" / "chunks" / a["chunk"]).read_bytes() == (out / "chunks" / b["chunk"]).read_bytes()
        assert a["T_W_anchor"] == b["T_W_anchor"]
    assert m["keyframes"] == 2 and m["link"] == {"frames": {"#t_w_o": 1, "#kf": 2}, "dropped": {}}


def test_live_places_by_t_w_o_and_drops_what_it_cannot_place(tmp_path):
    kfs = sorted(load(_spool_v3_rgb(tmp_path / "spool")).keyframes(), key=lambda b: (b.t_ns, b.agent))
    T = np.eye(4)
    T[:3, 3] = [4, 5, 0.5]
    frames = [_t_w_o(0, T)] + [("#kf", mock_tee.kf_payload(b)) for b in kfs]   # agent 1 gets no #t_w_o
    frames += [("#kf", b"\x02\x00\x00\x00{}"), ("#mystery", b"")]
    sock, out, th = _start(tmp_path)
    _send(sock, frames)
    t, m = _finish(th, out)
    assert {r["agent"] for r in t["submaps"]} == {0}
    np.testing.assert_allclose(np.asarray(t["submaps"][0]["T_W_anchor"]).reshape(4, 4), T, atol=1e-12)
    assert m["keyframes"] == 2 and m["link"]["dropped"] == {"unplaced": 2, "malformed": 1, "unknown_topic": 1}


def test_live_with_a_scene_places_by_the_briefing_until_t_w_o_arrives(tmp_path):
    kfs = sorted(load(_spool_v3_rgb(tmp_path / "spool")).keyframes(), key=lambda b: (b.t_ns, b.agent))
    scene = _scene(tmp_path / "scene", [(1.0, 2.0, 0.0), (5.0, 2.0, 180.0)])
    m0 = json.loads((scene / "mission_default.json").read_text())
    (scene / "mission_default.json").write_text(json.dumps({**m0, "agents": [{"agent": 0}, {"agent": 1}]}))
    sock, out, th = _start(tmp_path, ["--scene", str(scene)])
    _send(sock, [("#kf", mock_tee.kf_payload(b)) for b in kfs])     # no #t_w_o at all
    t, _ = _finish(th, out)
    assert {r["agent"] for r in t["submaps"]} == {0, 1}
    for r in t["submaps"]:
        np.testing.assert_allclose(np.asarray(r["T_W_anchor"]).reshape(4, 4), briefing_T_W_O(scene, r["agent"]),
                                   atol=1e-12)


@pytest.mark.parametrize("yaw_deg", [0, 90, 179.9, 180, -90, 270])
def test_xyzquat_round_trips_every_entrance_heading_including_180(yaw_deg):
    from test_live import se3
    T = se3(np.deg2rad(yaw_deg), [1, 2, 3])
    T[:3, :3] = T[:3, :3] @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]).T   # with the optical rotation, as the tee
    np.testing.assert_allclose(link.decode_t_w_o(_t_w_o(0, T)[1])[1], T, atol=1e-12)


def test_a_bad_t_w_o_is_dropped_not_fatal(tmp_path):
    sock, out, th = _start(tmp_path)
    _send(sock, [("#t_w_o", json.dumps({"agent": 0, "xyzquat": [0, 0, 0, 0, 0, 0, 0]}).encode()),
                 ("#t_w_o", b"not json")])
    th.join(timeout=30)
    assert not th.is_alive()
    m = json.loads((out / "manifest.json").read_text())
    assert m["link"]["dropped"] == {"malformed_t_w_o": 2} and m["keyframes"] == 0


def test_recorded_live_stream_replays_identical_geometry(tmp_path):
    """Post-run training input uses the exact received stream, including placement changes."""
    spool = _spool_v3_rgb(tmp_path / "spool")
    recording = tmp_path / "observed.link"
    sock, out, th = _start(tmp_path, ("--record-link", str(recording)))
    mock_tee.send(sock, [str(spool)], None)
    liv, _ = _finish(th, out)
    replay_out = tmp_path / "recorded-replay"
    assert service.main(["--recorded-link", str(recording), "--out", str(replay_out), *LIVE]) == 0
    rep = json.loads((replay_out / "table.json").read_text())
    assert len(liv["submaps"]) == len(rep["submaps"])
    for a, b in zip(liv["submaps"], rep["submaps"]):
        assert a["T_W_anchor"] == b["T_W_anchor"]
        assert (out / "chunks" / a["chunk"]).read_bytes() == (replay_out / "chunks" / b["chunk"]).read_bytes()
