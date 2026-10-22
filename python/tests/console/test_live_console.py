"""The console fed live (design LIVE_RUNTIME_L1 D154), against a fake ariadne_live run directory.

Telemetry appended WHILE the websocket is open must reach the client in later frames (the console tails, it does not
load once), and every frame served is logged to console_frames.jsonl. Runs in the console image:
  docker run --rm -v $PWD:/work -w /work/python ariadne-console:1 python -m pytest -q tests/console
"""
import json
import os

from fastapi.testclient import TestClient


def _scene(d):
    os.makedirs(d, exist_ok=True)
    json.dump({"scene_id": "s", "esdf": {"origin_m": [-1, -1], "cell_m": 0.05, "rows": 40, "cols": 60},
               "target_xy_m": [1.0, 0.5], "entrances": [{"start_xy_m": [0.0, 0.0]}, {"start_xy_m": [2.0, 0.0]}]},
              open(os.path.join(d, "scene.json"), "w"))


def test_live_frames_follow_the_telemetry(tmp_path, monkeypatch):
    run, scene = tmp_path / "run", tmp_path / "scene"
    run.mkdir()
    _scene(scene)
    tel = run / "telemetry.jsonl"
    tel.write_text(json.dumps({"type": "pose", "t_s": 1.0, "agent": 0, "xy": [0.1, 0.2], "yaw": 0.0}) + "\n")
    monkeypatch.setenv("ARIADNE_LIVE", f"{run}:{scene}")
    import importlib
    import operator_console.server as srv
    importlib.reload(srv)
    c = TestClient(srv.app)
    assert c.get("/api/meta").json()["live"] is True
    with c.websocket_connect("/ws/telemetry") as ws:
        f0 = ws.receive_json()
        assert f0["agents"][0]["x"] == 0.1 and f0["agents"][1]["x"] is None
        with open(tel, "a") as f:   # the run goes on: a decision and a later pose
            rec = {"t_s": 5.0, "agent": 1, "executed": True, "chooser": "classical", "result": "go",
                   "candidates_total": 2, "command": {"kind": "follow", "waypoints_xy_m": [[1.0, 1.0], [2.0, 1.5]]}}
            f.write(json.dumps({"type": "decision", "record": rec}) + "\n")
            f.write(json.dumps({"type": "pose", "t_s": 5.2, "agent": 1, "xy": [2.0, 0.3], "yaw": 1.0}) + "\n")
        f1 = ws.receive_json()
        assert f1["t_s"] == 5.2 and f1["agents"][1]["x"] == 2.0
        assert any("agent 1: go (follow to (2.0, 1.5))" in e["text"] and "executed" in e["text"] for e in f1["events"])
        (run / "results.yaml").write_text("mode: live\n")
        # A frame already in flight may still say ended: false; the stream must end within a few ticks.
        msgs = [ws.receive_json() for _ in range(2)]
        while not msgs[-1].get("end") and len(msgs) < 10:
            msgs.append(ws.receive_json())
        assert msgs[-1].get("end") is True
    log = [json.loads(l) for l in (run / "console_frames.jsonl").read_text().splitlines()]
    assert [r["seq"] for r in log][:2] == [0, 1] and log[1]["t_s"] == 5.2
