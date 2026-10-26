"""Hosted demo boundary tests: no Docker, cloud or real mission is launched."""
import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from operator_console import demo
from operator_console.live import LiveRun


@pytest.fixture
def worker(tmp_path, monkeypatch):
    monkeypatch.setenv("ARIADNE_DEMO", "1")
    m = demo.Demo(tmp_path)
    monkeypatch.setattr(demo, "manager", m)
    app = FastAPI()
    app.include_router(demo.router)
    return m, TestClient(app)


def ready(m):
    m.scene = m.root / "scene"
    m.scene.mkdir()
    sc = {"scene_id": "test", "esdf": {"origin_m": [0, 0], "cell_m": 1, "rows": 10, "cols": 10, "file": "esdf.bin"},
          "clearance": {"rho_responder_m": .4}, "target_xy_m": [5, 5],
          "entrances": [{"start_xy_m": [1, 1]}, {"start_xy_m": [8, 8]}]}
    (m.scene / "scene.json").write_text(json.dumps(sc))
    np.full((10, 10), 1, dtype="<f4").tofile(m.scene / "esdf.bin")
    (m.scene / "mission_live.json").write_text(json.dumps({"agents": [{}, {}]}))
    m.state = "ready"
    m.scene_spec = {"scene_id": "test", "preset": "ci_tiny", "seed": 0}
    m.meta_cache = LiveRun(m.root, m.scene).meta()


def running(m):
    ready(m)
    m.run = m.root / "run"
    m.run.mkdir()
    m.session = "test-session"
    (m.run / "telemetry.jsonl").write_text(json.dumps({"type": "pose", "t_s": 1., "agent": 0, "xy": [1, 1], "yaw": 0}) + "\n")
    m.lv = LiveRun(m.run, m.scene)
    m.state = "running"
    return m.run


def headers(c):
    r = c.post('/api/demo/lease')
    assert r.status_code == 200
    return {"X-Controller-Token": r.json()["token"], "Idempotency-Key": "test-command"}


def test_controller_and_spectator_boundary(worker):
    m, c = worker
    h = headers(c)
    assert c.post('/api/demo/lease').status_code == 403
    assert c.post('/api/demo/hold').status_code == 403
    assert c.post('/api/demo/heartbeat', headers=h).status_code == 200
    assert 'token' not in c.get('/api/demo/status').text
    assert c.post('/api/demo/generate', headers=h, json={"preset": "../../bad", "seed": 0}).status_code == 422
    assert c.post('/api/demo/generate', headers=h, json={"seed": True}).status_code == 422


def test_target_and_roster_validation(worker):
    m, c = worker
    ready(m)
    h = headers(c)
    for spec in ({"entrances": [0, 0]}, {"entrances": [9]}, {"target": [-1, 0]}, {"target": [1]},
                 {"entrances": [True]}, {"entrances": []}):
        assert c.post('/api/demo/validate', headers=h, json=spec).status_code == 422
    assert c.post('/api/demo/validate', headers=h, json={"entrances": [1], "target": [5, 5]}).json()['valid']


def test_directive_idempotency_ack_and_controller_loss(worker):
    m, c = worker
    src = running(m)
    h = headers(c)
    body = {"kind": "hold", "agent": 1}
    for _ in range(2):
        r = c.post('/api/demo/directive', headers=h, json=body)
        assert r.status_code == 200 and r.json()['state'] == 'queued'
    assert len(list((src / 'directives_inbox').glob('*.json'))) == 1
    assert c.post('/api/demo/directive', headers={**h, "Idempotency-Key": "../bad"}, json=body).status_code == 400
    (src / 'directives.jsonl').write_text(json.dumps({"id": "test-command", "effective_ns": 5e9})+'\n')
    m.tick()
    assert m.commands['test-command']['state'] == 'effective'
    assert len(m.snapshot['agents']) == 2  # agent 1 is waiting, not missing
    m.heartbeat = time.monotonic() - 31
    m.tick()
    assert len(list((src / 'directives_inbox').glob('*.json'))) == 3
    m.tick()
    assert len(list((src / 'directives_inbox').glob('*.json'))) == 3
    assert c.post('/api/demo/lease').status_code == 200


def test_approval_expiry_duplicate_and_stop(worker):
    m, c = worker
    src = running(m)
    h = headers(c)
    a = src / 'approvals'
    a.mkdir()
    for aid, delta in [('expired', -1), ('pending', 30)]:
        (a / f'{aid}.request.json').write_text(json.dumps({"approval_id": aid, "decision_id": "d1",
                                                          "deadline_unix_s": time.time()+delta}))
    assert c.post('/api/demo/approval', headers=h, json={"id": "expired", "verdict": "approved"}).status_code == 409
    body = {"id": "pending", "verdict": "approved"}
    assert c.post('/api/demo/approval', headers=h, json=body).status_code == 200
    assert c.post('/api/demo/approval', headers=h, json=body).status_code == 409
    assert c.post('/api/demo/stop', headers=h).json()['state'] == 'stopping'
    assert (src / 'stop.request.json').exists()


def test_snapshot_and_playback_have_no_ground_truth(worker):
    m, c = worker
    src = running(m)
    with (src / 'telemetry.jsonl').open('a') as f:
        f.write(json.dumps({"type":"gt_pose", "t_s":1., "agent":0, "xy":[9, 9], "yaw":0})+'\n')
    m.tick()
    assert all('gt' not in a for a in m.snapshot['agents'])
    assert c.get('/api/demo/playback').status_code == 409
    m.state = 'completed'
    r = c.get('/api/demo/playback')
    assert r.status_code == 200
    assert json.loads(r.text)['agents'][0]['x'] == 1
    with c.websocket_connect('/api/demo/stream') as ws:
        assert ws.receive_json()['session_id'] == 'test-session'


def test_generation_failure_is_visible(worker, monkeypatch):
    m, c = worker
    h = headers(c)
    # The fixture has no presets; failures must be reported, not leave a spinner forever.
    assert c.post('/api/demo/generate', headers=h, json={"seed": 1}).status_code == 200
    deadline = time.monotonic()+5
    while m.state == 'generating' and time.monotonic()<deadline:
        time.sleep(.01)
    assert m.state == 'failed' and 'Generation failed' in m.error


def test_approval_for_superseded_model_request_is_not_actionable(worker):
    m, c = worker
    src = running(m)
    h = headers(c)
    (src / 'approvals').mkdir()
    (src / 'approvals/old.request.json').write_text(json.dumps({
        'approval_id':'old', 'decision_id':'previous', 'deadline_unix_s':time.time()+60}))
    (src / 'supervisor.json').write_text(json.dumps({'decision_id':'current'}))
    assert m.approvals() == []
    assert c.post('/api/demo/approval', headers=h, json={'id':'old','verdict':'approved'}).status_code == 409


def test_artifacts_cannot_cross_session_boundaries(worker):
    m, c = worker
    running(m)
    assert c.get('/api/demo/media/cam_0_100.jpg?session=previous').status_code == 409
    assert c.get('/api/demo/chunk/0_1.splat?session=previous').status_code == 409
    assert c.get('/api/demo/map/0?session=previous').status_code == 409
    m.tick()
    assert all('gt' not in a for a in m.snapshot['agents'])
