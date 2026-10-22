"""Operator authority through the console (design AGENTIC_7 §0b; arch §23, §29.3), against a fake live run directory:
a directive lands in directives_inbox/ as one complete file; an LLM proposal reaches the inbox only after a person
confirms it (author records both); an approval is written where the runtime verifies it, and only for a pending request.
"""
import json
import os

from fastapi.testclient import TestClient


def _client(tmp_path, monkeypatch):
    outer, scene = tmp_path / "run", tmp_path / "scene"
    run = outer / "s-live-20261023T000000Z"
    run.mkdir(parents=True)
    (run / "telemetry.jsonl").write_text("")
    scene.mkdir()
    json.dump({"scene_id": "s", "esdf": {"origin_m": [0, 0], "cell_m": 0.05, "rows": 4, "cols": 4},
               "target_xy_m": [0, 0], "entrances": [{"start_xy_m": [0, 0]}]}, open(scene / "scene.json", "w"))
    monkeypatch.setenv("ARIADNE_LIVE", f"{outer}:{scene}")
    import importlib
    import operator_console.server as srv
    importlib.reload(srv)
    return TestClient(srv.app), run


def test_directive_proposal_and_approval(tmp_path, monkeypatch):
    c, run = _client(tmp_path, monkeypatch)
    r = c.post("/api/directive", json={"kind": "keep_out", "polygon": [[0, 0], [1, 0], [1, 1]]})
    assert r.status_code == 200 and r.json()["author"] == "operator"
    files = list((run / "directives_inbox").glob("*.json"))
    assert len(files) == 1 and json.loads(files[0].read_text())["kind"] == "keep_out"
    assert c.post("/api/directive", json={"kind": "teleport"}).status_code == 400
    (run / "directives_proposed").mkdir()
    (run / "directives_proposed" / "llm-1.json").write_text(json.dumps({"id": "llm-1", "kind": "hold", "agent": 1,
                                                                       "text": "agent 1 is near a collapse"}))
    assert [p["id"] for p in c.get("/api/directives/proposed").json()] == ["llm-1"]
    assert len(list((run / "directives_inbox").glob("*.json"))) == 1          # not applied before confirmation
    r = c.post("/api/directives/proposed/llm-1/confirm?operator=maintainer")
    assert r.json()["author"] == "llm-proposed/confirmed-by:maintainer"
    assert len(list((run / "directives_inbox").glob("*.json"))) == 2
    (run / "approvals").mkdir()
    (run / "approvals" / "t5000ms-a0-c1.request.json").write_text(json.dumps({"approval_id": "t5000ms-a0-c1"}))
    assert [a["approval_id"] for a in c.get("/api/approvals").json()] == ["t5000ms-a0-c1"]
    assert c.post("/api/approval/nope/approved").status_code == 404
    assert c.post("/api/approval/t5000ms-a0-c1/approved?operator=maintainer").json()["verdict"] == "approved"
    assert json.loads((run / "approvals" / "t5000ms-a0-c1.json").read_text())["verdict"] == "approved"
    assert c.get("/api/approvals").json() == []
