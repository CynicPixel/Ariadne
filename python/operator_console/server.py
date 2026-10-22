"""Operator console stub backend (arch §29.3; design CONSOLE_1.10): FastAPI, WebSocket telemetry at 5 Hz, REST
for metadata and map snapshots. Two sources, one frame schema:

  ARIADNE_CHAIN=collab_run:mission_run:scene ...   replay of a committed run chain (Phase 1)
  ARIADNE_LIVE=run_dir:scene ...                   a running ariadne_live (design LIVE_RUNTIME_L1 D154): tails its
                                                   telemetry; every frame served is logged to console_frames.jsonl
  uvicorn operator_console.server:app --host 127.0.0.1 --port 8080
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .live import FrameLog, LiveRun
from .replay import TICK_HZ, frame, load_chain

STATIC = Path(__file__).with_name("static")
app = FastAPI(title="Ariadne console (stub)")
_chain = None
_live = None
_live_log = None


def live():
    """The live run, or None when serving a replay chain."""
    global _live, _live_log
    if "ARIADNE_LIVE" not in os.environ:
        return None
    if _live is None:
        run, scene = os.environ["ARIADNE_LIVE"].split(":")
        _live = LiveRun(Path(run), Path(scene))
        _live_log = FrameLog(Path(run))
    _live.poll()
    return _live


def chain():
    global _chain
    if _chain is None:
        collab, mission, scene = os.environ["ARIADNE_CHAIN"].split(":")
        _chain = load_chain(Path(collab), Path(mission), Path(scene))
    return _chain


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text())


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/api/meta")
def meta() -> JSONResponse:
    lv = live()
    return JSONResponse(lv.meta() if lv else chain().meta)


@app.get("/api/map/{i}.png")
def map_png(i: int) -> FileResponse:
    lv = live()
    snaps = lv.snapshots if lv else chain().snapshots
    if not 0 <= i < len(snaps):
        raise HTTPException(404, "no such map version")
    return FileResponse(snaps[i][1], media_type="image/png")


@app.get("/api/frame")
def frame_at(t: float, speed: float = 10.0) -> JSONResponse:
    """One frame at mission time t (static view, screenshots): events from 0..t included."""
    lv = live()
    if lv:
        f = lv.frame(-1, -1.0)   # live: the latest state, every event so far
        return JSONResponse(f)
    ch = chain()
    n = int(round(t * TICK_HZ / speed))
    f = frame(ch, n, speed)
    f["events"] = [e for e in ch.events if e["t_s"] <= f["t_s"]]
    return JSONResponse(f)


@app.websocket("/ws/telemetry")
async def telemetry(ws: WebSocket, speed: float = 10.0) -> None:
    await ws.accept()
    lv = live()
    if lv:
        client = f"{ws.client.host}:{ws.client.port}" if ws.client else "?"
        n, since = 0, -1.0
        try:
            while True:
                lv.poll()
                f = lv.frame(n, since)
                await ws.send_json(f)
                _live_log.put(client, f)
                since = max(since, max((e["t_s"] for e in f["events"]), default=since))
                n += 1
                if lv.ended:
                    await ws.send_json({"end": True, "t_s": f["t_s"]})
                    await ws.close()
                    return
                await asyncio.sleep(1.0 / TICK_HZ)
        except WebSocketDisconnect:
            return
    ch = chain()
    n = 0
    try:
        while n * speed / TICK_HZ <= ch.end_s + speed / TICK_HZ:
            await ws.send_json(frame(ch, n, speed))
            n += 1
            await asyncio.sleep(1.0 / TICK_HZ)
        await ws.send_json({"end": True, "t_s": ch.end_s})
        await ws.close()
    except WebSocketDisconnect:
        return
