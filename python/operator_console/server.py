"""Operator console stub backend (arch §29.3; design CONSOLE_1.10): FastAPI, WebSocket telemetry at 5 Hz, REST
for metadata and map snapshots. In Phase 1 it replays a committed run chain; live runs will publish the same frames.

  ARIADNE_CHAIN=collab_run:mission_run:scene uvicorn operator_console.server:app --host 127.0.0.1 --port 8080
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .replay import TICK_HZ, frame, load_chain

STATIC = Path(__file__).with_name("static")
app = FastAPI(title="Ariadne console (stub)")
_chain = None


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
    return JSONResponse(chain().meta)


@app.get("/api/map/{i}.png")
def map_png(i: int) -> FileResponse:
    snaps = chain().snapshots
    if not 0 <= i < len(snaps):
        raise HTTPException(404, "no such map version")
    return FileResponse(snaps[i][1], media_type="image/png")


@app.get("/api/frame")
def frame_at(t: float, speed: float = 10.0) -> JSONResponse:
    """One frame at mission time t (static view, screenshots): events from 0..t included."""
    ch = chain()
    n = int(round(t * TICK_HZ / speed))
    f = frame(ch, n, speed)
    f["events"] = [e for e in ch.events if e["t_s"] <= f["t_s"]]
    return JSONResponse(f)


@app.websocket("/ws/telemetry")
async def telemetry(ws: WebSocket, speed: float = 10.0) -> None:
    await ws.accept()
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
