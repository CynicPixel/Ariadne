"""The splat panel's backend (6.4; RECON_6 §1/§3, D190/D197): serves what the reconstruction service writes.

The service (python/reconstruction/ariadne_recon/live/service.py) keeps, in ARIADNE_SPLAT_DIR:
  table.json            {"label", "version", "submaps": [{sid, agent, chunk, T_W_anchor (row-major 4x4), ...}]}
  chunks/<sid>_<n>.splat  the submap's Gaussians in its ANCHOR frame (32 B/Gaussian), replaced when it changes
Both are written atomically (temp file + rename), so a reader never sees half a file.

Routes (own paths, B's console endpoints untouched):
  GET /splat                      the panel page (static/splat.html)
  GET /api/splat/table            the current table
  GET /api/splat/chunk/{name}     one chunk (name checked against <int>_<int>.splat)
  WS  /ws/splat                   pushes the table whenever its version changes (polled every 0.5 s)

Mount with `app.include_router(splat.router)`, or run alone: `uvicorn operator_console.splat:app`.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

STATIC = Path(__file__).with_name("static")
CHUNK = re.compile(r"^\d+_\d+\.splat$")
router = APIRouter()


def splat_dir() -> Path:
    d = os.environ.get("ARIADNE_SPLAT_DIR")
    if not d:
        raise HTTPException(503, "ARIADNE_SPLAT_DIR not set: no reconstruction service attached")
    return Path(d)


def read_table() -> dict:
    p = splat_dir() / "table.json"
    if not p.is_file():
        return {"label": "no reconstruction yet", "version": 0, "submaps": []}
    return json.loads(p.read_text())


@router.get("/splat")
def page() -> HTMLResponse:
    return HTMLResponse((STATIC / "splat.html").read_text())


@router.get("/api/splat/table")
def table() -> JSONResponse:
    return JSONResponse(read_table())


@router.get("/api/splat/chunk/{name}")
def chunk(name: str) -> FileResponse:
    if not CHUNK.match(name):
        raise HTTPException(400, "bad chunk name")
    p = splat_dir() / "chunks" / name
    if not p.is_file():
        raise HTTPException(404, "chunk not found (replaced?)")
    return FileResponse(p, media_type="application/octet-stream")


@router.websocket("/ws/splat")
async def ws(sock: WebSocket) -> None:
    await sock.accept()
    seen = -1
    try:
        while True:
            t = read_table()
            if t.get("version", 0) != seen:
                seen = t.get("version", 0)
                await sock.send_text(json.dumps(t))
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, RuntimeError):
        return


app = FastAPI(title="Ariadne splat panel")
app.include_router(router)
