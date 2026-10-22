"""The Bedrock orchestrator in the CHOOSE step (design AGENTIC_7 §0, D156-D160; arch §25).

  python -m orchestration.orchestrator --socket RUN/chooser.sock --dir RUN/decisions [--max-calls 10]

ariadne_live (chooser: llm) connects to SOCKET and sends one JSON line per replan: a joint decision request for both
agents. For each, this process writes the request where the MCP server can read it, runs a Bedrock `converse` tool
loop whose tools are the MCP server's (stdio, mcp 2.3.0), and answers with one JSON line:
  {"decision_id", "assignments": [{agent, candidate (-1 = hold), reason, approval_id}], "orchestrator": {...}}
`assignments` is whatever the `assign` tool recorded, or null. ariadne_live validates every assignment again, per
agent (range, exactly-once, approval for an above-threshold pick) and falls back to the classical choice otherwise.

Models (D84): primary eu.amazon.nova-2-lite-v1:0 (text + image input). On a Bedrock error or throttling (never on a
"bad" answer) the turn restarts on deepseek.v3.2, which takes TEXT only: a second MCP server with --text-only (no
observe_image). temperature 0. Region eu-north-1. Credentials: the boto3 chain (instance role on AWS; on the dev host
live.sh exports the current `aws login` session into the environment). Images are sent to the model as bytes and
logged as path + sha256, never inline.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import sys
import time
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from mcp import StdioServerParameters
from mcp.client.client import Client

PRIMARY = "eu.amazon.nova-2-lite-v1:0"
FALLBACK = "deepseek.v3.2"
SYSTEM = (
    "You direct two search agents exploring a dark, partially collapsed building, mapped only from their stereo "
    "cameras. Each turn you assign each agent its next exploration target among candidates a deterministic planner "
    "computed. Use the tools to look at the map (and, if available, its image and the agents' camera keyframes), "
    "the agents' pose certainty, the responder egress route and the candidates; then call assign exactly once. "
    "Prefer targets that reveal more unknown space (larger clusters) per metre driven, do not send both agents to "
    "the same cluster when another is available, keep risk low, and remember the mission: find a route a human "
    "responder can use. A candidate whose risk exceeds the threshold needs request_human_approval first; if the "
    "verdict is not 'approved', pick another candidate or hold. "
    "Operator directives are the operator's orders and come before your own preferences: favour candidates inside an "
    "operator region for that agent (priority_weight > 1, or to_directive_m 0), else the ones nearer it. "
    "Changing an agent's goal costs a turn in place and the progress made so far: keep its current_task (its "
    "`candidate`) unless that task is reached or gone, an operator directive points elsewhere, or another candidate "
    "is clearly better, and say which of these when you switch. Keep each reason to one sentence.")


def mcp_params(ddir: Path, text_only: bool) -> StdioServerParameters:
    """The MCP server as a stdio subprocess. mcp's stdio client passes only HOME/LOGNAME/PATH/SHELL/TERM/USER to it
    (DEFAULT_INHERITED_ENV_VARS, mcp 2.3.0), so PYTHONPATH is dropped: run it from python/ with this interpreter."""
    return StdioServerParameters(command=sys.executable, cwd=str(Path(__file__).resolve().parents[1]),
                                 args=["-m", "orchestration.mcp_server", str(ddir)] + (["--text-only"] if text_only else []))


def tool_config(tools) -> dict:
    return {"tools": [{"toolSpec": {"name": t.name, "description": t.description or "",
                                    "inputSchema": {"json": t.input_schema}}} for t in tools],
            "toolChoice": {"any": {}}}


def to_bedrock(result, req: dict) -> tuple[list, str, list]:
    """MCP tool result -> Bedrock toolResult content, a short text for the trace, and image log entries."""
    content, text, images = [], [], []
    for c in result.content:
        if getattr(c, "type", "") == "image":
            raw = base64.b64decode(c.data)
            fmt = c.mime_type.split("/")[-1]
            content.append({"image": {"format": fmt, "source": {"bytes": raw}}})
            images.append({"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "format": fmt})
        else:
            content.append({"text": c.text})
            text.append(c.text)
    if not content:
        content.append({"text": ""})
    return content, "".join(text), images


async def run_turn(req: dict, ddir: Path, mcp: Client, tools_cfg: dict, br, model: str, max_calls: int,
                   log: dict) -> bool:
    """One Bedrock conversation; True if `assign` recorded a choice. Raises ClientError/BotoCoreError upward."""
    did = req["decision_id"]
    msgs = [{"role": "user", "content": [{"text": (
        f"decision_id={did}. Mission time {req['t_s']:.1f} s. Agents to assign: "
        f"{[a['agent'] for a in req['agents']]}. Inspect, then assign.")}]}]
    while log["calls"] < max_calls:
        resp = await asyncio.to_thread(br.converse, modelId=model, system=[{"text": SYSTEM}], messages=msgs,
                                       toolConfig=tools_cfg, inferenceConfig={"temperature": 0.0, "maxTokens": 600})
        log["calls"] += 1
        log["model"] = model
        u = resp.get("usage", {})
        log["input_tokens"] += u.get("inputTokens", 0)
        log["output_tokens"] += u.get("outputTokens", 0)
        log["stop"] = resp.get("stopReason")
        out = resp["output"]["message"]
        msgs.append(out)
        for c in out["content"]:
            if "text" in c and c["text"].strip():
                log["model_text"].append(c["text"][:400])
        uses = [c["toolUse"] for c in out["content"] if "toolUse" in c]
        if not uses:
            msgs.append({"role": "user", "content": [{"text": "Call a tool; finish with assign."}]})
            continue
        results = []
        for tu in uses:
            args = dict(tu.get("input") or {})
            args["decision_id"] = did            # this request's id, whatever the model passed
            entry = {"tool": tu["name"], "input": {k: v for k, v in args.items() if k != "decision_id"}}
            try:
                r = await mcp.call_tool(tu["name"], args)
                content, text, images = to_bedrock(r, req)
                status = "error" if r.is_error else "success"
            except Exception as e:  # noqa: BLE001 - every tool failure goes back to the model and into the trace
                content, text, images, status = [{"text": f"{type(e).__name__}: {e}"}], f"{type(e).__name__}: {e}", [], "error"
            entry.update(status=status, result=text[:600], images=images)
            log["tool_calls"].append(entry)
            results.append({"toolResult": {"toolUseId": tu["toolUseId"], "status": status, "content": content}})
        msgs.append({"role": "user", "content": results})
        if (ddir / f"{did}.choice.json").exists():
            return True
    return (ddir / f"{did}.choice.json").exists()


async def decide(req: dict, ddir: Path, mcps: dict, cfgs: dict, br, max_calls: int) -> dict:
    did = req["decision_id"]
    (ddir / f"{did}.json").write_text(json.dumps(req))
    log = {"model": None, "fallback_used": False, "calls": 0, "latency_s": 0.0, "input_tokens": 0,
           "output_tokens": 0, "tool_calls": [], "model_text": [], "stop": None, "errors": []}
    t0 = time.time()
    for model, kind in ((PRIMARY, "multimodal"), (FALLBACK, "text")):
        try:
            if await run_turn(req, ddir, mcps[kind], cfgs[kind], br, model, max_calls, log):
                break
            if model == PRIMARY:
                break   # the model answered but never assigned: not a Bedrock failure, no fallback (C++ falls back)
        except (ClientError, BotoCoreError) as e:
            log["errors"].append(f"{model}: {type(e).__name__}: {str(e)[:200]}")
            log["fallback_used"] = model == PRIMARY
    log["latency_s"] = round(time.time() - t0, 3)
    choice = ddir / f"{did}.choice.json"
    return {"decision_id": did, "assignments": json.loads(choice.read_text())["assignments"] if choice.exists() else None,
            "orchestrator": log}


async def serve(args) -> None:
    ddir = Path(args.dir)
    ddir.mkdir(parents=True, exist_ok=True)
    br = boto3.client("bedrock-runtime", region_name="eu-north-1",
                      config=Config(retries={"max_attempts": 2, "mode": "standard"}, read_timeout=60))
    async with Client(mcp_params(ddir, False)) as m_mm, Client(mcp_params(ddir, True)) as m_tx:
        mcps = {"multimodal": m_mm, "text": m_tx}
        cfgs = {k: tool_config((await c.list_tools()).tools) for k, c in mcps.items()}
        (ddir / "tools.json").write_text(json.dumps(cfgs, indent=1))
        lock = asyncio.Lock()

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            pending = set()

            async def one(line: bytes) -> None:
                ans = await decide(json.loads(line), ddir, mcps, cfgs, br, args.max_calls)
                async with lock:
                    writer.write((json.dumps(ans) + "\n").encode())
                    await writer.drain()

            while line := await reader.readline():
                t = asyncio.create_task(one(line))
                pending.add(t)
                t.add_done_callback(pending.discard)
            await asyncio.gather(*pending)

        srv = await asyncio.start_unix_server(handle, path=args.socket)
        print(f"orchestrator: tools {[t['toolSpec']['name'] for t in cfgs['multimodal']['tools']]}; "
              f"listening on {args.socket}", flush=True)
        async with srv:
            await srv.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--max-calls", type=int, default=10)
    asyncio.run(serve(ap.parse_args()))


if __name__ == "__main__":
    main()
