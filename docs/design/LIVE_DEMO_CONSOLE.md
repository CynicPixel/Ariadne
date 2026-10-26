# Hosted mission console — 2026-10-26

This amendment implements the agreed one-screen demo plan. It supersedes the
one-horizon expiry and skipped-map-export portions of AGENTIC_7 D169 for demo
mode. Evidence runs retain their lockstep chooser behavior.

## User flow

`infra/scripts/demo.sh` serves the new console at `/`. Legacy chain/live consoles
remain available through their existing environment configuration; `/demo` always
serves the new UI. A controller claims the shared worker; everyone else spectates.

1. Choose a preset and seed (Random chooses and displays a new seed).
2. Inspect the generated mesh and top-down geometry, explicitly labeled ground
   truth. Select existing entrance markers and a reachable target. There is no
   arbitrary spawn placement or cosmetic sensor degradation.
3. Deploy for 60, 120 or 300 simulation seconds. Setup gives way to observed
   reconstruction, estimated occupancy/routes, cameras with tracker features,
   and decision/approval cards. Ground truth remains an explicit separate view.
4. Hold is acknowledged by the runtime at a planning barrier; it is not an
   instantaneous physical stop. End run requests hold and orderly termination.
5. Replay recorded displayed observations/decisions without model calls, or train
   splats from the recorded reconstruction stream. A new live run of the same
   seed recreates the scene, not necessarily the autonomous execution.

The page and its panes never scroll. Decisions/evidence use bounded pages.
Laptop checks cover 1280×600, 1280×720, 1366×768, 1440×900 and 1920×1080.
Map keyboard controls: arrows move the cursor, Shift increases the step, Enter
places a target/region vertex, 1–3 toggle entrances, Escape cancels region drawing.

## Boundaries and contracts

- FastAPI owns one session controller and one telemetry tailer. An advisory
  process lock plus a launcher lock prevents concurrent simulations. The server
  must use one uvicorn worker. Existing mutation endpoints are blocked in hosted
  demo mode, so they cannot bypass controller ownership.
- `POST /api/demo/lease` returns an opaque controller token. Subsequent actions
  use `X-Controller-Token`; the browser heartbeats every five seconds. Thirty
  seconds without a heartbeat queues fleet hold. Spectators cannot mutate state.
- `generate`, `validate`, `start`, `directive`, `hold`, `stop`, `approval`, and
  `train` are POST actions under `/api/demo`. Start/directive require an
  `Idempotency-Key`. Coordinates and entrance indices are validated server-side;
  the worldgen clearance/reachability checker is authoritative for setup.
- `/api/demo/stream` broadcasts current state at 5 Hz; reconnection receives a
  complete snapshot. Payloads include a session ID, sequence and simulation time.
  Slow clients receive the latest state; they cannot backpressure the runtime.
- `/map/{version}`, `/media/{name}`, `/chunk/{name}`, `/table`, `/playback`,
  `/preview.png`, `/world.obj`, `/entrances`, and `/status` supply read-only data.
  Immutable media URLs include the session ID; cross-session requests are refused.
- Command states are queued, effective or rejected. Runtime intake records supply
  the acknowledgment. HTTP success alone never means that a command executed.
- Approvals are accepted only while pending and before their wall deadline. The
  outstanding supervisor decision ID must still match. Verdict files are atomic;
  duplicate/stale submissions fail. C++ independently checks candidate validity,
  original approval ID, exact approved goal and the approved risk envelope.

## Pipeline

ROS/Gazebo and algorithm boundaries remain unchanged. The frontend cannot publish
ROS commands. The hosted launcher uses `live_web_demo_v1.yaml`, Phase 4 planning,
radio coordination and live collaboration. The default exposed agent limit is two;
three requires an explicit deployment setting after hardware qualification.

`DemoCamera` exports at most 10 sensor frames/s per agent. JPEG work runs on a
separate latest-wins thread; overwritten display packets are counted. Each
atomic manifest names an immutable image and that image's actual tracker points.
This is a display channel, not a second feature detector. The worker callback
still obtains the rectified RGB image; its added cost must be measured on target
hardware. Maps are written before telemetry advertises their version.

The existing CPU reconstruction service seeds submaps from observed stereo depth.
The browser renders those records as points, capped at 200,000 displayed points,
with decoding in a Web Worker. Placement corrections update matrices. Geometry
and camera media are separate from telemetry. Historical chunks are retained for
recorded playback. The raw received reconstruction stream is saved as `recon.link`;
this contains received tee frames, not a guarantee that the display tee dropped no
keyframes. The canonical consumed sensor recording remains the full run substrate.

Post-run `--recorded-link` feeds the same ingestion path into the GPU trainer.
The controller uses only an already-installed GPU image, never an implicit cloud
launch or pull. Failure retains the observed point cloud. Trained models up to
500,000 Gaussians use Spark; larger models retain the bounded point representation.
Generated world geometry is never inserted into observed reconstruction.

Demo LLM supervision now retains one outstanding request for up to the configured
wall timeout, allowing a human review to finish. Deterministic planning continues
while it is pending. On arrival, the original goal is matched and validated against
the current candidates; an invalid override falls back. Request timeout or link
failure is visible in the trace. `supervisor.json` tells the console which request
is still actionable. There is no free-text command translator in this release.

## Run locally / provision a worker

Host dependencies: Python 3.12, Node/npm for the one-time browser build, Docker,
and the repository's pinned sim/core/orchestrator/reconstruction images. The host
controller needs Docker access; serve it through an HTTPS reverse proxy for judging.
It binds loopback by default and must not be started with multiple workers.

```sh
python3 -m venv /tmp/ariadne-demo
/tmp/ariadne-demo/bin/pip install -r python/operator_console/requirements-demo.txt
npm --prefix python/operator_console ci
npm --prefix python/operator_console run build
bash infra/scripts/live.sh build
# Build core runtime + ariadne_collab_worker in the pinned core-dev image as usual.
ARIADNE_DEMO_PYTHON=/tmp/ariadne-demo/bin/python bash infra/scripts/demo.sh 8080
```

Deployment settings:

- `ARIADNE_DEMO_MAX_AGENTS=2` (1–3): hardware-qualified fleet size.
- `ARIADNE_DEMO_VIDEO_URL`: published demo video URL; no guessed/placeholder link.
- `ARIADNE_RECON_GPU_IMAGE`: installed CUDA trainer image; default
  `ariadne/recon-gpu:build`. CPU reconstruction uses `ARIADNE_RECON_IMAGE` as before.
- `ARIADNE_LIVE_BINARY`: optional runtime binary path relative to the repository;
  default `build/docker/cpp/runtime/ariadne_live`. Its collab worker is a sibling.
- Existing image and AWS region settings are inherited by the launcher. Bedrock
  uses the existing credential path. Missing credentials fail visibly.

GitHub points to the repository remote. Paper uses `docs/Latex/main.pdf` if present;
that file currently belongs to the maintainer's uncommitted paper work. Video is
hidden until configured. Publishing these artifacts and hardware qualification
remain deployment gates, not claims made by a local browser test.

Artifacts live under `build/demo/`: isolated generated scenes and session run
folders containing configuration/scene identity, launcher logs, playback JSONL,
canonical runtime recordings and reconstruction input/output. A controller restart
requests shutdown of unfinished sessions before permitting another launch.

## Validation and release gates

Tests: `python/tests/console/test_demo.py`, opt-in
`RUN_BROWSER_TESTS=1 ... pytest python/tests/console/test_demo_browser.py`,
`DemoCamera.PublishesMatchedDownsampledImageAndFeatures`, and reconstruction
`test_recorded_live_stream_replays_identical_geometry`. The browser integration
uses synthetic runtime telemetry and real world generation; it does not count as
Gazebo or Bedrock acceptance.

Release targets remain measurements to perform on the deployed worker/laptop:
RTF ≥0.95 for every enabled fleet size; ≥30 fps interactive viewport; p95 primary
camera age <500 ms and telemetry age <1 s; warm page usable within two seconds;
real approval/fallback and controller-disconnect exercises. Test both stale media
and reconstruction failure without losing mission control. Do not label the demo
real-time solely because its configured Gazebo real-time factor is 1.

The repository-wide Phase 1 premerge was attempted during this implementation and
failed before tests because the existing FetchContent subbuild cache records
`/data/code/Ariadne` while the container mounts `/work`. An isolated pinned-container
build is used for targeted verification; it does not substitute for that gate.

### Implementation verification (development tree)

- 49 Python/worldgen/reconstruction/browser checks passed; Chromium exercised
  setup, approval, hold, stop, playback, point rendering and laptop layouts.
- 14 targeted C++ checks passed, including matched image/features and live
  link/planner/lockstep gates. Runtime and collaboration worker compiled in the
  pinned OpenCV 5 container using `build/demo-cpp`.
- 5 mocked Bedrock/MCP orchestration checks passed in `ariadne-orchestrator:1`.
- Real local two-agent smoke: `build/demo-runtime-smoke/live-20261026T061547Z-9965`,
  ten simulation seconds, classical chooser, Mesa software rendering. Both
  cameras exported 640×360 images with 500/495 tracked points and zero recorded
  display overwrites. Two occupancy snapshots and two reconstruction submaps were
  produced; the reconstruction service received 28 keyframes without service-side
  drops. This is a functional smoke, not a real-time or LLM performance result.
- Real controlled-stop smoke: `build/demo-stop-smoke/live-20261026T062336Z-820d`.
  Stop requested after camera time 2 s; runtime reported `operator_stopped: true`,
  `actual_end_s: 5`, exit 0, with MCAP and the reconstruction table preserved.
- Full Phase 1 premerge remains blocked at its existing FetchContent cache-path
  mismatch. GPU training/display performance, full hosted collaborative/Bedrock
  acceptance, final video URL and public HTTPS publication remain open.
