"""worldgen CLI. Run through infra/scripts/worldgen.sh (pinned ariadne-sim image).

  census   --preset P --seeds A:B --out DIR     achievable lengths (bands ignored)
  accept   --preset P --seeds A:B --out DIR     acceptance of the real generator
  preview  --preset P --seed N --out FILE.png   top-down map of one scene
  generate --preset P --seed N --out DIR        full scene directory (build.py)
           [--rig configs/sim/rig_stereo_v1.yaml]  ... plus mission_default.{sdf,json}, bridge

Every command that writes a directory writes manifest.json beside its results:
command, UTC time, git commit/dirty and image id (from the wrapper's environment),
generator version, python/numpy versions, preset name + digest, seed range.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import sys

import numpy as np

from . import GENERATOR_VERSION
from .params import load_preset


def _seeds(text: str) -> range:
    a, b = text.split(":")
    r = range(int(a), int(b))
    if len(r) == 0:
        raise argparse.ArgumentTypeError("empty seed range")
    return r


def _manifest(args, preset, extra=None) -> dict:
    m = {
        "tool": "dataset_tools.worldgen",
        "command": " ".join(sys.argv),
        "time_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": os.environ.get("ARIADNE_GIT_COMMIT", "unknown"),
        "git_dirty": os.environ.get("ARIADNE_GIT_DIRTY", "unknown"),
        "image_id": os.environ.get("ARIADNE_IMAGE_DIGEST", "unknown"),
        "generator_version": GENERATOR_VERSION,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "preset": {"name": preset.name, "digest": preset.digest(), "path": args.preset,
                   "values": preset.to_canonical()},
    }
    if extra:
        m.update(extra)
    return m


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dataset_tools.worldgen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("census", "accept"):
        p = sub.add_parser(name)
        p.add_argument("--preset", required=True)
        p.add_argument("--seeds", type=_seeds, required=True)
        p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
        p.add_argument("--out", required=True)
    for name in ("preview", "generate"):
        p = sub.add_parser(name)
        p.add_argument("--preset", required=True)
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--out", required=True)
        if name == "generate":
            p.add_argument("--rig", default=None, help="also compose mission_default.* for this rig")
            p.add_argument("--live-rtf", type=float, default=None,
                           help="with --rig: also compose mission_live.* (D151) at this real-time factor")
    args = ap.parse_args(argv)

    preset = load_preset(args.preset)
    if args.cmd in ("census", "accept"):
        from . import sweep
        os.makedirs(args.out, exist_ok=True)
        fn = sweep.census if args.cmd == "census" else sweep.acceptance
        result = fn(preset, args.seeds, workers=args.workers)
        _write_json(os.path.join(args.out, f"{args.cmd}.json"), result)
        _write_json(os.path.join(args.out, "manifest.json"),
                    _manifest(args, preset, {"seeds": [args.seeds.start, args.seeds.stop],
                                             "workers": args.workers}))
        brief = {k: v for k, v in result.items() if k not in ("rows", "failures")}
        print(json.dumps(brief, indent=1))
        if result["errors"]:
            print(f"ERROR: {result['errors']} seed(s) raised; tracebacks in {args.cmd}.json", file=sys.stderr)
            return 1
        return 0
    from .scene import generate
    s = generate(preset, args.seed)
    if args.cmd == "generate":
        from .build import write_scene
        os.makedirs(args.out, exist_ok=True)
        write_scene(s, args.out)
        if args.rig:
            from .mission import write_mission
            write_mission(s, args.out, args.rig)
            if args.live_rtf is not None:
                write_mission(s, args.out, args.rig, name="live", live=True, rtf=args.live_rtf)
        _write_json(os.path.join(args.out, "manifest.json"),
                    _manifest(args, preset, {"seed": args.seed, "scene_id": s.scene_id}))
    else:
        from .preview import write_preview
        write_preview(s, args.out)
    print(json.dumps({"scene_id": s.scene_id, "attempt": s.attempt, "target_room": s.target_room,
                      "lengths": s.lengths}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
