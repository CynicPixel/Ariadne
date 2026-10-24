"""Join the rendered scenes into one film and merge their cue sheets into one subtitle file (SRT) for the voice pass.

  docker run --rm --user $(id -u):$(id -g) -v $PWD:/repo -w /repo/python/video --entrypoint python \
      manimcommunity/manim:v0.22.0@sha256:3c3c... assemble.py [480p15|1080p60]

Output: media/cut/ariadne_<quality>.mp4, media/cut/ariadne_<quality>.srt, and a per-scene timing table on stdout.
Runs inside the pinned Manim image (PyAV; there is no ffmpeg binary). Re-encodes H.264, so the scenes may differ in
nothing but content.
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import av

HERE = Path(__file__).resolve().parent
ORDER = [  # (scene file stem, class): the film's running order
    ("s01_worldgen", "WorldGen"), ("s03_seeing", "Seeing"), ("s04a_ba", "BundleAdjust"),
    ("s05_sharing", "Sharing"), ("s06_trust", "Trust"),
    ("s07_cbba", "SharingWork"), ("s08_agentic", "HumanInLoop"), ("s09_thread", "Thread"),
    ("s10_splat", "SharedModel"), ("s11_architecture", "Architecture"), ("s12_close", "Close"),
]


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def main(quality: str = "480p15") -> int:
    media = HERE / "media"
    out_dir = media / "cut"
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = int(quality.split("p")[1])
    out = av.open(str(out_dir / f"ariadne_{quality}.mp4"), "w")
    stream = None
    t0, n_out, cues, rows = 0.0, 0, [], []
    for stem, cls in ORDER:
        src = media / "videos" / stem / quality / f"{cls}.mp4"
        if not src.exists():
            print(f"MISSING {src}")
            return 1
        inp = av.open(str(src))
        vs = inp.streams.video[0]
        if stream is None:
            stream = out.add_stream("libx264", rate=fps)
            stream.width, stream.height, stream.pix_fmt = vs.codec_context.width, vs.codec_context.height, "yuv420p"
            stream.time_base = Fraction(1, fps)
        n = 0
        for frame in inp.decode(vs):
            frame = frame.reformat(width=stream.width, height=stream.height, format="yuv420p")
            frame.pts, frame.time_base = n_out, Fraction(1, fps)
            for pkt in stream.encode(frame):
                out.mux(pkt)
            n += 1
            n_out += 1
        dur = n / fps
        cap = media / "captions" / f"{cls}.json"
        for c in json.loads(cap.read_text()) if cap.exists() else []:
            cues.append((t0 + c["start_s"], t0 + c["start_s"] + c["dur_s"], c["text"]))
        rows.append((stem, cls, t0, dur))
        t0 += dur
        inp.close()
    for pkt in stream.encode():
        out.mux(pkt)
    out.close()
    (out_dir / f"ariadne_{quality}.srt").write_text(
        "\n".join(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{txt}\n" for i, (a, b, txt) in enumerate(cues, 1)))
    words = sum(len(t.split()) for _, _, t in cues)
    for stem, cls, start, dur in rows:
        print(f"{srt_time(start)[:8]}  {dur:6.1f} s  {stem}.{cls}")
    print(f"TOTAL {t0:.1f} s ({t0 / 60:.2f} min), {len(cues)} cues, {words} narration words")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
