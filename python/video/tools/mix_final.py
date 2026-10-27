"""Final cut: the assembled film sped to the target length, with the narration laid on its subtitle cues.

The picture is retimed uniformly (factor k = film length / target): output frame j shows source frame floor(j k).
Each narration take (build/video_assets/narration/wav/<take>.wav; a take is one sentence, which may span several
subtitle cues, named by its first cue's id) starts at its first cue's start / k and must end before the next take starts; a line longer than its slot is sped up with ffmpeg's atempo (never slowed), and
the factor is reported. Runs in the pinned Manim image (PyAV + libavfilter).

  docker run --rm --user $(id -u):$(id -g) -v $PWD:/repo -w /repo/python/video --entrypoint python \
      manimcommunity/manim:v0.22.0 tools/mix_final.py [--target 300] [--quality 1080p60]
Output: media/cut/ariadne_final_<quality>.mp4 (the subtitles are burned into the picture)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

HERE = Path(__file__).resolve().parents[1]
NARR = HERE.parents[1] / "build/video_assets/narration"
SR = 48000
BASE = 1.06  # every take is sped up this much (Ming's ~188 wpm -> ~199 wpm, within the approved 196 wpm pace)
PAD = 1.0  # s of picture per beat beyond its narration, before the leftover time is shared out
LAG = 2.0  # s a line may start late, or run past the next line's cue (narration trailing the picture slightly)
BREATH = 0.12  # s of silence between lines


def srt_cues(path: Path) -> list[tuple[float, float, str]]:
    def t(s: str) -> float:
        h, m, rest = s.split(":")
        sec, ms = rest.split(",")
        return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000
    blocks = [b.splitlines() for b in path.read_text().strip().split("\n\n")]
    return [(t(b[1].split(" --> ")[0]), t(b[1].split(" --> ")[1]), " ".join(b[2:])) for b in blocks]


def load_wav(path: Path) -> np.ndarray:
    """Mono float32 at SR."""
    c = av.open(str(path))
    rs = av.AudioResampler(format="flt", layout="mono", rate=SR)
    out = [f.to_ndarray().reshape(-1) for fr in c.decode(c.streams.audio[0]) for f in rs.resample(fr)]
    out += [f.to_ndarray().reshape(-1) for f in rs.resample(None)]
    c.close()
    return np.concatenate(out).astype(np.float32)


def atempo(x: np.ndarray, factor: float) -> np.ndarray:
    """Speed x up by factor (pitch kept), with ffmpeg's atempo through PyAV's filter graph."""
    g = av.filter.Graph()
    src = g.add_abuffer(format="flt", sample_rate=SR, layout="mono", time_base=Fraction(1, SR))
    tempo = g.add("atempo", f"{factor:.5f}")
    sink = g.add("abuffersink")
    src.link_to(tempo)
    tempo.link_to(sink)
    g.configure()
    frame = av.AudioFrame.from_ndarray(x.reshape(1, -1), format="flt", layout="mono")
    frame.sample_rate, frame.pts, frame.time_base = SR, 0, Fraction(1, SR)
    g.push(frame)
    g.push(None)
    out = []
    while True:
        try:
            out.append(g.pull().to_ndarray().reshape(-1))
        except (av.error.BlockingIOError, av.error.EOFError):
            break
    return np.concatenate(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=300.0)
    ap.add_argument("--quality", default="1080p60")
    ap.add_argument("--dry-run", action="store_true", help="place the narration and report; no video")
    a = ap.parse_args()
    cut = HERE / "media/cut"
    src_path = cut / f"ariadne_{a.quality}.mp4"
    cues = srt_cues(cut / f"ariadne_{a.quality}.srt")
    lines = [json.loads(l) for l in (NARR / "cues.jsonl").read_text().splitlines() if l.strip()]
    assert [c[2] for c in cues] == [r["text"] for r in lines], "the SRT and the cue file are out of step"
    takes = []  # (take id, film start of its first cue, text of its first cue)
    for c, r in zip(cues, lines):
        if not takes or takes[-1][0] != r["take"]:
            takes.append((r["take"], c[0], c[2]))

    inp = av.open(str(src_path))
    vs = inp.streams.video[0]
    fps = int(a.quality.split("p")[1])
    n_out = int(round(a.target * fps))

    # ---- the time map: each beat gets its own speed-up, so a talk-heavy beat is not squeezed like a quiet one ----------
    sys.path.insert(0, str(HERE))
    from assemble import ORDER  # noqa: E402
    beats, F = [], 0.0
    for stem, cls in ORDER:
        b = av.open(str(HERE / "media/videos" / stem / a.quality / f"{cls}.mp4"))
        n = sum(1 for _ in b.decode(video=0))
        b.close()
        beats.append([cls, F, n / fps])  # name, film start, film length
        F += n / fps
    film = F
    man = {l["id"]: l for l in json.loads((NARR / "manifest.json").read_text())["lines"]}
    speech = {cls: 0.0 for cls, _, _ in beats}
    for tid, t0, _ in takes:
        cls = next(b[0] for b in beats if b[1] <= t0 < b[1] + b[2])
        speech[cls] += man[tid]["duration_s"] / BASE + BREATH
    T0 = [speech[cls] + PAD for cls, _, _ in beats]
    spare = [max(0.0, L - t) for (_, _, L), t in zip(beats, T0)]
    extra = a.target - sum(T0)
    assert extra >= 0, f"narration alone needs {sum(T0):.1f} s > {a.target} s"
    T = [t + extra * s_ / sum(spare) for t, s_ in zip(T0, spare)]
    O = np.r_[0, np.cumsum(T)]

    def fwd(t: float) -> float:  # film time -> output time
        for (cls, f0, L), o, tb in zip(beats, O, T):
            if t < f0 + L or cls == beats[-1][0]:
                return o + (t - f0) * tb / L
        return a.target

    def inv(t: float) -> float:  # output time -> film time
        for (cls, f0, L), o, tb in zip(beats, O, T):
            if t < o + tb or cls == beats[-1][0]:
                return f0 + (t - o) * L / tb
        return film

    # ---- audio: every line at its cue, sped up only where it would run into the next ----------------------------------
    audio = np.zeros(int(a.target * SR) + SR, np.float32)
    report = []
    prev_end = 0.0
    for i, (cid, t0, txt) in enumerate(takes):
        cue = fwd(t0)
        start = max(cue, prev_end + BREATH)  # a line may wait for the previous one to finish...
        nxt = fwd(takes[i + 1][1]) if i + 1 < len(takes) else a.target
        avail = min(nxt + LAG, a.target) - start - BREATH  # ...and may run up to LAG past the next cue
        x = load_wav(NARR / "wav" / f"{cid}.wav")
        f = max(BASE, len(x) / SR / avail)  # BASE, plus whatever is still left over
        if f > 1.0:
            x = atempo(x, f)
        s = int(start * SR)
        audio[s:s + len(x)] += x[: len(audio) - s]
        prev_end = start + len(x) / SR
        report.append((cid, round(start - cue, 2), round(avail, 2), round(f, 3), txt[:60]))
    if a.dry_run:
        for (cls, _, L), tb in zip(beats, T):
            print(f"  {cls:14s} {L:5.1f} s -> {tb:5.1f} s (x{L / tb:.2f})")
        sped = [r for r in report if r[3] > BASE + 1e-6]
        print(f"{len(sped)} takes above x{BASE}, max x{max(r[3] for r in report):.3f}; max lag {max(r[1] for r in report):.2f} s")
        for r in sped:
            print("  ", *r)
        return
    peak = float(np.abs(audio).max())
    audio *= 0.89 / peak if peak > 0 else 1.0  # -1 dBFS peak

    # ---- picture: uniform retime ------------------------------------------------------------------------------------
    out = av.open(str(cut / f"ariadne_final_{a.quality}.mp4"), "w")
    v = out.add_stream("libx264", rate=fps)
    v.width, v.height, v.pix_fmt = vs.codec_context.width, vs.codec_context.height, "yuv420p"
    v.options = {"crf": "18", "preset": "medium"}
    v.time_base = Fraction(1, fps)
    au = out.add_stream("aac", rate=SR)
    au.layout = "stereo"
    j = 0
    src_of = [int(inv(jj / fps) * fps) for jj in range(n_out)]  # nondecreasing
    for i, frame in enumerate(inp.decode(vs)):
        while j < n_out and src_of[j] <= i:
            fr = frame.reformat(format="yuv420p")
            fr.pts, fr.time_base = j, Fraction(1, fps)
            for p in v.encode(fr):
                out.mux(p)
            j += 1
    for p in v.encode():
        out.mux(p)
    pcm = audio[: int(a.target * SR)]
    st = np.repeat(pcm.reshape(1, -1), 2, axis=0)
    for o in range(0, st.shape[1], 1024):
        fr = av.AudioFrame.from_ndarray(np.ascontiguousarray(st[:, o:o + 1024]), format="fltp", layout="stereo")
        fr.sample_rate, fr.pts, fr.time_base = SR, o, Fraction(1, SR)
        for p in au.encode(fr):
            out.mux(p)
    for p in au.encode():
        out.mux(p)
    out.close()
    inp.close()

    sped = [r for r in report if r[3] > BASE + 1e-6]
    print(f"film {film:.1f} s -> {a.target:.1f} s; {len(report)} takes, {len(sped)} above x{BASE}, "
          f"max x{max([r[3] for r in report]):.3f}")
    for r in sped:
        print("  ", *r)


if __name__ == "__main__":
    main()
