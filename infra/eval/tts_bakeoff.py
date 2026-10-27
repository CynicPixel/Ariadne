#!/usr/bin/env python3
"""Narration TTS listening test ("bake-off") for the demo film (2026-10-27).

Open-weights voices only, each in its own venv (their pins conflict), driven by infra/eval/tts_bakeoff_aws.sh:

  python3 tts_bakeoff.py ming --repo DIR --model DIR --out DIR
  python3 tts_bakeoff.py chatterbox --out DIR
  python3 tts_bakeoff.py kokoro --out DIR
  python3 tts_bakeoff.py manifest --out DIR        # durations + measured wpm of every WAV -> DIR/manifest.json

Each clip is DIR/<voice>_<line>_<pace>.wav, pace = natural | 196 | brisk, plus one JSON line in DIR/clips.jsonl
(model, voice or description, controls). "196" uses the model's own speed control, calibrated to 196 words/min from
the natural clip's measured rate (Kokoro: continuous speed; Ming: its only speed control, BASIC's three levels).
"brisk" puts the pace into a Ming voice-design description instead, because VOICE_DESIGN rejects BASIC's controls.
A failing voice records its error and the others still run.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from pathlib import Path

LINES = {
    "L1": "Imagine a building after an earthquake. The lights are out, and somewhere inside, someone may need help.",
    "L2": "So Ariadne remembers where each robot entered, and refuses any loop that would teleport a robot.",
    "L3": "An AI on Amazon Bedrock reads the robots' map and camera frames, then picks their next goals; "
          "plain code checks every pick before anything moves.",
}
TARGET_WPM = 196
STYLES = {
    "ming-a": "calm, warm male documentary narrator, mid-thirties, clear neutral English, natural and unhurried",
    "ming-b": "confident male science presenter, slightly deep voice, crisp diction, natural warmth",
}
BRISK = ", speaking at a brisk, energetic documentary pace"
SEED = 626


def words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def speech_seconds(x, sr: int, floor_db: float = -40.0) -> float:
    """Length between the first and last 20 ms frame within 40 dB of the clip's peak (leading/trailing silence off)."""
    import numpy as np
    n = max(1, int(0.02 * sr))
    f = len(x) // n
    if f == 0:
        return len(x) / sr
    rms = np.sqrt((x[: f * n].reshape(f, n).astype(np.float64) ** 2).mean(1) + 1e-20)
    on = np.nonzero(20 * np.log10(rms / rms.max()) > floor_db)[0]
    return (on[-1] - on[0] + 1) * n / sr if len(on) else len(x) / sr


def measured_wpm(path: Path, text: str) -> float:
    import soundfile as sf
    x, sr = sf.read(str(path), always_2d=True)
    return 60 * words(text) / speech_seconds(x.mean(1), sr)


def record(out: Path, **row) -> None:
    with open(out / "clips.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def attempt(out: Path, name: str, fn, **row) -> bool:
    """Run one clip; on failure log it and carry on."""
    t0 = time.time()
    try:
        fn(out / name)
        record(out, file=name, gen_s=round(time.time() - t0, 2), **row)
        print("ok", name, flush=True)
        return True
    except Exception as e:  # noqa: BLE001 - one voice failing must not stop the test
        record(out, file=name, error=f"{type(e).__name__}: {e}", **row)
        traceback.print_exc()
        return False


def run_kokoro(out: Path) -> None:
    import numpy as np
    import soundfile as sf
    import torch
    from kokoro import KPipeline
    for voice, lang in (("am_michael", "a"), ("bm_george", "b")):   # a = US English, b = British English
        pipe = KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M")

        def say(text, speed):
            def go(p):
                torch.manual_seed(SEED)
                a = np.concatenate([np.asarray(r.audio) for r in pipe(text, voice=voice, speed=speed)])
                sf.write(str(p), a, 24000)
            return go
        for line, text in LINES.items():
            stem = f"kokoro-{voice}_{line}"
            row = dict(model="hexgrad/Kokoro-82M (kokoro 0.9.4)", voice=voice, line=line)
            if not attempt(out, f"{stem}_natural.wav", say(text, 1.0), pace="natural", speed=1.0, **row):
                continue
            s = min(2.0, max(0.5, TARGET_WPM / measured_wpm(out / f"{stem}_natural.wav", text)))
            say(text, s)(out / f"{stem}_196.wav")        # one refinement: speed is not exactly proportional to rate
            s = min(2.0, max(0.5, s * TARGET_WPM / measured_wpm(out / f"{stem}_196.wav", text)))
            attempt(out, f"{stem}_196.wav", say(text, s), pace="196", speed=round(s, 4), **row)


def run_chatterbox(out: Path) -> None:
    import soundfile as sf
    import torch
    from chatterbox.tts import ChatterboxTTS
    m = ChatterboxTTS.from_pretrained(device="cuda")
    for line, text in LINES.items():
        def go(p, text=text):
            torch.manual_seed(SEED)
            wav = m.generate(text)                           # default voice, no cloning reference
            sf.write(str(p), wav.squeeze(0).cpu().numpy(), m.sr)
        attempt(out, f"chatterbox_{line}_natural.wav", go, model="ResembleAI/chatterbox (chatterbox-tts 0.1.7)",
                voice="default", line=line, pace="natural", note="no speed control: natural pace only")


def run_ming(out: Path, repo: Path, model_dir: Path) -> None:
    import torch
    sys.path.insert(0, str(repo))
    from ming_tts import MingTTS
    m = MingTTS(str(model_dir))
    row = dict(model="inclusionAI/Ming-3.0-tts-0.5b")

    def synth(**kw):
        def go(p):
            torch.manual_seed(SEED)
            m.synthesize(text=kw.pop("text"), output_path=str(p), **kw)
        return go
    for line, text in LINES.items():
        for v, style in STYLES.items():
            for pace, s in (("natural", style), ("brisk", style + BRISK)):
                attempt(out, f"{v}_{line}_{pace}.wav",
                        synth(task="VOICE_DESIGN", text=text, controls={"style": s}, use_zero_spk_emb=True),
                        voice=f"voice design: {s}", line=line, pace=pace, **row)
        attempt(out, f"ming-james_{line}_natural.wav",
                synth(task="IP", text=text, controls={"ip": "US_James"}, use_zero_spk_emb=True),
                voice="built-in US_James", line=line, pace="natural", **row)
        # BASIC (speed/pitch/volume) needs a speaker reference: US_James's own natural clip, at speed 快速 (fast).
        ref = out / f"ming-james_{line}_natural.wav"
        if ref.exists():
            attempt(out, f"ming-james_{line}_196.wav",
                    synth(task="BASIC", text=text, controls={"speed": "快速", "pitch": "中", "volume": "中"},
                          prompt_wav_path=str(ref), use_spk_emb=True),
                    voice="US_James (its natural clip as the BASIC reference)", line=line, pace="196",
                    controls="speed 快速 (fast; BASIC has only slow/medium/fast), pitch/volume 中", **row)


def manifest(out: Path, extra: dict) -> None:
    import soundfile as sf
    rows = [json.loads(s) for s in (out / "clips.jsonl").read_text().splitlines()] if (out / "clips.jsonl").exists() else []
    final = {}
    for r in rows:                                          # the last attempt of a file wins
        final[r["file"]] = r
    for r in final.values():
        p = out / r["file"]
        if "error" in r or not p.exists():
            continue
        x, sr = sf.read(str(p), always_2d=True)
        if sr not in (24000, 44100) or x.shape[1] != 1:   # the film wants 24 or 44.1 kHz mono
            import torch
            import torchaudio
            y = torch.as_tensor(x.mean(1), dtype=torch.float32)
            if sr not in (24000, 44100):
                r["original_sample_rate"], y = sr, torchaudio.functional.resample(y, sr, 24000)
                sr = 24000
            sf.write(str(p), y.numpy(), sr)
            x, sr = sf.read(str(p), always_2d=True)
        r.update(sample_rate=sr, channels=x.shape[1], duration_s=round(len(x) / sr, 3),
                 speech_s=round(speech_seconds(x.mean(1), sr), 3), words=words(LINES[r["line"]]))
        r["wpm"] = round(60 * r["words"] / r["speech_s"], 1)
    doc = {"what": "narration TTS listening test for the demo film; open weights only", "target_wpm": TARGET_WPM,
           "wpm_method": "words / time between the first and last 20 ms frame within 40 dB of the clip's peak",
           "lines": LINES, "seed": SEED, **extra, "clips": sorted(final.values(), key=lambda r: r["file"])}
    (out / "manifest.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    ok = [r for r in final.values() if "wpm" in r]
    print(f"manifest: {len(ok)} clips ok, {len(final) - len(ok)} failed")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ming", "chatterbox", "kokoro", "manifest"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", type=Path)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--extra", default="{}", help="JSON merged into the manifest (instance, image digest, ...)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.mode == "ming":
        run_ming(a.out, a.repo, a.model)
    elif a.mode == "chatterbox":
        run_chatterbox(a.out)
    elif a.mode == "kokoro":
        run_kokoro(a.out)
    else:
        manifest(a.out, json.loads(a.extra))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
