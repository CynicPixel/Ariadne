#!/usr/bin/env python3
"""Demo-film narration with Ming-3.0-tts-0.5b's built-in voice US_James (picked by the maintainer from the listening
test, infra/eval/tts_bakeoff.py), driven by infra/eval/narration_aws.sh:

  python3 narration_ming.py --cues cues.jsonl --repo DIR --model DIR --out DIR

Cues sharing a `take` (the first cue id of their sentence) are joined with single spaces in id order into one take,
written as DIR/wav/<take>.wav. Every take's text gets the bake-off's exact call (task IP, controls {"ip": "US_James"}, zero speaker
embedding), with no speed control. Each take is checked, because nobody listens on the instance:
  intelligible  Whisper (small.en) transcript vs the spoken text, character error rate on letters and digits only,
                so "Open C V" and "OpenCV" match; CER > MAX_CER is a failed take
  not clipped   at most CLIP_SAMPLES samples at |x| >= 0.999
  plausible     measured rate within WPM_RANGE
A failed take is re-generated with the next seed, up to TAKES times; the best take by CER is kept either way.
Kept takes are trimmed to EDGE_S of silence at each end (44.1 kHz mono) and written to DIR/wav/<id>.wav;
DIR/manifest.json lists each id's duration, measured wpm, CER, seed and attempts.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

SEED, TAKES, MAX_CER, CLIP_SAMPLES, WPM_RANGE, EDGE_S = 626, 3, 0.15, 10, (120, 260), 0.04


def words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
       "seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def spell(n: int) -> str:
    """Whisper writes digits; the cues spell numbers out (pre-normalised), so spell the transcript's too."""
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + spell(n % 100))
    if n < 1_000_000:
        return spell(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + spell(n % 1000))
    return str(n)


def letters(text: str) -> str:
    text = re.sub(r"(\d),(\d{3})", r"\1\2", text.lower()).replace("%", " percent")
    text = re.sub(r"\d+", lambda m: " " + spell(int(m.group())) + " ", text)
    return re.sub(r"[^a-z]", "", text)


def cer(ref: str, hyp: str) -> float:
    a, b = letters(ref), letters(hyp)
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[len(b)] / max(1, len(a))


def trim(x: np.ndarray, sr: int, floor_db: float = -45.0) -> np.ndarray:
    """Cut to the first and last 10 ms frame within 45 dB of the peak, then keep EDGE_S of what was around it."""
    n = int(0.01 * sr)
    f = len(x) // n
    rms = np.sqrt((x[: f * n].reshape(f, n).astype(np.float64) ** 2).mean(1) + 1e-20)
    on = np.nonzero(20 * np.log10(rms / rms.max()) > floor_db)[0]
    if not len(on):
        return x
    pad = int(EDGE_S * sr)
    return x[max(0, on[0] * n - pad): min(len(x), (on[-1] + 1) * n + pad)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cues", type=Path, required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--extra", default="{}", help="JSON merged into the manifest (instance, image digest, ...)")
    a = ap.parse_args()
    import soundfile as sf
    import torch
    import torchaudio
    from transformers import pipeline
    sys.path.insert(0, str(a.repo))
    from ming_tts import MingTTS

    lines = sorted((json.loads(s) for s in a.cues.read_text().splitlines() if s.strip()), key=lambda c: c["id"])
    cues = {}                       # one take per sentence: a cue's `take` is its sentence's first cue id
    for c in lines:
        t = cues.setdefault(c.get("take", c["id"]), {"id": c.get("take", c["id"]), "beat": c.get("beat"), "cues": [],
                                                     "spoken": ""})
        t["cues"].append(c["id"])
        t["spoken"] = (t["spoken"] + " " + c["spoken"]).strip()
    cues = list(cues.values())
    (a.out / "wav").mkdir(parents=True, exist_ok=True)
    (a.out / "takes").mkdir(exist_ok=True)
    tts = MingTTS(str(a.model))
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small.en", device=0)
    rows = []
    for c in cues:
        takes = []
        for k in range(TAKES):
            p = a.out / "takes" / f"{c['id']}_{k}.wav"
            t0 = time.time()
            torch.manual_seed(SEED + k)
            tts.synthesize(task="IP", text=c["spoken"], controls={"ip": "US_James"}, use_zero_spk_emb=True,
                           output_path=str(p))
            x, sr = sf.read(str(p), always_2d=True)
            x = x.mean(1)
            if sr != 44100:
                x = torchaudio.functional.resample(torch.as_tensor(x, dtype=torch.float32), sr, 44100).numpy()
                sr = 44100
            y = trim(x, sr)
            heard = asr({"raw": torchaudio.functional.resample(torch.as_tensor(y, dtype=torch.float32), sr, 16000)
                         .numpy(), "sampling_rate": 16000})["text"].strip()
            t = dict(seed=SEED + k, gen_s=round(time.time() - t0, 2), duration_s=round(len(y) / sr, 3),
                     wpm=round(60 * words(c["spoken"]) / (len(y) / sr - 2 * EDGE_S), 1),
                     clipped_samples=int((np.abs(y) >= 0.999).sum()), cer=round(cer(c["spoken"], heard), 4),
                     heard=heard)
            t["ok"] = t["cer"] <= MAX_CER and t["clipped_samples"] <= CLIP_SAMPLES and \
                WPM_RANGE[0] <= t["wpm"] <= WPM_RANGE[1]
            takes.append((t, y))
            print(c["id"], k, json.dumps({q: t[q] for q in ("ok", "cer", "wpm", "duration_s", "clipped_samples")}),
                  flush=True)
            if t["ok"]:
                break
        best, y = min(takes, key=lambda ty: (not ty[0]["ok"], ty[0]["cer"]))
        sf.write(str(a.out / "wav" / f"{c['id']}.wav"), y.astype(np.float32), 44100, subtype="PCM_16")
        rows.append(dict(id=c["id"], cues=c["cues"], beat=c.get("beat"), spoken=c["spoken"], words=words(c["spoken"]),
                         **{q: best[q] for q in ("duration_s", "wpm", "cer", "heard", "seed", "clipped_samples", "ok")},
                         attempts=len(takes), takes=[t for t, _ in takes]))
    bad = [r["id"] for r in rows if not r["ok"]]
    doc = {"what": "demo-film narration, Ming-3.0-tts-0.5b built-in voice US_James (task IP, natural speed)",
           "cues_sha256": __import__("hashlib").sha256(a.cues.read_bytes()).hexdigest(), "sample_rate": 44100,
           "channels": 1, "edge_silence_s": EDGE_S, "checks": {"asr": "openai/whisper-small.en", "max_cer": MAX_CER,
           "max_clipped_samples": CLIP_SAMPLES, "wpm_range": WPM_RANGE, "takes": TAKES},
           "wpm_method": "words / (trimmed duration - both edge pads)", "failed_checks": bad,
           "total_s": round(sum(r["duration_s"] for r in rows), 2), **json.loads(a.extra), "lines": rows}
    (a.out / "manifest.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    print(f"{len(rows)} lines, {len(bad)} failing checks after {TAKES} takes: {bad}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
