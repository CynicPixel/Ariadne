"""Media helpers for the report figures and the devpost folder (run inside manimcommunity/manim:v0.22.0, which has
PyAV + Pillow; the host has no ffmpeg).

  python3 media_tools.py sheet  IN.mp4 OUT.png [N=16]                 contact sheet, timestamps burned in
  python3 media_tools.py frame  IN.mp4 OUT.png T_S                    one still at T_S seconds
  python3 media_tools.py gif    IN.mp4 OUT.gif T0 T1 [--fps 10] [--width 640] [--label TEXT]
  python3 media_tools.py frames DIR_GLOB OUT.gif [--fps 15] [--width 640] [--step 1] [--label TEXT]

GIFs use one adaptive palette per GIF (computed from a sample of the frames) so colours stay stable across frames.
"""
import argparse
import glob

import av
from PIL import Image, ImageDraw, ImageFont


def decode(path, t0=0.0, t1=None):
    """Yield (t_s, PIL.Image) for every frame with t0 <= t < t1."""
    with av.open(path) as c:
        s = c.streams.video[0]
        for f in c.decode(s):
            t = float(f.pts * s.time_base)
            if t < t0:
                continue
            if t1 is not None and t >= t1:
                break
            yield t, f.to_image()


def duration(path):
    with av.open(path) as c:
        s = c.streams.video[0]
        return float(s.duration * s.time_base) if s.duration else float(c.duration) / av.time_base


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def label(img, text):
    """Burn a provenance label (e.g. 'recorded frames · illustrative tracks') into the lower-left corner."""
    if not text:
        return img
    d = ImageDraw.Draw(img)
    f = font(max(11, img.width // 55))
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=f)
    pad = 4
    y = img.height - (y1 - y0) - 3 * pad
    d.rectangle([pad, y - pad, pad + (x1 - x0) + 2 * pad, y + (y1 - y0) + 2 * pad], fill=(0, 0, 0))
    d.text((2 * pad, y), text, fill=(255, 255, 255), font=f)
    return img


def resize(img, width):
    if not width or img.width <= width:
        return img
    return img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)


def write_gif(frames, out, fps, text):
    frames = [label(f.convert("RGB"), text) for f in frames]
    # One shared palette from a strip of sampled frames: no per-frame colour flicker.
    sample = frames[:: max(1, len(frames) // 12)]
    strip = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for i, f in enumerate(sample):
        strip.paste(f, (0, i * f.height))
    pal = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(out, save_all=True, append_images=q[1:], duration=round(1000 / fps), loop=0, optimize=True)
    print(f"{out}: {len(q)} frames @ {fps} fps, {frames[0].width}x{frames[0].height}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("sheet"); a.add_argument("inp"); a.add_argument("out"); a.add_argument("n", nargs="?", type=int, default=16)
    a = sub.add_parser("frame"); a.add_argument("inp"); a.add_argument("out"); a.add_argument("t", type=float)
    for name in ("gif", "frames"):
        a = sub.add_parser(name)
        a.add_argument("inp"); a.add_argument("out")
        if name == "gif":
            a.add_argument("t0", type=float); a.add_argument("t1", type=float)
        else:
            a.add_argument("--step", type=int, default=1)
        a.add_argument("--fps", type=float, default=10); a.add_argument("--width", type=int, default=640)
        a.add_argument("--label", default="")
    a = ap.parse_args()

    if a.cmd == "sheet":
        d = duration(a.inp)
        times = [d * (i + 0.5) / a.n for i in range(a.n)]
        picks, k = [], 0
        for t, img in decode(a.inp):
            if k < len(times) and t >= times[k]:
                img = resize(img, 320)
                ImageDraw.Draw(img).text((4, 4), f"{t:5.1f}s", fill=(255, 255, 0), font=font(16))
                picks.append(img); k += 1
        cols = 4
        rows = (len(picks) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * picks[0].width, rows * picks[0].height))
        for i, p in enumerate(picks):
            sheet.paste(p, ((i % cols) * p.width, (i // cols) * p.height))
        sheet.save(a.out)
        print(f"{a.out}: {len(picks)} tiles, duration {d:.1f}s")
    elif a.cmd == "frame":
        for t, img in decode(a.inp, a.t):
            img.save(a.out); print(f"{a.out}: t={t:.2f}s {img.width}x{img.height}"); break
    elif a.cmd == "gif":
        src_fps = None
        frames, last = [], -1e9
        for t, img in decode(a.inp, a.t0, a.t1):
            if t - last >= 1.0 / a.fps - 1e-6:
                frames.append(resize(img, a.width)); last = t
        write_gif(frames, a.out, a.fps, a.label)
    else:
        paths = sorted(glob.glob(a.inp))[:: a.step]
        write_gif([resize(Image.open(p), a.width) for p in paths], a.out, a.fps, a.label)


if __name__ == "__main__":
    main()
