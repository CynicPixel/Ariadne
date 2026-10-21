#!/usr/bin/env python3
"""Fetch ONE TUM RGB-D sequence (2.6, D90).

    infra/scripts/fetch_tum.py freiburg1_xyz      # -> .cache/datasets/tum/rgbd_dataset_freiburg1_xyz/{rgb,depth,...}

Sturm et al., "A Benchmark for the Evaluation of RGB-D SLAM Systems", IROS 2012. One .tgz per sequence
(rgb/, depth/, rgb.txt, depth.txt, groundtruth.txt, accelerometer.txt). The archive is streamed to disk, its
sha256 checked against the pin below (first fetch of an unpinned sequence prints the hash to pin), extracted,
and deleted; provenance.json records URL, sha256 and licence.

LICENCE. Creative Commons Attribution 4.0 (CC BY 4.0), per the dataset's download page. Data stays in .cache/
(gitignored), like EuRoC.
Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://cvg.cit.tum.de/rgbd/dataset"
# sha256 of the .tgz, pinned from the first fetch (2026-10-21).
SHA256 = {
    "freiburg1_xyz": "a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98",
    "freiburg1_desk": "e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c",
    "freiburg2_xyz": "a4903027a7fe6bc6573a1ce96a8afa5689f15d74f4a3f0627e91b5799f62e40b",
}
LICENCE = "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)"
UA = "ariadne-dataset-fetch/1.0 (research)"
CHUNK = 1 << 20


def url_for(seq: str) -> str:
    fr = seq.split("_")[0]   # freiburg1
    if fr not in ("freiburg1", "freiburg2", "freiburg3"):
        sys.exit(f"unknown TUM sequence '{seq}' (expected e.g. freiburg1_xyz)")
    return f"{BASE}/{fr}/rgbd_dataset_{seq}.tgz"


def download(url: str, dst: Path, attempts: int = 5) -> str:
    for i in range(attempts):
        try:
            h = hashlib.sha256()
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=120) as r, \
                    open(dst, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                got = 0
                while chunk := r.read(CHUNK):
                    f.write(chunk)
                    h.update(chunk)
                    got += len(chunk)
                    if total and got % (64 * CHUNK) < CHUNK:
                        print(f"  {got >> 20} / {total >> 20} MiB", file=sys.stderr)
            return h.hexdigest()
        except (urllib.error.URLError, TimeoutError) as e:
            if i == attempts - 1:
                raise
            print(f"  {e}; retrying in {5 * 2 ** i}s", file=sys.stderr)
            time.sleep(5 * 2 ** i)
    raise RuntimeError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sequence", help="e.g. freiburg1_xyz, freiburg1_desk, freiburg2_xyz")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2] / ".cache/datasets/tum"))
    a = ap.parse_args()
    root = Path(a.root)
    out = root / f"rgbd_dataset_{a.sequence}"
    if (out / "groundtruth.txt").exists():
        print(f"{out} already present")
        return
    root.mkdir(parents=True, exist_ok=True)
    url = url_for(a.sequence)
    tgz = root / f"{out.name}.tgz.part"
    print(f"fetching {url}")
    digest = download(url, tgz)
    pinned = SHA256.get(a.sequence)
    if pinned and pinned != digest:
        tgz.unlink()
        sys.exit(f"sha256 mismatch for {a.sequence}: got {digest}, pinned {pinned}")
    if not pinned:
        print(f"UNPINNED: add  \"{a.sequence}\": \"{digest}\",  to SHA256", file=sys.stderr)
    with tarfile.open(tgz) as t:
        for m in t.getmembers():   # refuse absolute paths and parent escapes
            if m.name.startswith("/") or ".." in Path(m.name).parts or not (m.isfile() or m.isdir()):
                sys.exit(f"refusing archive member {m.name}")
        t.extractall(root)
    tgz.unlink()
    (out / "provenance.json").write_text(json.dumps(
        {"url": url, "sha256": digest, "pinned": bool(pinned), "licence": LICENCE,
         "fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2) + "\n")
    print(f"ok: {out}")


if __name__ == "__main__":
    main()
