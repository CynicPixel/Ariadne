#!/usr/bin/env python3
"""Fetch ONE EuRoC MAV sequence (ASL format) without downloading its whole archive.

    infra/scripts/fetch_euroc.py MH_01_easy        # -> .cache/datasets/euroc/MH_01_easy/mav0/...

WHY THIS SHAPE. EuRoC (Burri et al., IJRR 2016; DOI 10.3929/ethz-b-000690084)
is hosted on the ETH Research Collection as one zip per environment:
machine_hall.zip is 12.7 GB and holds all five MH sequences, each as a ROS
bag plus an ASL-format zip. We need one ASL zip. A zip keeps its index (the
central directory) at the END of the file, and the server honours HTTP Range,
so:
  1. read the central directory with a few small ranged requests;
  2. download exactly the member's bytes in ONE streaming ranged GET (a naive
     zipfile-over-HTTP would issue a request per read, ~20k of them, and ETH
     rate-limits scrapers);
  3. verify the member's CRC-32 from the central directory, then record the
     sha256 of the result as provenance.

LICENCE. "In Copyright - Non-Commercial Use Permitted" (rightsstatements.org
InC-NC 1.0), read from the record's dc.rights metadata on 2026-09-29. Data stays
in .cache/ (gitignored, dockerignored, and excluded from spinup.sh's snapshot);
EuRoC pixels must not appear in any submitted material (DECISIONS D35).

Standard library only: nothing to install.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import sys
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path

API = "https://www.research-collection.ethz.ch/server/api/core/bitstreams"
# Bitstream UUIDs of the ORIGINAL files, read from the DSpace REST API
# (/server/api/pid/find?id=hdl:20.500.11850/690084 -> bundles -> bitstreams).
ARCHIVES = {
    "machine_hall": "7b2419c1-62b5-4714-b7f8-485e5fe3e5fe",
    "vicon_room1": "02ecda9a-298f-498b-970c-b7c44334d880",
    "vicon_room2": "ea12bc01-3677-4b4c-853d-87c7870b8c44",
}
LICENCE = "In Copyright - Non-Commercial Use Permitted (http://rightsstatements.org/page/InC-NC/1.0/)"
UA = "ariadne-dataset-fetch/1.0 (research; single-member ranged download)"
CHUNK = 1 << 20


def archive_for(sequence: str) -> str:
    if sequence.startswith("MH_"):
        return "machine_hall"
    if sequence.startswith("V1_"):
        return "vicon_room1"
    if sequence.startswith("V2_"):
        return "vicon_room2"
    sys.exit(f"unknown EuRoC sequence '{sequence}'")


def request(url: str, start: int, end: int | None = None, attempts: int = 7):
    """A ranged GET with backoff on 429/5xx. Returns the open response."""
    rng = f"bytes={start}-" + ("" if end is None else str(end))
    for i in range(attempts):
        try:
            resp = urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": UA, "Range": rng}), timeout=120)
            if resp.status != 206:
                sys.exit(f"server ignored Range (HTTP {resp.status}); refusing to download the whole archive")
            return resp
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or i == attempts - 1:
                raise
            wait = int(e.headers.get("Retry-After") or 0) or 5 * 2 ** i
            print(f"  HTTP {e.code}; retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError) as e:
            if i == attempts - 1:
                raise
            print(f"  {e}; retrying in {5 * 2 ** i}s", file=sys.stderr)
            time.sleep(5 * 2 ** i)
    raise RuntimeError("unreachable")


class RangedFile(io.RawIOBase):
    """Seekable read-only view of a remote file, used ONLY to parse the zip
    index. Reads are block-cached so zipfile's many small reads cost a handful
    of requests."""

    def __init__(self, url: str):
        self.url, self.pos, self.cache = url, 0, {}
        head = request(url, 0, 0)
        self.size = int(head.headers["Content-Range"].split("/")[1])
        head.read()
        self.requests = 1

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _block(self, b: int) -> bytes:
        if b not in self.cache:
            lo = b * CHUNK
            hi = min(self.size, lo + CHUNK) - 1
            self.cache[b] = request(self.url, lo, hi).read()
            self.requests += 1
        return self.cache[b]

    def readinto(self, buf) -> int:
        n = min(len(buf), self.size - self.pos)
        out, got = memoryview(buf), 0
        while got < n:
            b, off = divmod(self.pos + got, CHUNK)
            data = self._block(b)[off: off + n - got]
            out[got: got + len(data)] = data
            got += len(data)
        self.pos += got
        return got


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence", nargs="?", default="MH_01_easy")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2] / ".cache/datasets/euroc"))
    args = ap.parse_args()

    seq, archive = args.sequence, archive_for(args.sequence)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    inner = root / f"{seq}.zip"
    prov_path = root / f"{seq}.provenance.json"
    url = f"{API}/{ARCHIVES[archive]}/content"

    if prov_path.exists() and inner.exists():
        prov = json.loads(prov_path.read_text())
        if sha256(inner) == prov["sha256"]:
            print(f"{seq}: already present, sha256 verified ({prov['sha256'][:16]}...)")
            return extract(inner, root / seq)
        print(f"{seq}: sha256 mismatch against provenance; re-fetching")

    # 1. The index.
    remote = RangedFile(url)
    with zipfile.ZipFile(io.BufferedReader(remote, buffer_size=CHUNK)) as zf:
        member = f"{archive}/{seq}/{seq}.zip"
        info = zf.getinfo(member)
    print(f"{member}: {info.compress_size / 1e9:.2f} GB "
          f"({'stored' if info.compress_type == zipfile.ZIP_STORED else 'deflated'}), "
          f"archive {remote.size / 1e9:.1f} GB, index read in {remote.requests} requests")
    if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        sys.exit(f"unsupported compression method {info.compress_type}")

    # The local header repeats the name and may carry a different extra field
    # than the central directory, so its real length is read, not assumed.
    lh = request(url, info.header_offset, info.header_offset + 29).read()
    sig, *_rest = struct.unpack("<IHHHHHIIIHH", lh)
    if sig != 0x04034B50:
        sys.exit("bad local file header signature")
    name_len, extra_len = struct.unpack("<HH", lh[26:30])
    data_start = info.header_offset + 30 + name_len + extra_len

    # 2. One streaming ranged GET for the member, resumable.
    part = inner.with_suffix(".zip.part")
    have = part.stat().st_size if part.exists() else 0
    total = info.compress_size
    if info.compress_type == zipfile.ZIP_DEFLATED and have:
        have = 0   # a deflate stream cannot resume mid-way without its state
    if have < total:
        resp = request(url, data_start + have, data_start + total - 1)
        mode = "ab" if have else "wb"
        with part.open(mode) as out:
            got, t0 = have, time.time()
            inflater = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
            for block in iter(lambda: resp.read(CHUNK), b""):
                out.write(inflater.decompress(block) if inflater else block)
                got += len(block)
                if got % (256 * CHUNK) < CHUNK:
                    rate = (got - have) / max(1e-6, time.time() - t0) / 1e6
                    print(f"  {got / 1e9:.2f} / {total / 1e9:.2f} GB  ({rate:.1f} MB/s)", flush=True)
            if inflater:
                out.write(inflater.flush())

    # 3. Integrity: CRC-32 from the central directory, then sha256 provenance.
    crc = 0
    with part.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            crc = zlib.crc32(block, crc)
    if crc != info.CRC:
        part.unlink()
        sys.exit(f"CRC mismatch ({crc:08x} != {info.CRC:08x}); partial file removed")
    part.rename(inner)
    digest = sha256(inner)
    prov_path.write_text(json.dumps({
        "dataset": "EuRoC MAV (Burri et al., IJRR 2016)",
        "doi": "10.3929/ethz-b-000690084",
        "sequence": seq,
        "source_url": url,
        "archive_member": member,
        "crc32": f"{info.CRC:08x}",
        "sha256": digest,
        "bytes": inner.stat().st_size,
        "licence": LICENCE,
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2) + "\n")
    print(f"{seq}: CRC ok, sha256 {digest}")
    return extract(inner, root / seq)


def extract(inner: Path, dest: Path) -> int:
    marker = dest / ".extracted"
    if marker.exists():
        print(f"{dest}: already extracted")
        return 0
    with zipfile.ZipFile(inner) as zf:
        members = [m for m in zf.namelist()
                   if not m.startswith("__MACOSX/") and not Path(m).name.startswith("._")
                   and Path(m).name != ".DS_Store"]
        zf.extractall(dest, members)   # zipfile.testzip-equivalent CRC checks happen on read
    marker.write_text("ok\n")
    print(f"{dest}: extracted {len(members)} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
