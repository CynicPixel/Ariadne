"""Deterministic randomness for worldgen: a bit-exact port of cpp/core/random/philox.hpp.

Why a port instead of numpy.random:
  NumPy guarantees its BitGenerator *streams*, but not the outputs of
  Generator.integers / choice / shuffle / normal across versions. A world must be
  recoverable from its seed forever, so every mapping from random bits to a value
  is owned here, and the bits themselves come from the same Philox4x32-10 that the
  C++ core uses (Salmon et al., SC'11). The same (seed, frame, site) names the same
  words in both languages; tests/ check the Random123 known-answer vectors and
  words produced by the C++ PhiloxStream.

Two faces:
  PhiloxStream       scalar, stateful-per-site stream (layout decisions: a few
                     thousand draws), mirrors the C++ class exactly, including
                     the counter layout and the exhaustion guard.
  philox4x32_10_np   vectorized over uint32 arrays (texture noise: one hash per
                     lattice point). Pure integer arithmetic, so bit-identical on
                     every platform and numpy version.
"""
from __future__ import annotations

import numpy as np

M0 = 0xD2511F53
M1 = 0xCD9E8D57
W0 = 0x9E3779B9
W1 = 0xBB67AE85
MASK32 = 0xFFFFFFFF

MAX_SITE = (1 << 48) - 1
MAX_BLOCK = 0xFFFF


def philox4x32_10(ctr: tuple[int, int, int, int], key: tuple[int, int]) -> tuple[int, int, int, int]:
    """Philox4x32 with 10 rounds on Python ints (exact 32-bit arithmetic)."""
    c0, c1, c2, c3 = (v & MASK32 for v in ctr)
    k0, k1 = key[0] & MASK32, key[1] & MASK32
    for rnd in range(10):
        if rnd > 0:
            k0 = (k0 + W0) & MASK32
            k1 = (k1 + W1) & MASK32
        p0 = M0 * c0
        p1 = M1 * c2
        hi0, lo0 = p0 >> 32, p0 & MASK32
        hi1, lo1 = p1 >> 32, p1 & MASK32
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
    return c0, c1, c2, c3


class SiteExhausted(RuntimeError):
    """Raised instead of wrapping the block index (which would reuse numbers)."""


class PhiloxStream:
    """Mirror of ariadne::core::PhiloxStream: the stream named by (seed, frame_id, site).

    Counter layout (identical to C++): v0,v1 = frame_id; v2 = site low 32 bits;
    v3 low 16 = site bits 32..47; v3 high 16 = refill block index.
    """

    def __init__(self, seed: int, frame_id: int, site: int):
        if not 0 <= seed < 1 << 64 or not 0 <= frame_id < 1 << 64:
            raise ValueError("seed and frame_id must be unsigned 64-bit")
        if not 0 <= site <= MAX_SITE:
            # C++ masks in release builds; here the out-of-domain call is a bug.
            raise ValueError("site exceeds the 48-bit stream domain; use site_from_hash64")
        self._key = (seed & MASK32, seed >> 32)
        self._ctr = (frame_id & MASK32, frame_id >> 32, site & MASK32, (site >> 32) & 0xFFFF)
        self._block = 0
        self._refill()

    def _refill(self) -> None:
        c0, c1, c2, c3 = self._ctr
        self._buf = philox4x32_10((c0, c1, c2, c3 | (self._block << 16)), self._key)
        self._used = 0

    def next_u32(self) -> int:
        if self._used == 4:
            self._block += 1
            if self._block > MAX_BLOCK:
                raise SiteExhausted("PhiloxStream: site exhausted after 262144 words; "
                                    "draw from a new site rather than wrapping")
            self._refill()
        w = self._buf[self._used]
        self._used += 1
        return w

    def next_u64(self) -> int:
        hi = self.next_u32()
        lo = self.next_u32()
        return (hi << 32) | lo

    def uniform(self, lo: float = 0.0, hi: float = 1.0) -> float:
        """53-bit uniform in [lo, hi); the [0,1) value matches C++ NextUniform bit for bit."""
        u = (self.next_u64() >> 11) * (1.0 / 9007199254740992.0)
        return lo + (hi - lo) * u

    def integer(self, lo: int, hi: int) -> int:
        """Unbiased integer in [lo, hi] (inclusive) by rejection on 64-bit words.

        Rejection (not modulo) because modulo of 2^64 by a non-power-of-two span
        is biased; the rejection zone is the top partial copy of the span.
        """
        if hi < lo:
            raise ValueError("empty integer range")
        span = hi - lo + 1
        limit = (1 << 64) - ((1 << 64) % span)
        while True:
            w = self.next_u64()
            if w < limit:
                return lo + w % span

    def bernoulli(self, p: float) -> bool:
        return self.uniform() < p

    def choice(self, seq):
        if not seq:
            raise ValueError("choice from empty sequence")
        return seq[self.integer(0, len(seq) - 1)]

    def shuffled(self, seq) -> list:
        """Fisher-Yates, from the end: element i swaps with a uniform j in [0, i]."""
        out = list(seq)
        for i in range(len(out) - 1, 0, -1):
            j = self.integer(0, i)
            out[i], out[j] = out[j], out[i]
        return out


MASK64 = 0xFFFFFFFFFFFFFFFF


def _mix64(z: int) -> int:
    """splitmix64 finalizer (a bijection on 64-bit words)."""
    z &= MASK64
    z ^= z >> 30
    z = (z * 0xBF58476D1CE4E5B9) & MASK64
    z ^= z >> 27
    z = (z * 0x94D049BB133111EB) & MASK64
    z ^= z >> 31
    return z


def site_from_hash64(value: int) -> int:
    """Mirror of C++ SiteFromHash64: splitmix64 finalizer, then the 48-bit site mask."""
    return _mix64(value) & MAX_SITE


def site_of(*parts: int) -> int:
    """A 48-bit site naming a call site by a tuple of non-negative ints, e.g. (STAGE, index).

    Order-sensitive fold: h <- mix64(h * FNV_PRIME + part + 1). Because mix64 is a
    bijection, distinct prefixes stay distinct before the final 48-bit mask; after
    it, collisions are birthday-bounded (~2^-48 per pair), never systematic.
    """
    h = 0
    for p in parts:
        if p < 0 or p > MASK64:
            raise ValueError("site parts must be unsigned 64-bit")
        h = _mix64((h * 0x100000001B3 + p + 1) & MASK64)
    return h & MAX_SITE


# --- vectorized ------------------------------------------------------------------

def _mulhilo(a: int, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.uint64(a) * b.astype(np.uint64)
    return (p >> np.uint64(32)).astype(np.uint32), (p & np.uint64(MASK32)).astype(np.uint32)


def philox4x32_10_np(c0, c1, c2, c3, k0: int, k1: int):
    """Philox4x32-10 over broadcastable uint32 arrays; returns four uint32 arrays.

    uint32 * uint32 is widened to uint64 first, so the product is exact; the
    key bumps wrap in Python ints and are masked, exactly as the C++ uint32 adds.
    """
    c0, c1, c2, c3 = np.broadcast_arrays(*(np.asarray(c, dtype=np.uint32) for c in (c0, c1, c2, c3)))
    c0, c1, c2, c3 = (x.copy() for x in (c0, c1, c2, c3))
    k0 &= MASK32
    k1 &= MASK32
    for rnd in range(10):
        if rnd > 0:
            k0 = (k0 + W0) & MASK32
            k1 = (k1 + W1) & MASK32
        hi0, lo0 = _mulhilo(M0, c0)
        hi1, lo1 = _mulhilo(M1, c2)
        c0, c1, c2, c3 = hi1 ^ c1 ^ np.uint32(k0), lo1, hi0 ^ c3 ^ np.uint32(k1), lo0
    return c0, c1, c2, c3


def unit_from_u32(w: np.ndarray) -> np.ndarray:
    """uint32 -> float64 in [0, 1) by an exact multiply by 2^-32 (as C++ UniformFromBits32)."""
    return w.astype(np.float64) * (1.0 / 4294967296.0)
