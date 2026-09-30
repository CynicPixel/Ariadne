"""The Python Philox is the SAME generator as cpp/core/random/philox.hpp."""
import numpy as np
import pytest

from dataset_tools.worldgen import rng


# Random123 known-answer vectors (same as tests/numerical/test_philox.cpp).
KATS = [
    ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((0xFFFFFFFF,) * 4, (0xFFFFFFFF, 0xFFFFFFFF), (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0),
     (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
]


@pytest.mark.parametrize("ctr,key,expect", KATS)
def test_scalar_kat(ctr, key, expect):
    assert rng.philox4x32_10(ctr, key) == expect


@pytest.mark.parametrize("ctr,key,expect", KATS)
def test_vectorized_kat(ctr, key, expect):
    out = rng.philox4x32_10_np(*[np.array([c], dtype=np.uint32) for c in ctr], *key)
    assert tuple(int(o[0]) for o in out) == expect


# Words printed by the C++ PhiloxStream (core-dev image, 2026-09-30); 10 words
# cross the first refill, so the block-index placement is checked too.
CPP_WORDS = [3679194993, 1379596678, 4069301830, 3330701958, 142256433, 401767196,
             2952611258, 1604643296, 1230276150, 201578919]
CPP_UNIFORMS = [0.13670795042878603, 0.62318600155797133, 0.79057419749244506]


def test_stream_matches_cpp():
    s = rng.PhiloxStream(0x0123456789ABCDEF, 42, 7)
    assert [s.next_u32() for _ in range(10)] == CPP_WORDS


def test_uniform_matches_cpp_at_domain_edges():
    s = rng.PhiloxStream(20260930, (1 << 64) - 1, rng.MAX_SITE)
    assert [s.uniform() for _ in range(3)] == CPP_UNIFORMS


def test_site_hash_matches_cpp():
    assert rng.site_from_hash64(0) == 0
    assert rng.site_from_hash64(0xDEADBEEFCAFEF00D) == 82335603970499


def test_site_domain_is_enforced():
    with pytest.raises(ValueError):
        rng.PhiloxStream(1, 0, rng.MAX_SITE + 1)


def test_exhaustion_raises_instead_of_wrapping():
    s = rng.PhiloxStream(1, 2, 3)
    for _ in range(4 * (rng.MAX_BLOCK + 1)):
        s.next_u32()
    with pytest.raises(rng.SiteExhausted):
        s.next_u32()


def test_integer_is_unbiased_and_in_range():
    s = rng.PhiloxStream(5, 0, 1)
    draws = [s.integer(0, 2) for _ in range(30000)]
    assert set(draws) == {0, 1, 2}
    counts = np.bincount(draws)
    # 3 sigma of a binomial(30000, 1/3) is ~245.
    assert np.all(np.abs(counts - 10000) < 300)
    with pytest.raises(ValueError):
        s.integer(3, 2)


def test_shuffle_is_a_permutation_and_deterministic():
    a = rng.PhiloxStream(9, 1, 1).shuffled(range(50))
    b = rng.PhiloxStream(9, 1, 1).shuffled(range(50))
    assert a == b and sorted(a) == list(range(50)) and a != list(range(50))


def test_site_of_is_order_sensitive_and_in_domain():
    sites = {rng.site_of(i, j) for i in range(50) for j in range(50)}
    assert len(sites) == 2500
    assert rng.site_of(1, 2) != rng.site_of(2, 1)
    assert all(0 <= s <= rng.MAX_SITE for s in sites)
