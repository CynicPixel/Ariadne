import hashlib

import numpy as np
import pytest

from dataset_tools.worldgen.params import parse_preset
from dataset_tools.worldgen.scene import GenerationFailed, generate

PRESET = parse_preset({"name": "test"})


def scene_digest(s) -> str:
    h = hashlib.sha256()
    lengths = sorted((k, round(d["responder_m"], 9), round(d["agent_m"], 9)) for k, d in s.lengths.items())
    h.update(repr((s.attempt, s.target_room, s.target_cell, s.start_cells, lengths, s.grid)).encode())
    h.update(np.packbits(s.free).tobytes())
    h.update(np.round(s.esdf_m, 9).tobytes())
    return h.hexdigest()


def test_generate_is_deterministic():
    a, b = generate(PRESET, 5), generate(PRESET, 5)
    assert scene_digest(a) == scene_digest(b)
    assert a.scene_id == b.scene_id
    assert scene_digest(generate(PRESET, 6)) != scene_digest(a)


@pytest.mark.parametrize("seed", range(4))
def test_accepted_scene_satisfies_the_preset(seed):
    s = generate(PRESET, seed)
    lo, hi = PRESET.target.path_length_m
    assert lo <= s.lengths[0]["responder_m"] <= hi
    for k, d in s.lengths.items():
        assert np.isfinite(d["responder_m"])
        assert d["agent_m"] <= d["responder_m"] + 1e-9    # the agent can use every responder route
    assert s.trav_responder[s.target_cell]
    assert all(s.trav_responder[c] for c in s.start_cells)
    assert not (s.trav_responder & ~s.trav_agent).any()   # responder-traversable implies agent


def test_unsatisfiable_band_fails_loudly_with_reasons():
    p = parse_preset({"name": "x", "target": {"path_length_m": [500.0, 600.0]},
                      "search": {"max_attempts": 5}})
    with pytest.raises(GenerationFailed) as ei:
        generate(p, 0)
    assert sum(ei.value.reasons.values()) == 5
