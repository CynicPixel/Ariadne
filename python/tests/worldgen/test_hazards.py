"""D112 structural hazards (design traversability-risk.md#risk-zones §4): opt-in without moving any existing digest, GT traversability
unchanged (a hazard threatens a route, never blocks it), placement constraints, geometry, GT risk layers."""
from pathlib import Path

import numpy as np
import pytest

from dataset_tools.worldgen import build, scene as scene_mod
from dataset_tools.worldgen.geometry import K_HAZARD, hazard_surfaces
from dataset_tools.worldgen.params import load_preset

CONFIGS = Path(__file__).resolve().parents[3] / "configs" / "worldgen"

# Preset.digest() of every pre-D112 preset, recorded before the hazards section existed. Scene ids, recordings and
# the Phase 1 gate key on these; an opt-in section must not move them.
PRE_D112_DIGESTS = {
    "ci_tiny": "c79dd57418377fb054176f89de0ba0dc820b9fc59d4ce454345e7f6c7840ac58",
    "maze_short": "d0cbe911fd3647445037ddecaf905ed3e86ec92c011791003ca0519fd365ac58",
    "maze_medium": "8c2965eef10dfb2b3ca0266670460a1c81d2cba418b78c93f8ce1c92135fdcbb",
    "maze_long": "b59c7f82c106ab564e5877aca145a2dc95c5a5a11f5f079d3d0a8e6a8c53d4d6",
}


@pytest.mark.parametrize("name", sorted(PRE_D112_DIGESTS))
def test_existing_preset_digests_are_unchanged(name):
    p = load_preset(CONFIGS / f"{name}.yaml")
    assert p.digest() == PRE_D112_DIGESTS[name]
    assert "hazards" not in p.to_canonical()


@pytest.fixture(scope="module")
def twins():
    plain = scene_mod.generate(load_preset(CONFIGS / "ci_tiny.yaml"), 0)
    hz = scene_mod.generate(load_preset(CONFIGS / "ci_tiny_hazard.yaml"), 0)
    return plain, hz


def test_hazards_change_no_ground_truth_traversability(twins):
    plain, hz = twins
    assert not plain.hazards and hz.hazards
    assert hz.attempt == plain.attempt   # same layout: streams are keyed by (seed, attempt, site), not the preset
    for k in ("free", "structure_free", "esdf_m", "trav_agent", "trav_responder"):
        assert np.array_equal(getattr(plain, k), getattr(hz, k)), k
    assert hz.target_cell == plain.target_cell and hz.start_cells == plain.start_cells
    assert [s.segments for s in hz.scripts] == [s.segments for s in plain.scripts]


def test_placement_constraints(twins):
    _, s = twins
    P, g = s.preset, s.grid.g
    hz = P.hazards
    assert hz.cantilevers[0] <= sum(h.kind == "cantilever" for h in s.hazards) <= hz.cantilevers[1]
    for h in s.hazards:
        room = s.layout.rooms[h.room]
        assert room.x0 <= h.rect.x0 < h.rect.x1 <= room.x1 and room.y0 <= h.rect.y0 < h.rect.y1 <= room.y1
        assert h.z0 >= 1.9 and h.z1 <= s.wall_height_m - 0.1 + 1e-9, (h, s.wall_height_m)
        if h.kind == "cantilever":   # held by its wall: the footprint touches that side of the room
            assert {"W": h.rect.x0 == room.x0, "E": h.rect.x1 == room.x1,
                    "S": h.rect.y0 == room.y0, "N": h.rect.y1 == room.y1}[h.side]
    gap = max(1, int(round(hz.gap_m / g)))
    for i, a in enumerate(s.hazards):
        for b in s.hazards[i + 1:]:
            assert not (a.rect.x0 < b.rect.x1 + gap and b.rect.x0 < a.rect.x1 + gap
                        and a.rect.y0 < b.rect.y1 + gap and b.rect.y0 < a.rect.y1 + gap)


def test_geometry_has_six_outward_faces_with_a_downward_bottom(twins):
    _, s = twins
    surf = hazard_surfaces(s.hazards, s.grid.g, s.preset.appearance.wall_px_per_m)
    assert len(surf) == 6 * len(s.hazards) and all(x.kind == K_HAZARD for x in surf)
    for hi, h in enumerate(s.hazards):
        faces = surf[6 * hi:6 * hi + 6]
        centre = np.array([(h.rect.x0 + h.rect.x1) / 2 * s.grid.g, (h.rect.y0 + h.rect.y1) / 2 * s.grid.g,
                           (h.z0 + h.z1) / 2])
        for f in faces:
            o, u, v = (np.array(t, float) for t in (f.quad.p0, f.quad.eu, f.quad.ev))
            assert np.dot(f.quad.normal(), o + (u + v) / 2 - centre) > 0, "faces point outward"
        bottom = faces[5].quad
        assert bottom.normal()[2] < 0 and abs(bottom.p0[2] - h.z0) < 1e-12


def test_gt_risk_layers(twins):
    plain, s = twins
    gt, gt0 = build.ground_truth(s), build.ground_truth(plain)
    assert "risk_zone" not in gt0 and "hazard_footprint" not in gt0   # no-hazard scenes keep their exact npz
    foot, zone = gt["hazard_footprint"], gt["risk_zone"]
    assert foot.any() and (zone >= foot).all() and zone.sum() > foot.sum()
    # the margin, by hand: a cell 0.45 m beyond a footprint edge is in the zone, one at 0.6 m is not
    h = s.hazards[0]
    r0, c0 = s.grid.index(h.rect.x0, h.rect.y0)
    r1, _ = s.grid.index(h.rect.x1, h.rect.y1)
    rm = (r0 + r1) // 2
    for dc, want in ((-9, True), (-12, False)):   # cell centre (c0 + dc + 0.5): 8.5 or 11.5 cells = 0.425 / 0.575 m
        if 0 <= c0 + dc:
            assert zone[rm, c0 + dc] == want or (want is False and foot[rm, c0 + dc])
