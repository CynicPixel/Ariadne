import json

import numpy as np
import pytest

from dataset_tools.worldgen import geometry as geo
from dataset_tools.worldgen.build import surfaces, write_scene
from dataset_tools.worldgen.params import parse_preset
from dataset_tools.worldgen.png import pixel_digest, read_png
from dataset_tools.worldgen.raster import Grid
from dataset_tools.worldgen.scene import generate

PRESET = parse_preset({"name": "test", "lattice": {"nx": 3, "ny": 3},
                       "target": {"path_length_m": [5.0, 30.0]}})


def boundary_edge_count(free):
    return int((free[1:] != free[:-1]).sum() + (free[:, 1:] != free[:, :-1]).sum())


def test_wall_runs_cover_every_free_solid_edge_once():
    rng = np.random.default_rng(0)
    free = np.zeros((40, 50), bool)
    for _ in range(12):
        r, c = rng.integers(1, 30), rng.integers(1, 40)
        free[r:r + rng.integers(2, 9), c:c + rng.integers(2, 9)] = True
    grid = Grid(0, 0, 50, 40, 0.05)
    runs = geo.wall_runs(free, grid)
    assert sum(r.b - r.a for r in runs) == boundary_edge_count(free)


def test_every_wall_quad_faces_into_free_space():
    s = generate(PRESET, 1)
    g = s.grid
    for surf in surfaces(s):
        if surf.kind != geo.K_WALL:
            continue
        x, y = surf.probe_xy_units
        r, c = int(np.floor(y)) - g.oy, int(np.floor(x)) - g.ox
        assert s.structure_free[r, c], surf
        n = surf.quad.normal()
        # the probe point is on the normal side of the face
        p0 = np.array(surf.quad.p0)
        probe_m = np.array([x * g.g, y * g.g, 0.0])
        assert np.dot(probe_m - p0, n) > 0


def test_floor_rectangles_partition_the_free_mask():
    rng = np.random.default_rng(1)
    free = rng.random((30, 40)) > 0.3
    rects = geo.free_rectangles(free, max_side=7)
    cover = np.zeros(free.shape, int)
    for r0, c0, r1, c1 in rects:
        assert r1 - r0 <= 7 and c1 - c0 <= 7
        cover[r0:r1, c0:c1] += 1
    assert np.array_equal(cover, free.astype(int))


def test_prop_faces_point_outward():
    s = generate(PRESET, 2)
    for surf in surfaces(s):
        if surf.kind != geo.K_PROP:
            continue
        cx, cy = (np.array(surf.probe_xy_units) * s.grid.g).tolist()
        centre = np.array([cx, cy, 0.0])
        q = surf.quad
        mid = np.array(q.p0) + 0.5 * np.array(q.eu) + 0.5 * np.array(q.ev)
        assert np.dot(mid - centre, q.normal()) > 0


def test_build_is_reproducible_and_textures_match_manifest(tmp_path):
    s = generate(PRESET, 3)
    a, b = tmp_path / "a", tmp_path / "b"
    ma = write_scene(s, str(a))
    mb = write_scene(generate(PRESET, 3), str(b))
    assert ma == mb
    for name in ("model.sdf", "world.sdf", "model_inline.xml", "scene.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    objs = sorted(p.name for p in (a / "meshes").iterdir())
    assert objs == sorted(p.name for p in (b / "meshes").iterdir())
    for obj in objs:
        assert (a / "meshes" / obj).read_bytes() == (b / "meshes" / obj).read_bytes()
    rows = {r["name"]: r for r in ma["surfaces"]}
    for name in list(rows)[:10]:
        img = read_png(a / "textures" / f"{name}.png")
        assert pixel_digest(img) == rows[name]["pixel_sha256"]
        assert list(img.shape[:2]) == rows[name]["size_px"]
    gt = np.load(a / "scene_gt.npz")
    assert gt["free"].sum() < gt["structure_free"].sum() or not s.props
    assert gt["trav_responder"][tuple(gt["target_cell"])]
    route = gt["route_responder_start0"]
    assert tuple(route[0]) == tuple(gt["start_cells"][0]) and tuple(route[-1]) == tuple(gt["target_cell"])
    manifest = json.loads((a / "scene.json").read_text())
    assert manifest["scene_id"] == s.scene_id


def test_texture_key_is_a_bijection_per_attempt():
    s = generate(PRESET, 4)
    from dataset_tools.worldgen.build import texture_key
    sids = [x.sid for x in surfaces(s)]
    keys = {texture_key(s, sid) for sid in sids}
    assert len(keys) == len(sids)
