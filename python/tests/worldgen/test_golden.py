"""Frozen output of the generator. If this fails, the generator's OUTPUT changed:
either the change was unintended (fix it), or it is intended, in which case bump
dataset_tools.worldgen.GENERATOR_VERSION and replace GOLDEN with the new digests.
Asserted inside the pinned ariadne-sim image (infra/scripts/worldgen.sh test): the
route angles use libm atan2/sqrt, whose last bits may differ on other platforms."""
import hashlib
import os

import numpy as np
import pytest

from dataset_tools.worldgen import GENERATOR_VERSION
from dataset_tools.worldgen.build import write_scene
from dataset_tools.worldgen.params import load_preset
from dataset_tools.worldgen.scene import generate

ROOT = __file__.rsplit("/python/", 1)[0]

GOLDEN = {  # (preset, seed) -> sha256 over the scene directory (see scene_dir_digest)
    ("ci_tiny", 0): "8c7007565488a5fb69bf9d69e23cfe6b5ae6dc45e4856fa9189e713c37ed47b7",
    ("maze_short", 7): "6fffa1f6c6bf2f91b938c78dbf4b994b9971bf814fc292a0f97f1576f7165945",
}


def scene_dir_digest(d) -> str:
    h = hashlib.sha256()
    for rel in ("scene.json", "model.sdf", "world.sdf", "model_inline.xml", "esdf_f32.bin"):
        h.update(rel.encode())
        h.update(open(os.path.join(d, rel), "rb").read())
    for sub in ("routes", "meshes"):
        for name in sorted(os.listdir(os.path.join(d, sub))):
            h.update(f"{sub}/{name}".encode())
            h.update(open(os.path.join(d, sub, name), "rb").read())
    gt = np.load(os.path.join(d, "scene_gt.npz"))
    for key in sorted(gt.files):
        h.update(key.encode())
        h.update(repr((gt[key].shape, str(gt[key].dtype))).encode())
        h.update(np.ascontiguousarray(gt[key]).tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_golden_scene(key, tmp_path):
    assert GENERATOR_VERSION == 1
    preset, seed = key
    s = generate(load_preset(f"{ROOT}/configs/worldgen/{preset}.yaml"), seed)
    write_scene(s, str(tmp_path))
    got = scene_dir_digest(str(tmp_path))
    if GOLDEN[key] is None:
        pytest.fail(f"no golden recorded for {key}; measured {got}")
    assert got == GOLDEN[key]
