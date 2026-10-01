"""The gz warning gate (SMOKE §4n): every gz [Wrn]/[Err] fails a recording unless
it is on the justified allowlist. Lines below are verbatim gz output (ANSI kept)."""
import os
import sys

ROOT = __file__.rsplit("/python/", 1)[0]
sys.path.insert(0, os.path.join(ROOT, "infra", "scripts"))

import sim_record as S  # noqa: E402

Y = "\x1b[1;33m"
Z = "\x1b[0m"
OBJ = (f"{Y}[Wrn] [OBJLoader.cc:202] {Z}{Y}Missing material for shape[{Z}{Y}wall_000013{Z}{Y}] {Z}"
       f"{Y}in OBJ file[{Z}{Y}/work/x/meshes/wall_000013.obj{Z}{Y}]{Z}")
MASK = (f"{Y}[Wrn] [Ogre2Camera.cc:423] {Z}{Y}Ogre2Camera::SetVisibilityMask: Mask bits {Z}{Y}c0000000{Z}{Y} are "
        f"set but will be ignored as they conflict with the {Z}{Y}reserved bits used internally by the ogre2 backend.{Z}")
DIST = f"{Y}[Wrn] [ImageBrownDistortionModel.cc:112] {Z}{Y}ImageBrownDistortionModel is not supported in {Z}{Y}ogre2{Z}"


def test_allowlisted_warnings_pass_and_are_counted():
    fails, allowed = S.gz_diagnostics([OBJ, OBJ, MASK, "[Msg] plain message"])
    assert fails == []
    assert sorted(allowed.values()) == [1, 2]


def test_ignored_rig_feature_fails():
    fails, _ = S.gz_diagnostics([OBJ, DIST, DIST])
    assert fails == ["[Wrn] [ImageBrownDistortionModel.cc:112] ImageBrownDistortionModel is not supported in ogre2"]


def test_any_error_fails_even_if_text_matches_an_allowed_warning():
    fails, _ = S.gz_diagnostics([OBJ.replace("[Wrn]", "[Err]")])
    assert len(fails) == 1


def test_unknown_warning_fails():
    fails, _ = S.gz_diagnostics(["[Wrn] [Foo.cc:1] something new"])
    assert fails == ["[Wrn] [Foo.cc:1] something new"]
