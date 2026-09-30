import numpy as np
import pytest

from dataset_tools.worldgen import mesh, png


def test_png_round_trip_rgb_and_gray(tmp_path):
    rgb = (np.arange(7 * 5 * 3) % 256).astype(np.uint8).reshape(7, 5, 3)
    gray = (np.arange(6 * 9) * 7 % 256).astype(np.uint8).reshape(6, 9)
    for img in (rgb, gray):
        p = tmp_path / "t.png"
        png.write_png(p, img)
        back = png.read_png(p)
        assert back.dtype == np.uint8 and np.array_equal(back, img)
        assert png.pixel_digest(back) == png.pixel_digest(img)


def test_png_rejects_non_uint8(tmp_path):
    with pytest.raises(TypeError):
        png.write_png(tmp_path / "x.png", np.zeros((2, 2), np.float32))


def test_pixel_digest_depends_on_shape():
    a = np.zeros((2, 6), np.uint8)
    assert png.pixel_digest(a) != png.pixel_digest(a.reshape(3, 4))


def test_quad_normal_uv_and_winding():
    q = mesh.Quad((3.0, 2.2, 0.25), (0.0, -2.0, 0.0), (0.0, 0.0, 1.5))
    assert np.allclose(q.normal(), [-1, 0, 0])
    assert q.size() == (2.0, 1.5)
    text = mesh.obj_text(q, "w")
    assert "vn -1.00000 0.00000 0.00000" in text
    # CCW seen from the front: (v2-v1) x (v3-v1) is along the front normal.
    c = q.corners()
    assert np.dot(np.cross(c[1] - c[0], c[2] - c[0]), q.normal()) > 0
    assert text.count("\nf ") == 2 and "-0.00000" not in text


def test_quad_rejects_sheared_or_degenerate():
    with pytest.raises(ValueError):
        mesh.obj_text(mesh.Quad((0, 0, 0), (1, 0, 0), (0.5, 0, 1)))
    with pytest.raises(ValueError):
        mesh.Quad((0, 0, 0), (1, 0, 0), (2, 0, 0)).normal()


def test_obj_is_byte_stable(tmp_path):
    q = mesh.Quad((0.1, 0.2, 0.3), (0.0, 4.05, 0.0), (0.0, 0.0, 2.8))
    mesh.write_obj(tmp_path / "a.obj", q)
    mesh.write_obj(tmp_path / "b.obj", q)
    assert (tmp_path / "a.obj").read_bytes() == (tmp_path / "b.obj").read_bytes()
