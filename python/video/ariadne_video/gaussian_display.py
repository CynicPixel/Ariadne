"""Gaussian model export and CPU alpha-composited display at a fixed Manim camera.

The completion model is generated from WorldGen, not measured reconstruction.
The binary is the same 32-byte .splat layout as the reconstruction service.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from manim import Animation, ImageMobject, config, smooth
from scipy.spatial.transform import Rotation

RECORD = np.dtype([("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("q", "u1", 4)])


def worldgen_splat(scene_dir: Path, out: Path) -> Path:
    """Seed oriented, flattened Gaussians on every textured quad except the roof."""
    out.mkdir(parents=True, exist_ok=True)
    parts, sources = [], []
    for mesh in sorted((scene_dir / "meshes").glob("*.obj")):
        if mesh.stem.startswith("ceiling_"):
            continue
        texture_path = scene_dir / "textures" / (mesh.stem + ".png")
        vertices = np.array([[float(v) for v in line.split()[1:]]
                             for line in mesh.read_text().splitlines() if line.startswith("v ")])
        if len(vertices) != 4:
            raise ValueError(f"Expected WorldGen quad: {mesh}")
        origin, eu, ev = vertices[0], vertices[1] - vertices[0], vertices[3] - vertices[0]
        lu, lv = np.linalg.norm(eu), np.linalg.norm(ev)
        normal = np.cross(eu, ev) / (lu * lv)
        spacing = 0.085 if mesh.stem.startswith("floor_") else 0.055
        nu, nv = max(1, int(np.ceil(lu / spacing))), max(1, int(np.ceil(lv / spacing)))
        u, v = np.meshgrid((np.arange(nu) + 0.5) / nu, (np.arange(nv) + 0.5) / nv)
        u, v = u.ravel(), v.ravel()
        data = np.zeros(len(u), dtype=RECORD)
        data["p"] = origin + u[:, None] * eu + v[:, None] * ev
        data["s"] = [0.7 * lu / nu, 0.7 * lv / nv, 0.003]
        # scipy is xyzw; .splat stores wxyz mapped to unsigned bytes.
        q = Rotation.from_matrix(np.column_stack((eu / lu, ev / lv, normal))).as_quat()
        q = q[[3, 0, 1, 2]]
        if q[0] < 0:
            q = -q
        data["q"] = np.clip(np.round(q * 128 + 128), 0, 255).astype(np.uint8)
        with Image.open(texture_path) as image:
            texture = np.array(image.convert("RGB"))
        h, w = texture.shape[:2]
        rgb = texture[np.minimum(h - 1, ((1 - v) * h).astype(int)),
                      np.minimum(w - 1, (u * w).astype(int))].astype(float)
        shade = 0.72 + 0.28 * abs(float(normal @ [0.3, -0.4, 0.866]))
        data["c"][:, :3] = np.round(rgb * shade).astype(np.uint8)
        data["c"][:, 3] = 235
        parts.append(data)
        sources.append({"mesh": mesh.name, "gaussians": len(data),
                        "mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
                        "texture_sha256": hashlib.sha256(texture_path.read_bytes()).hexdigest()})
    result = np.concatenate(parts)
    path = out / "worldgen_complete.splat"
    result.tofile(path)
    manifest = {"source": "WorldGen textured surfaces", "measured": False,
                "scene_id": json.loads((scene_dir / "scene.json").read_text())["scene_id"],
                "format": "position f32x3, scale f32x3, RGBA u8x4, quaternion wxyz u8x4",
                "gaussians": len(result), "roof": "omitted for interior view",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "surfaces": sources}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def gaussian_pixels(records: np.ndarray, transform: np.ndarray, frame, camera, crop=True, native=True):
    """Project 3D covariance and depth-composite elliptical Gaussian kernels.

    This is a CPU display renderer, not GPU training. Returns a tightly cropped
    RGBA image at the exact current camera pose. The camera stays still afterward.
    """
    world = records["p"].astype(float) @ transform[:3, :3].T + transform[:3, 3]
    points = world.copy()
    points[:, :2] -= frame.c
    points = points * frame.s + frame.center
    q = (records["q"].astype(float) - 128) / 128
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    rotation = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    axes = np.einsum("ij,njk->nik", transform[:3, :3], rotation)
    axes *= records["s"][:, None, :] * frame.s
    width, height = int(config.pixel_width), int(config.pixel_height)
    pixel_scale = np.array([width / config.frame_width, -height / config.frame_height])
    projected = camera.project_points(points)
    xy = projected[:, :2] * pixel_scale + [width / 2, height / 2]
    cov = np.zeros((len(points), 2, 2))
    for k in range(3):
        delta = (camera.project_points(points + axes[:, :, k])[:, :2] - projected[:, :2]) * pixel_scale
        cov += delta[:, :, None] * delta[:, None, :]
    cov[:, 0, 0] += 0.3
    cov[:, 1, 1] += 0.3
    inverse = np.linalg.inv(cov)
    extent = np.minimum(64, np.ceil(3 * np.sqrt(np.stack((cov[:, 0, 0], cov[:, 1, 1]), axis=1)))).astype(int)
    rgba = np.ascontiguousarray(records["c"].astype(float) / 255)
    library = Path(__file__).resolve().parents[3] / "build/video_assets/gaussian_raster.so"
    if native and library.exists():
        native = ctypes.CDLL(str(library)).raster_gaussians
        native.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 6
        native.restype = None
        arrays = [np.ascontiguousarray(xy), np.ascontiguousarray(inverse),
                  np.ascontiguousarray(extent, dtype=np.int32), rgba,
                  np.ascontiguousarray(np.argsort(projected[:, 2]), dtype=np.int64)]
        result = np.zeros((height, width, 4), dtype=np.float32)
        native(len(records), width, height, *[a.ctypes.data for a in arrays], result.ctypes.data)
        rgb, alpha = result[:, :, :3], result[:, :, 3]
    else:
        rgb = np.zeros((height, width, 3), dtype=np.float32)
        alpha = np.zeros((height, width), dtype=np.float32)
        # Increasing camera z is back to front in Manim's projection.
        for i in np.argsort(projected[:, 2]):
            x, y = xy[i]
            rx, ry = extent[i]
            x0, x1 = max(0, int(np.floor(x - rx))), min(width, int(np.ceil(x + rx)) + 1)
            y0, y1 = max(0, int(np.floor(y - ry))), min(height, int(np.ceil(y + ry)) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            dx, dy = np.meshgrid(np.arange(x0, x1) + 0.5 - x, np.arange(y0, y1) + 0.5 - y)
            inv = inverse[i]
            power = inv[0, 0] * dx * dx + 2 * inv[0, 1] * dx * dy + inv[1, 1] * dy * dy
            opacity = np.minimum(0.99, rgba[i, 3] * np.exp(-0.5 * power)).astype(np.float32)
            opacity[power > 9] = 0
            patch = rgb[y0:y1, x0:x1]
            patch *= 1 - opacity[:, :, None]
            patch += rgba[i, :3] * opacity[:, :, None]
            a = alpha[y0:y1, x0:x1]
            a *= 1 - opacity
            a += opacity
    ys, xs = np.nonzero(alpha > 1 / 255)
    if len(xs) == 0:
        raise ValueError("Gaussian layer has no visible pixels")
    x0, x1, y0, y1 = (xs.min(), xs.max() + 1, ys.min(), ys.max() + 1) if crop else (0, width, 0, height)
    a = alpha[y0:y1, x0:x1]
    colour = rgb[y0:y1, x0:x1] / np.maximum(a[:, :, None], 1e-8)
    pixels = np.clip(np.round(np.dstack((colour, a)) * 255), 0, 255).astype(np.uint8)
    return pixels, (x0, x1, y0, y1)


def gaussian_layer(records, transform, frame, camera) -> ImageMobject:
    pixels, (x0, x1, y0, y1) = gaussian_pixels(records, transform, frame, camera)
    width, height = int(config.pixel_width), int(config.pixel_height)
    image = ImageMobject(pixels)
    image.width = (x1 - x0) * config.frame_width / width
    image.move_to([(0.5 * (x0 + x1) - width / 2) * config.frame_width / width,
                   (height / 2 - 0.5 * (y0 + y1)) * config.frame_height / height, 0])
    return image


def gaussian_turnaround(records, frame, camera, out: Path, count=180) -> list[Path]:
    """Cache one full camera revolution of the saved Gaussian model at 30 fps."""
    key = hashlib.sha256(records.tobytes() + repr((int(config.pixel_width), int(config.pixel_height),
        camera.get_phi(), camera.get_theta(), camera.get_zoom(), frame.c.tolist(),
        frame.s, frame.center.tolist(), count, "gaussian-v1")).encode()).hexdigest()[:16]
    directory = out / key
    directory.mkdir(parents=True, exist_ok=True)
    theta = camera.get_theta()
    paths = [directory / f"frame_{i:04d}.png" for i in range(count + 1)]
    try:
        for i, path in enumerate(paths):
            if not path.exists():
                camera.theta_tracker.set_value(theta + 2 * np.pi * i / count)
                camera.reset_rotation_matrix()
                pixels, _ = gaussian_pixels(records, np.eye(4), frame, camera, crop=False)
                Image.fromarray(pixels).save(path, compress_level=1)
            if i % 30 == 0:
                print(f"Gaussian turnaround: {i}/{count}", flush=True)
    finally:
        camera.theta_tracker.set_value(theta)
        camera.reset_rotation_matrix()
    return paths


class AlphaFade(Animation):
    """Fade an RGBA Gaussian layer without changing positions or revealing it early."""
    def __init__(self, image: ImageMobject, **kwargs):
        self.target_alpha = image.pixel_array[:, :, 3].copy()
        image.pixel_array[:, :, 3] = 0
        super().__init__(image, **kwargs)

    def interpolate_mobject(self, alpha: float) -> None:
        self.mobject.pixel_array[:, :, 3] = np.round(self.target_alpha * smooth(alpha)).astype(np.uint8)
