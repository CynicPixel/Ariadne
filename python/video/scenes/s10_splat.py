"""Beat 10: maze skeleton -> trained submaps -> complete-map illustration.

Training: AWS 20261024T121121Z-gpu-ae604c, full 200 s mission 626abe,
100 iterations/KF. Assembly samples actual trained Gaussian centres/colours;
this is an editorial 3D reveal in submap creation order, not a training replay.
WorldGen walls are a reference skeleton, not reconstructed geometry. Every
chunk uses its recorded T_W_anchor. The ending uses a CUDA gsplat model
optimized against rendered WorldGen camera views, not a measured robot run. Both the submaps and ending display real Gaussian footprints
with alpha compositing, rather than PMobject point sprites.

Fetch: bash python/video/tools/fetch_shared_model.sh
Render: python/video/render.sh scenes/s10_splat.py SharedModel -qm
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from manim import (DEGREES, DOWN, UP, FadeIn, FadeOut,
                   LaggedStart, Polygon, Text, ThreeDScene,
                   VGroup, ImageMobject, ValueTracker, config, linear)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ariadne_video.gaussian_display import AlphaFade, RECORD, gaussian_layer, gaussian_turnaround
from PIL import Image
from ariadne_video.data import load_scene
from ariadne_video.evidence import REPO
from ariadne_video.mobjects import maze_frame
from ariadne_video.style import DATA, DIM, FONT, TEXT, THREAD, WALL, CaptionedScene, tag

ASSETS = REPO / "build/video_assets/splat_626abe_render_20261024_aws"
WALL_H = 2.95
DISPLAY_POINTS = 60000


def trained_clouds(table: dict) -> list[tuple[dict, np.ndarray]]:
    """Deterministic per-submap sampling, preserving real colours and world placement."""
    rows = sorted(table["submaps"], key=lambda row: row["sid"])
    total = sum(row["gaussians"] for row in rows)
    clouds = []
    for row in rows:
        source = ASSETS / "splat_batch/chunks" / row["chunk"]
        data = np.fromfile(source, dtype=RECORD)
        if len(data) != row["gaussians"]:
            raise ValueError(f"Gaussian count differs from table: {source}")
        count = min(len(data), max(1, round(DISPLAY_POINTS * len(data) / total)))
        rng = np.random.default_rng(row["sid"])
        sample = data[rng.choice(len(data), count, replace=False)]
        clouds.append((row, sample))
    return clouds


class SharedModel(CaptionedScene, ThreeDScene):
    def construct(self):
        render = ASSETS / "render"
        model = ASSETS / "splat_batch"
        if not (model / "table.json").exists():
            raise FileNotFoundError("Fetch trained chunks: bash python/video/tools/fetch_shared_model.sh")
        table = json.loads((model / "table.json").read_text())
        poses = json.loads((render / "poses.json").read_text())
        manifest = json.loads((ASSETS / "training_manifest.json").read_text())
        scene = load_scene(DATA / "maze_demo_0")
        if any(not run_id.startswith(scene.scene_id + "-") for run_id in manifest["spool_run_id"]):
            raise ValueError("WorldGen skeleton and training mission are different scenes")
        assert manifest["iters"] == 100
        assert len(table["submaps"]) == poses["submaps"]
        assert sum(row["gaussians"] for row in table["submaps"]) == poses["gaussians"]
        frame = maze_frame(scene, height=4.8, center=(0, 0.2, 0))
        samples = trained_clouds(table)
        walls = VGroup(*[
            Polygon(frame.p(x0, y0, 0), frame.p(x1, y1, 0),
                    frame.p(x1, y1, WALL_H), frame.p(x0, y0, WALL_H),
                    stroke_color=WALL, stroke_width=1.0, stroke_opacity=0.55,
                    fill_color=WALL, fill_opacity=0.025)
            for x0, y0, x1, y1 in scene.walls
        ])
        title = Text("One shared model", font=FONT, font_size=30, color=TEXT).move_to(UP * 3.55).set_z_index(20)
        chip = tag("measured", "trained submaps · assembly animation").move_to(UP * 3.02).set_z_index(20)
        note = Text("WorldGen building outline · sampled Gaussian centres", font=FONT,
                    font_size=16, color=DIM).move_to(DOWN * 2.65)
        self.add_fixed_in_frame_mobjects(title, chip, note)
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)
        self.play(FadeIn(walls), FadeIn(title), FadeIn(chip), FadeIn(note), run_time=0.8)
        self.say("As the robots explore, their views build one shared model.", dur=4.0)
        self.move_camera(phi=62 * DEGREES, theta=-60 * DEGREES, zoom=1.05, run_time=2.0)

        # Render true Gaussian footprints at the settled camera. Image layers
        # alpha-composite correctly; Cairo PMobject fades are a no-op.
        clouds = []
        for row, sample in samples:
            transform = np.array(row["T_W_anchor"]).reshape(4, 4)
            cloud = gaussian_layer(sample, transform, frame, self.camera)
            self.add_fixed_in_frame_mobjects(cloud)
            self.remove(cloud)
            clouds.append((row, cloud))
        def assemble(parts, seconds):
            return LaggedStart(*[
                AlphaFade(cloud, run_time=1.5)
                for row, cloud in parts
            ], lag_ratio=0.2, run_time=seconds)

        self.say("Each robot's camera keyframes become small, coloured Gaussian splats.", dur=4.5,
                 run_with=[assemble(clouds[:15], 4.5)])
        self.say("Piece by piece, their submaps fill the building's shared frame.", dur=6.0,
                 run_with=[assemble(clouds[15:], 6.0)])
        stats = Text(f"{manifest['keyframes']:,} keyframes   /   {len(clouds)} submaps   /   "
                     f"{poses['gaussians']:,} Gaussians", font=FONT, font_size=20,
                     color=THREAD).move_to(UP * 2.5).set_z_index(20)
        self.add_fixed_in_frame_mobjects(stats)
        self.say("These trained submaps come from a two-hundred-second mission.",
                 dur=3.5, run_with=[FadeIn(stats), walls.animate.set_stroke(opacity=0.18)])

        # The full building is an illustrative endpoint. Remove measured-run
        # statistics and the measured badge before introducing generated geometry.
        trained_dir = ASSETS / "worldgen_trained"
        training_record = json.loads((trained_dir / "manifest.json").read_text())
        if training_record.get("steps", 0) < 1 or training_record.get("training") != "CUDA gsplat gradient optimization":
            raise ValueError("The complete model must be gsplat-trained, not surface-seeded")
        splat_path = trained_dir / "trained_worldgen.splat"
        complete = gaussian_layer(np.fromfile(splat_path, dtype=RECORD), np.eye(4), frame, self.camera)
        self.add_fixed_in_frame_mobjects(complete)
        self.remove(complete)
        chip2 = tag("illustrative", "gsplat trained on rendered views of the generated maze, not a robot run")
        chip2.move_to(UP * 3.02).set_z_index(20)
        self.add_fixed_in_frame_mobjects(chip2)
        self.remove(chip2)
        self.play(FadeOut(stats), FadeOut(chip), FadeOut(note), run_time=0.4)
        self.say("Without a mission time limit, exploration can continue toward a complete shared map.",
                 dur=5.0, run_with=[AlphaFade(complete, run_time=3.5), FadeIn(chip2, run_time=0.6),
                                    *[FadeOut(cloud, run_time=3.5) for _, cloud in clouds],
                                    FadeOut(walls, run_time=3.5)])
        self.wait(0.5)
        self.clear_caption()
        final_records = np.fromfile(splat_path, dtype=RECORD)
        paths = gaussian_turnaround(final_records, frame, self.camera, ASSETS / "worldgen_trained/turnaround")
        turn = ImageMobject(str(paths[0]))
        turn.width = config.frame_width
        self.add_fixed_in_frame_mobjects(turn)
        self.remove(complete)
        position, previous = ValueTracker(0), [-1]
        def turn_frame(mob):
            i = min(len(paths) - 1, max(0, int(position.get_value())))
            if i != previous[0]:
                with Image.open(paths[i]) as image:
                    mob.pixel_array = np.array(image.convert("RGBA"))
                previous[0] = i
        turn.add_updater(turn_frame)
        self.play(position.animate(run_time=6.0, rate_func=linear).set_value(len(paths) - 1))
        turn.clear_updaters()
        self.wait(0.5)
        self.play(FadeOut(turn), FadeOut(title), FadeOut(chip2), run_time=0.6)
