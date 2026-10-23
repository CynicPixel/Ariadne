"""Write a generated scene to disk (design §3-§4).

<out>/
  model.config, model.sdf   the static scene as ONE gz model (visual quads + ceiling
                            lights); model_inline.xml is the same <model> element for
                            worlds to inline (relative <include> does not resolve, SMOKE 4i)
  meshes/<sid>.obj          one textured quad per surface (mesh.py convention)
  textures/<sid>.png        its albedo (texture.py), deterministic pixels
  world.sdf                 standalone viewing world, scene model inlined
  esdf_f32.bin              the navigable ESDF as raw float32 (geometry in scene.json)
  scene_gt.npz              hidden GT (never visible to perception): grid, structure
                            and navigable free masks, ESDF, traversability at both rho,
                            geodesic distance fields from every start, egress routes
  scene.json                manifest: identity, preset, topology, lengths, surfaces
                            with pixel digests, lights, props, files
  routes/agent_<k>.json     the mover script: timed S-curve segments (routes.py)
  preview.png               top-down map (preview.py)

Float formatting in SDF is fixed (%.5f) and file order is sorted, so two builds of
the same (seed, preset) produce identical text files; PNGs are compared by pixel
digest (zlib builds may differ in compressed bytes).
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import GENERATOR_VERSION
from .furnish import hash_key, lights, zone_styles
from .assets import load_charuco
from .geometry import K_BOARD, K_CEIL, K_FLOOR, K_PROP, K_WALL, board_surfaces, floor_ceiling_surfaces, \
    prop_surfaces, wall_runs, wall_surfaces
from .layout import _edge_key
from .mesh import write_obj
from .png import pixel_digest, write_png
from .preview import write_preview
from .raster import extract_path, geodesic
from .texture import Style, albedo

KIND_NAMES = {K_WALL: "wall", K_FLOOR: "floor", K_CEIL: "ceiling", K_PROP: "prop", K_BOARD: "board"}


def _f(x: float) -> str:
    s = f"{x:.5f}"
    return "0.00000" if s == "-0.00000" else s


def model_name(scene) -> str:
    return f"ariadne_scene_{scene.scene_id}"


def _zone_of(scene, surface):
    """The appearance zone a surface belongs to: rooms win over vestibules over passages."""
    L = scene.layout
    if surface.kind == K_PROP:
        prop = scene.props[(surface.sid & 0xFFFFFF) >> 4]
        return ("room", prop.room)
    if surface.kind == K_BOARD:
        return ("room", scene.boards[surface.sid & 0xFFFFFF].room)
    x, y = surface.probe_xy_units
    ranked = sorted(L.free_rects(), key=lambda t: {"room": 0, "vestibule": 1, "passage": 2}[t[0]])
    for kind, owner, r in ranked:
        if r.x0 <= x <= r.x1 and r.y0 <= y <= r.y1:
            return (kind, owner)
    raise AssertionError(f"surface {surface.sid:#x} is in no zone")


def surfaces(scene):
    P = scene.preset
    g = scene.grid.g
    runs = wall_runs(scene.structure_free, scene.grid)
    out = wall_surfaces(runs, g, scene.wall_height_m, P.appearance.wall_px_per_m)
    out += floor_ceiling_surfaces(scene.structure_free, scene.grid, scene.wall_height_m,
                                  P.appearance.floor_px_per_m)
    out += prop_surfaces(scene.props, g, P.appearance.wall_px_per_m)
    ch = load_charuco()
    out += board_surfaces(scene.boards, g, ch.size_m[0], ch.images[0].shape[:2])
    return out


def texture_key(scene, sid: int) -> int:
    """32-bit Philox counter word for a surface: the surface id XOR an attempt-dependent
    constant (a bijection, so ids stay distinct within a scene)."""
    return (sid ^ ((scene.attempt * 0x9E3779B9) & 0xFFFFFFFF)) & 0xFFFFFFFF


def _visual(name, mesh_uri, tex_uri) -> str:
    return (f'      <visual name="{name}"><cast_shadows>true</cast_shadows>'
            f'<geometry><mesh><uri>{mesh_uri}</uri></mesh></geometry>'
            f'<material><diffuse>1 1 1 1</diffuse><specular>0.05 0.05 0.05 1</specular>'
            f'<pbr><metal><albedo_map>{tex_uri}</albedo_map><roughness>0.9</roughness>'
            f'<metalness>0</metalness></metal></pbr></material></visual>\n')


def _light(pl, P) -> str:
    d = _f(P.lighting.light_diffuse)
    return (f'      <light type="point" name="{pl.name}"><pose>{_f(pl.x_m)} {_f(pl.y_m)} {_f(pl.z_m)} 0 0 0</pose>'
            f'<diffuse>{d} {d} {d} 1</diffuse><specular>0.1 0.1 0.1 1</specular>'
            f'<attenuation><range>{_f(P.lighting.light_range_m)}</range><constant>0.4</constant>'
            f'<linear>0.08</linear><quadratic>0.02</quadratic></attenuation>'
            f'<cast_shadows>false</cast_shadows></light>\n')


def write_scene(scene, out_dir: str, textures: bool = True) -> dict:
    P = scene.preset
    os.makedirs(os.path.join(out_dir, "meshes"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "textures"), exist_ok=True)
    styles = zone_styles(scene.layout)
    surf = surfaces(scene)
    surf_rows, visuals = [], []
    for s in sorted(surf, key=lambda x: x.sid):
        name = f"{KIND_NAMES[s.kind]}_{s.sid & 0xFFFFFF:06x}"
        write_obj(os.path.join(out_dir, "meshes", f"{name}.obj"), s.quad, name)
        zone = _zone_of(scene, s)
        style: Style = styles[zone]
        row = {"name": name, "sid": s.sid, "kind": KIND_NAMES[s.kind],
               "zone": [zone[0], hash_key(zone)], "size_px": list(s.size_px),
               "size_m": [round(v, 5) for v in s.quad.size()],
               "normal": [round(float(v), 6) for v in s.quad.normal()]}
        if textures:
            if s.kind == K_BOARD:
                gray = load_charuco().images[scene.boards[s.sid & 0xFFFFFF].agent]
                img = np.repeat(gray[:, :, None], 3, axis=2)
            else:
                img = albedo(scene.seed, texture_key(scene, s.sid), s.size_px[0], s.size_px[1],
                             style, P.appearance.tau)
            write_png(os.path.join(out_dir, "textures", f"{name}.png"), img)
            row["pixel_sha256"] = pixel_digest(img)
        surf_rows.append(row)
        visuals.append(_visual(name, f"meshes/{name}.obj", f"textures/{name}.png"))
    pls = lights(scene.layout, scene.wall_height_m)
    mname = model_name(scene)
    model_xml = (f'  <model name="{mname}">\n    <static>true</static>\n    <link name="structure">\n'
                 + "".join(visuals) + "".join(_light(pl, P) for pl in pls)
                 + "    </link>\n  </model>\n")
    with open(os.path.join(out_dir, "model.sdf"), "w", newline="\n") as f:
        f.write('<?xml version="1.0" ?>\n<sdf version="1.10">\n' + model_xml + "</sdf>\n")
    with open(os.path.join(out_dir, "model.config"), "w", newline="\n") as f:
        f.write('<?xml version="1.0" ?>\n<model>\n'
                f'  <name>{mname}</name>\n  <version>{GENERATOR_VERSION}</version>\n'
                '  <sdf version="1.10">model.sdf</sdf>\n'
                '  <description>Ariadne procedural scene (dataset_tools.worldgen). Included by '
                'model://DIRNAME only with GZ_SIM_RESOURCE_PATH set to the parent directory; '
                'world.sdf inlines it instead (SMOKE 4i).</description>\n'
                '</model>\n')
    with open(os.path.join(out_dir, "world.sdf"), "w", newline="\n") as f:
        f.write(world_xml(scene, model_xml))
    with open(os.path.join(out_dir, "model_inline.xml"), "w", newline="\n") as f:
        f.write(model_xml)          # mission composer inlines this beside the agents

    os.makedirs(os.path.join(out_dir, "routes"), exist_ok=True)
    for sc in scene.scripts:
        with open(os.path.join(out_dir, "routes", f"agent_{sc.agent}.json"), "w", newline="\n") as f:
            json.dump(sc.to_json(), f, indent=1, sort_keys=True)
            f.write("\n")
    gt = ground_truth(scene)
    np.savez_compressed(os.path.join(out_dir, "scene_gt.npz"), **gt)
    # The mover's collision-envelope check reads this (C++, no npz/zlib dependency).
    scene.esdf_m.astype("<f4").tofile(os.path.join(out_dir, "esdf_f32.bin"))
    write_preview(scene, os.path.join(out_dir, "preview.png"))
    manifest = scene_manifest(scene, surf_rows, pls)
    with open(os.path.join(out_dir, "scene.json"), "w", newline="\n") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")
    return manifest


def world_xml(scene, model_xml: str, extra: str = "") -> str:
    """A world file that INLINES the scene model. It must live in the scene directory:
    mesh/texture URIs are relative to the SDF file (SMOKE 4h); <include> of a
    relative path does not resolve in sdformat14 (SMOKE 4i)."""
    amb = _f(scene.preset.lighting.ambient)
    return ('<?xml version="1.0" ?>\n<sdf version="1.10">\n'
            f'  <world name="scene_{scene.scene_id}">\n'
            '    <physics name="1ms" type="ignored"><max_step_size>0.001</max_step_size>'
            '<real_time_factor>1.0</real_time_factor></physics>\n'
            '    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>\n'
            '    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">'
            '<render_engine>ogre2</render_engine></plugin>\n'
            f'    <scene><ambient>{amb} {amb} {amb} 1</ambient><background>0 0 0 1</background>'
            f'<grid>false</grid><shadows>{"true" if scene.preset.lighting.shadows else "false"}</shadows></scene>\n'
            + model_xml + extra + "  </world>\n</sdf>\n")


def ground_truth(scene) -> dict:
    g = scene.grid
    out = {
        "grid_origin_units": np.array([g.ox, g.oy], np.int64),
        "grid_m": np.array(g.g),
        "structure_free": scene.structure_free,
        "free": scene.free,
        "esdf_m": scene.esdf_m.astype(np.float32),
        "trav_agent": scene.trav_agent,
        "trav_responder": scene.trav_responder,
        "target_cell": np.array(scene.target_cell, np.int64),
        "start_cells": np.array(scene.start_cells, np.int64),
    }
    for k, s in enumerate(scene.start_cells):
        for label, trav in (("responder", scene.trav_responder), ("agent", scene.trav_agent)):
            dist, parent = geodesic(trav, s, g.g)
            out[f"geodesic_{label}_from_start{k}_m"] = dist.astype(np.float32)
            if np.isfinite(dist[scene.target_cell]):
                out[f"route_{label}_start{k}"] = np.array(extract_path(parent, g.width, scene.target_cell),
                                                         np.int32)
    return out


def board_manifest(scene) -> list:
    """Per board: its asset, marker ids, and the world corners of the SQUARES area (inside
    the white margin) as seen from the front, TL TR BR BL. Evaluation maps OpenCV's board
    frame onto these; no convention is assumed here."""
    from .geometry import board_quad
    ch = load_charuco()
    u0, v0, u1, v1 = ch.squares_fraction
    out = []
    for bp in scene.boards:
        q = board_quad(bp, scene.grid.g, ch.size_m[0])
        p0, eu, ev = (np.asarray(v, float) for v in (q.p0, q.eu, q.ev))
        at = lambda u, v: [round(float(c), 6) for c in p0 + u * eu + v * ev]
        info = ch.meta["boards"][bp.agent]
        out.append({"agent": bp.agent, "room": list(bp.room), "wall": bp.wall, "asset": info["file"],
                    "first_id": info["first_id"], "last_id": info["last_id"],
                    "dictionary": ch.meta["dictionary"], "squares_xy": ch.meta["squares_xy"],
                    "square_m": ch.meta["square_m"], "marker_m": ch.meta["marker_m"],
                    "normal": [round(float(c), 6) for c in q.normal()],
                    "squares_corners_world_tl_tr_br_bl": [at(u0, v1), at(u1, v1), at(u1, v0), at(u0, v0)]})
    return out


def scene_manifest(scene, surf_rows, pls) -> dict:
    L = scene.layout
    g = scene.grid.g
    P = scene.preset
    return {
        "scene_id": scene.scene_id,
        "generator_version": GENERATOR_VERSION,
        "seed": scene.seed,
        "attempt": scene.attempt,
        "rejections": dict(sorted(scene.rejections.items())),
        "preset": {"name": P.name, "digest": P.digest(), "values": P.to_canonical()},
        "frame": "world metres; origin = lattice SW corner; x east, y north, z up",
        "grid": {"origin_units": [scene.grid.ox, scene.grid.oy], "cell_m": g,
                 "shape_rows_cols": [scene.grid.height, scene.grid.width]},
        "wall_height_m": scene.wall_height_m,
        "esdf": {"file": "esdf_f32.bin", "dtype": "float32 little-endian, row-major",
                 "rows": scene.grid.height, "cols": scene.grid.width, "cell_m": g,
                 "origin_m": [scene.grid.ox * g, scene.grid.oy * g],
                 "layout": "row r, col c covers [ox+c, ox+c+1) x [oy+r, oy+r+1) cells; row 0 is SOUTH",
                 "cap_m": 4.0, "meaning": "metres from the cell centre to the nearest obstacle "
                                            "(negative inside obstacles); navigable = structure minus props"},
        "topology": {
            "cells": sorted([list(c), L.kind[c]] for c in L.cells),
            "rooms": {f"{c[0]},{c[1]}": [r.x0 * g, r.y0 * g, r.x1 * g, r.y1 * g]
                      for c, r in sorted(L.rooms.items())},
            "passages": [{"a": list(p.a), "b": list(p.b), "width_m": round(p.width * g, 5),
                          "narrow": p.narrow, "braid": e in L.braid_edges}
                         for e, p in sorted(L.passages.items(), key=lambda t: _edge_key(t[0]))],
            "stubs": sorted(list(c) for c in L.stub_cells),
            "target_room": list(scene.target_room),
        },
        "entrances": [{"agent": k, "room": list(e.room), "side": e.side, "width_m": round(e.width * g, 5),
                       "start_xy_m": [e.start[0] * g, e.start[1] * g], "start_yaw_deg": e.start_yaw_deg}
                      for k, e in enumerate(L.entrances)],
        "target_xy_m": list(scene.grid.cell_center_m(*scene.target_cell)),
        "lengths_m": {str(k): v for k, v in scene.lengths.items()},
        "clearance": {"rho_agent_m": P.clearance.rho_agent_m, "rho_responder_m": P.clearance.rho_responder_m},
        "props": [{"room": list(p.room), "rect_m": [p.rect.x0 * g, p.rect.y0 * g, p.rect.x1 * g, p.rect.y1 * g],
                   "height_m": p.height_m} for p in scene.props],
        "lights": [{"name": pl.name, "xyz_m": [pl.x_m, pl.y_m, pl.z_m]} for pl in pls],
        "charuco_boards": board_manifest(scene),
        "routes": {"template": P.routes.template,
                   "start_delays_s": list(scene.start_delays_s),
                   "min_agent_separation_m": (round(scene.min_separation_m, 4)
                                              if np.isfinite(scene.min_separation_m) else None),
                   "agent_visuals": "none (Phase 1: agents are invisible to each other; D37)",
                   "duration_s": {str(sc.agent): round(sc.duration, 3) for sc in scene.scripts},
                   "segments": {str(sc.agent): len(sc.segments) for sc in scene.scripts},
                   "overlap": scene.overlap},
        "surfaces": surf_rows,
    }
