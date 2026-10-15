"""Evaluation harness v1 (design HARNESS_1.9): one report per Phase 1 run chain.

  python -m analysis.report compute --collab-run C --recon-run R --mission-run M --mission-compare A0 A1 \
      --scene S --registry REGISTRY.yaml --out OUT [--allow-dirty]
  python -m analysis.report render --out OUT          (after ariadne_report_check wrote OUT/report_check.txt)

`compute` reads only the chain's committed outputs, computes every §30.2 registry metric with the definition
fixed in the design, runs the gates (clean inputs, consistent merge transform) and the oracles (C++ per-agent
results, the 1.6 merged ATE, evo 1.31.1 APE/RPE), and writes report.yaml / oracle.yaml / inputs.yaml.
`render` draws the figures and the self-contained report.html. Nothing here is hand-edited (§36).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

from . import trajectory as tr

CORRECT_T_M, CORRECT_R_DEG = 0.05, 1.0       # 1.6 edge correctness (collab_v1 edge_correct_*)
ENERGY_M_PER_RAD = 0.25                      # [PROXY] energy: 1 rad of turning ~ 0.25 m of travel
ORACLE_REL_TOL = 1e-6

# The definitions fixed in docs/design/HARNESS_1.9.md (shown beside every number in the report).
DEFINITIONS = {
    "ate_rmse_m": "merged two-agent trajectory, one SE(3) Umeyama alignment, position RMSE (evo APE)",
    "ate_rotation_rmse_rad": "same alignment; RMSE of angle(R_gt^T R_align R_est)",
    "rpe_translation_rmse_m": "pose pairs 1 s apart (20 frames, all pairs), per agent, pooled; RMSE of |t_E|",
    "rpe_rotation_rmse_rad": "same pairs; RMSE of angle(R_E)",
    "inter_agent_rpe_m": "RMSE of the translation error of the exchanged verified inter-agent edges vs GT",
    "closure_precision": "correct / accepted over every pair the verifier accepted (exchange + negative probe)",
    "false_closure_rate": "incorrect / exchanged edges: wrong closures that reached the map",
    "chamfer_m": "fused cloud vs GT mesh samples: mean of accuracy and GT-to-cloud distance",
    "completeness": "fraction of GT mesh samples within 5 cm of the fused cloud",
    "coverage": "GT-free cells the merged mission map observed free",
    "heldout_psnr_db": "39 held-out views rendered from the UNTRAINED Gaussian initialization",
    "heldout_ssim": "same views; SSIM (Wang et al. 2004)",
    "heldout_lpips": "same views; LPIPS AlexNet (Zhang et al. 2018), computed on the remote GPU",
    "bytes_total": "every inter-agent message, wire bytes at the serializer (ByteLedger)",
    "bytes_per_keyframe": "total / keyframes of both agents",
    "bytes_per_verified_loop": "total / verified exchanged edges",
    "bytes_per_map_area_m2": "total / observed-free area of the merged mission map",
    "descriptor_bytes": "GLOBAL_DESCRIPTOR class",
    "local_feature_bytes": "MATCH_REQUEST + MATCH_RESPONSE classes",
    "pose_graph_bytes": "POSE_EDGE + POSE_GRAPH classes",
    "time_to_first_usable_map_s": "[ASSUMPTION] first replan with any target-to-exit route at rho_agent",
    "time_to_first_valid_egress_s": "first replan with an all-observed-free route at rho_responder (0.40 m)",
    "route_validity": "final egress route: fraction of cells in GT traversability at rho_responder",
    "route_confidence": "final egress route: 5th percentile of per-cell support [PROXY]",
    "route_confidence_calibration": "[PROXY] Brier score of support vs GT validity over this run's replans; not a calibration estimate",
    "distance_travelled_m": "GT path length of both agents",
    "energy_proxy": "[PROXY] GT path length + 0.25 m/rad x GT yaw travel",
}


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ly(p: Path) -> dict:
    return yaml.safe_load(Path(p).read_text())


def read_csv(p: Path) -> list[dict]:
    with open(p) as f:
        return list(csv.DictReader(f))


class Inputs:
    """Every file the report reads, with its sha256 (inputs.yaml)."""

    def __init__(self):
        self.files: dict[str, str] = {}

    def use(self, p: Path) -> Path:
        self.files[str(p)] = sha(p)
        return p


def agent_trajectories(collab: Path, T_O0_O1: np.ndarray, inp: Inputs):
    out = []
    for k in (0, 1):
        d = next((collab / "raw").glob(f"*-agent{k}-*"))
        t, E = tr.load_tum(inp.use(d / "estimate_body.tum"))
        tg, G = tr.load_tum(inp.use(d / "raw" / "groundtruth_body.tum"))
        t, E, G = tr.associate(t, E, tg, G)
        out.append({"agent": k, "t": t, "E_own": E, "E": E if k == 0 else T_O0_O1 @ E, "G": G,
                    "results": ly(inp.use(d / "results.yaml"))})
    return out


def path_length(T: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1).sum())


def yaw_travel(T: np.ndarray) -> float:
    yaw = np.arctan2(T[:, 1, 0], T[:, 0, 0])
    d = np.diff(yaw)
    return float(np.abs((d + np.pi) % (2 * np.pi) - np.pi).sum())


def evo_oracle(trajs, E_all, G_all) -> dict:
    """evo 1.31.1 APE (merged, one alignment) and RPE (per agent, delta 20 frames, all pairs, pooled).
    Agent 1's stamps are offset by 1e6 s so evo can never pair one agent's estimate with the other's GT."""
    from evo.core import metrics
    from evo.core.metrics import Unit
    from evo.core.trajectory import PoseTrajectory3D
    import evo

    def traj(T, t_ns):
        return PoseTrajectory3D(poses_se3=list(T), timestamps=t_ns.astype(np.float64) * 1e-9)

    t_all = np.concatenate([a["t"] + a["agent"] * 10 ** 15 for a in trajs])
    ref, est = traj(G_all, t_all), traj(E_all, t_all)
    est.align(ref, correct_scale=False)
    out = {"evo_version": evo.__version__}
    for name, rel in (("ape_translation_rmse_m", metrics.PoseRelation.translation_part),
                      ("ape_rotation_rmse_rad", metrics.PoseRelation.rotation_angle_rad)):
        m = metrics.APE(rel)
        m.process_data((ref, est))
        out[name] = float(m.get_statistic(metrics.StatisticsType.rmse))
    for name, rel in (("rpe_translation_rmse_m", metrics.PoseRelation.translation_part),
                      ("rpe_rotation_rmse_rad", metrics.PoseRelation.rotation_angle_rad)):
        errs = []
        for a in trajs:
            m = metrics.RPE(rel, delta=20, delta_unit=Unit.frames, all_pairs=True)
            m.process_data((traj(a["G"], a["t"]), traj(a["E_own"], a["t"])))
            errs.append(m.error)
        out[name] = tr.rms(np.concatenate(errs))
    return out


def route_gt_validity(cells, trav) -> float:
    c = np.array(cells)
    return float(trav[c[:, 0], c[:, 1]].mean())


def compute(a) -> int:
    inp = Inputs()
    collab, recon, mission, scene = Path(a.collab_run), Path(a.recon_run), Path(a.mission_run), Path(a.scene)
    gates = {}

    # --- gates: clean inputs, one merge transform -----------------------------------------------------
    dirty = {}
    for p in [collab / "manifest.yaml", recon / "manifest_prepare.yaml", recon / "manifest_score.yaml",
              mission / "manifest.yaml"] + [Path(c) / "manifest.yaml" for c in a.mission_compare]:
        dirty[str(p)] = str(ly(inp.use(p))["git_dirty"]).lower()
    gates["inputs_clean"] = all(v == "false" for v in dirty.values())
    mres = ly(inp.use(mission / "results.yaml"))
    T01 = np.eye(4)
    T01[:3, :] = np.array(mres["T_O0_O1"], float).reshape(3, 4)
    T01s = np.eye(4)
    T01s[:3, :] = np.array(ly(inp.use(recon / "spool.yaml"))["merge"]["T_O0_O1"], float).reshape(3, 4)
    gates["merge_transform_consistent"] = bool(np.allclose(T01, T01s, atol=1e-9))
    if not gates["merge_transform_consistent"]:
        raise SystemExit("gate: the mission's T_O0_O1 differs from the spool's")
    if not gates["inputs_clean"] and not a.allow_dirty:
        raise SystemExit(f"gate: dirty inputs {[k for k, v in dirty.items() if v != 'false']} (use --allow-dirty for dev)")

    # --- localization ------------------------------------------------------------------------------------
    trajs = agent_trajectories(collab, T01, inp)
    E_all = np.concatenate([x["E"] for x in trajs])
    G_all = np.concatenate([x["G"] for x in trajs])
    merged = tr.ate(E_all, G_all)
    rpe_t, rpe_r = zip(*[tr.rpe_errors(x["E_own"], x["G"], 20) for x in trajs])
    M = {"ate_rmse_m": merged["rmse_m"], "ate_rotation_rmse_rad": merged["rotation_rmse_rad"],
         "rpe_translation_rmse_m": tr.rms(np.concatenate(rpe_t)), "rpe_rotation_rmse_rad": tr.rms(np.concatenate(rpe_r))}

    # --- oracles -------------------------------------------------------------------------------------------
    cres = ly(inp.use(collab / "results.yaml"))
    oracle = {"internal": {}, "evo": {}}
    for x in trajs:
        own = tr.ate(x["E_own"], x["G"])
        et, er = tr.rpe_errors(x["E_own"], x["G"], 20)
        ref = x["results"]
        oracle["internal"][f"agent{x['agent']}"] = {
            "ate_rmse_m": [own["rmse_m"], ref["ate_se3"]["rmse_m"]],
            "ate_rotation_rmse_rad": [own["rotation_rmse_rad"], ref["ate_se3"]["rotation_rmse_rad"]],
            "rpe_translation_rmse_m": [tr.rms(et), ref["rpe_1s"]["translation_rmse_m"]],
            "rpe_rotation_rmse_rad": [tr.rms(er), ref["rpe_1s"]["rotation_rmse_rad"]]}
    oracle["internal"]["merged_ate_vs_1.6"] = {"ate_rmse_m": [merged["rmse_m"], cres["merge"]["merged_two_agent_ate_rmse_m"]],
                                               "samples": [merged["n"], cres["merge"]["samples"]]}
    ok_internal = merged["n"] == cres["merge"]["samples"]
    for k, pairs in oracle["internal"].items():   # C++ YAML prints 6 significant digits
        for name, (ours, ref) in pairs.items():
            if name != "samples":
                ok_internal &= abs(ours - ref) <= 5e-6 * abs(ref) + 1e-12
    ev = evo_oracle(trajs, E_all, G_all)
    oracle["evo"] = {"tool": ev.pop("evo_version"), "pairs": {}}
    ok_evo = True
    for name, ours in (("ape_translation_rmse_m", M["ate_rmse_m"]), ("ape_rotation_rmse_rad", M["ate_rotation_rmse_rad"]),
                       ("rpe_translation_rmse_m", M["rpe_translation_rmse_m"]), ("rpe_rotation_rmse_rad", M["rpe_rotation_rmse_rad"])):
        rel = abs(ours - ev[name]) / abs(ev[name])
        oracle["evo"]["pairs"][name] = {"harness": ours, "evo": ev[name], "rel_diff": rel}
        ok_evo &= rel <= ORACLE_REL_TOL
    oracle["verdict"] = {"internal": bool(ok_internal), "evo": bool(ok_evo), "rel_tol": ORACLE_REL_TOL}
    if not (ok_internal and ok_evo):
        (Path(a.out)).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "oracle.yaml").write_text(yaml.safe_dump(oracle, sort_keys=False))
        raise SystemExit("oracle disagreement (see oracle.yaml)")

    # --- registration (1.6) ---------------------------------------------------------------------------------
    cand = read_csv(inp.use(collab / "candidates.csv"))
    probe = read_csv(inp.use(collab / "negative_probe.csv"))
    exch = [r for r in cand if r["result"] == "verified"]
    acc_probe = [r for r in probe if r["result"] == "verified"]
    accepted = exch + acc_probe
    correct = [r for r in accepted if r["edge_correct"] == "1"]
    M["inter_agent_rpe_m"] = tr.rms([float(r["edge_t_err_m"]) for r in exch])
    M["closure_precision"] = len(correct) / len(accepted)
    M["false_closure_rate"] = sum(r["edge_correct"] != "1" for r in exch) / len(exch)
    reg_note = {"exchanged_edges": len(exch), "probe_pairs": len(probe), "probe_accepted": len(acc_probe),
                "accepted_total": len(accepted), "correct_total": len(correct),
                "false_accepts_in_probe": sum(r["edge_correct"] != "1" for r in acc_probe)}

    # --- mapping (1.7 + 1.8) -----------------------------------------------------------------------------------
    prep, score = ly(inp.use(recon / "prepare.yaml")), ly(inp.use(recon / "score.yaml"))
    meval = ly(inp.use(mission / "eval.yaml"))
    M["chamfer_m"] = prep["geometry"]["chamfer_m"]
    M["completeness"] = prep["geometry"]["completeness"]
    M["coverage"] = meval["map"]["coverage"]
    M["heldout_psnr_db"], M["heldout_ssim"], M["heldout_lpips"] = (score["heldout_psnr_db"], score["heldout_ssim"],
                                                                    score["heldout_lpips"])

    # --- communication (1.6 ByteLedger) ----------------------------------------------------------------------------
    b = cres["bytes"]
    wire = lambda cls: b[cls]["wire_bytes"] if cls in b else 0
    import cv2
    slab = cv2.imread(str(inp.use(mission / "slab.png")), cv2.IMREAD_UNCHANGED)
    cell = ly(inp.use(scene / "scene.json"))["esdf"]["cell_m"]
    area = float((slab == 1).sum()) * cell * cell
    kfs = cres["keyframes"]["agent0"] + cres["keyframes"]["agent1"]
    M["bytes_total"] = b["total_wire_bytes"]
    M["bytes_per_keyframe"] = b["total_wire_bytes"] / kfs
    M["bytes_per_verified_loop"] = b["total_wire_bytes"] / len(exch)
    M["bytes_per_map_area_m2"] = b["total_wire_bytes"] / area
    M["descriptor_bytes"] = wire("GLOBAL_DESCRIPTOR")
    M["local_feature_bytes"] = wire("MATCH_REQUEST") + wire("MATCH_RESPONSE")
    M["pose_graph_bytes"] = wire("POSE_EDGE") + wire("POSE_GRAPH")

    # --- mission (1.8) -----------------------------------------------------------------------------------------
    gt = np.load(inp.use(scene / "scene_gt.npz"))
    routes = [json.loads(l) for l in inp.use(mission / "routes.jsonl").read_text().splitlines()]
    egress = [r for r in routes if r["kind"] == "egress" and r["found"]]
    M["time_to_first_usable_map_s"] = mres["time_to_first_agent_route_s"]
    M["time_to_first_valid_egress_s"] = mres["time_to_first_valid_route_s"]
    M["route_validity"] = meval["final_egress"]["valid_fraction"]
    M["route_confidence"] = meval["final_egress"]["support_p5"]
    briers = [(r["support_p5"] - route_gt_validity(r["cells"], gt["trav_responder"])) ** 2 for r in egress]
    M["route_confidence_calibration"] = float(np.mean(briers)) if briers else float("nan")
    calib_note = {"routes": len(egress), "distinct_routes": len({json.dumps(r["cells"]) for r in egress}),
                  "status": "[PROXY] Brier score of support_p5 vs GT validity within ONE run: not a calibration estimate"}
    gt_len = sum(path_length(x["G"]) for x in trajs)
    est_len = sum(path_length(x["E"]) for x in trajs)
    M["distance_travelled_m"] = gt_len
    M["energy_proxy"] = gt_len + ENERGY_M_PER_RAD * sum(yaw_travel(x["G"]) for x in trajs)

    comparison = meval.get("comparison", {})
    report = {
        "chain": {"collab_run": collab.name, "recon_run": recon.name, "mission_run": mission.name,
                  "scene": ly(scene / "scene.json")["scene_id"]},
        "metrics": {k: (float(v) if v is not None else None) for k, v in M.items()},
        "definitions": DEFINITIONS,
        "notes": {"registration": reg_note, "route_confidence_calibration": calib_note,
                  "distance_travelled_m": {"gt_m": gt_len, "estimated_m": est_len,
                                           "definition": "GT path length of both agents (the estimate beside it)"},
                  "energy_proxy": f"[PROXY] GT path length + {ENERGY_M_PER_RAD} m/rad x GT yaw travel, until 4.3 defines it",
                  "ate": "end-of-run merged estimate: agent 1's poses placed through T_O0_O1 retroactively",
                  "map_area_m2": area, "keyframes": kfs, "mission_comparison": comparison},
        "gates": gates,
    }
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "report.yaml").write_text("# Ariadne evaluation report (design HARNESS_1.9). Generated.\n" +
                                     yaml.safe_dump(report, sort_keys=False))
    (out / "oracle.yaml").write_text("# Oracles (design HARNESS_1.9). Generated.\n" + yaml.safe_dump(oracle, sort_keys=False))
    reg = ly(inp.use(Path(a.registry)))
    (out / "registry.yaml").write_text(Path(a.registry).read_text())
    inputs = {"files": inp.files, "input_manifests_git_dirty": dirty,
              "harness": {"git_commit": os.environ.get("ARIADNE_GIT_COMMIT", "unknown"),
                          "git_dirty": os.environ.get("ARIADNE_GIT_DIRTY", "unknown"),
                          "image_id": os.environ.get("ARIADNE_IMAGE_ID", "unknown"), "command": " ".join(sys.argv)}}
    (out / "inputs.yaml").write_text(yaml.safe_dump(inputs, sort_keys=False))
    # figure data, so `render` needs no inputs beyond OUT
    A = merged["alignment"]   # estimate (agent 0 odometry frame) -> world, the ATE alignment
    np.savez_compressed(out / "figures" / "data.npz",
                        **{f"E{x['agent']}": x["E"][:, :3, 3] @ A[:3, :3].T + A[:3, 3] for x in trajs}, **{f"G{x['agent']}": x["G"][:, :3, 3] for x in trajs},
                        **{f"t{x['agent']}": x["t"] for x in trajs},
                        **{f"err{x['agent']}": merged["errors_m"][sum(len(y["t"]) for y in trajs[:x["agent"]]):][:len(x["t"])] for x in trajs})
    (out / "figures" / "bytes.json").write_text(json.dumps({k: v for k, v in b.items() if isinstance(v, dict)}))
    import shutil
    shutil.copy(mission / "map_vs_gt.png", out / "figures" / "map_vs_gt.png")
    print(yaml.safe_dump({"metrics": report["metrics"], "gates": gates, "oracle_verdict": oracle["verdict"]}, sort_keys=False))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="analysis.report")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    for k in ("--collab-run", "--recon-run", "--mission-run", "--scene", "--registry", "--out"):
        c.add_argument(k, required=True)
    c.add_argument("--mission-compare", nargs="*", default=[])
    c.add_argument("--allow-dirty", action="store_true")
    r = sub.add_parser("render")
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "compute":
        return compute(a)
    from .report_render import render
    return render(Path(a.out))


if __name__ == "__main__":
    sys.exit(main())
