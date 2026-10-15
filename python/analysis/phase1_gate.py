"""The Phase 1 exit gate verdict (plan "Phase 1 testing / exit gate", design PHASE1_GATE).

  python -m analysis.phase1_gate --gate G --big BIG --config configs/benchmark/phase1_gate.yaml

Reads what infra/scripts/phase1.sh produced in G (and the large outputs in BIG), writes G/gate.yaml, prints one
line per check and `VERDICT PASS|FAIL`. Checks:

  i1           the I1 exit condition: two agents explore and build local maps, merge via >= 1 verified inter-agent
               closure, a splat is rendered, an exit route is produced -- all in this one run
  contracts    every §4 contract (and the §18.2 bundle) counted at every hop, from contracts.yaml
  stages       ctest, the protocol tests, fault identity, §30.2 report check, console chain check
  determinism  this run vs the committed per-item evidence, file by file, with named exclusions only
"""
from __future__ import annotations

import argparse
import hashlib
import math
import re
from pathlib import Path

import yaml


def ly(p: Path):
    return yaml.safe_load(p.read_text())


def one(d: Path, pattern: str) -> Path:
    """The single run directory in d whose name matches the regex `pattern` (fully)."""
    m = sorted(p for p in d.iterdir() if p.is_dir() and re.fullmatch(pattern, p.name))
    if len(m) != 1:
        raise SystemExit(f"phase1_gate: expected one {d}/{pattern}, found {[x.name for x in m]}")
    return m[0]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class NoAliases(yaml.SafeDumper):
    def ignore_aliases(self, data):   # gate.yaml is read by people: no &id001 / *id001
        return True


class Gate:
    def __init__(self):
        self.rows: list[dict] = []

    def check(self, group: str, name: str, ok: bool, detail) -> None:
        self.rows.append({"group": group, "check": name, "ok": bool(ok), "detail": detail})

    @property
    def ok(self) -> bool:
        return all(r["ok"] for r in self.rows)


def finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def check_i1(g: Gate, C: Path, F: Path, RC: Path, big: Path, M: Path, M0: Path, M1: Path) -> None:
    res = ly(C / "results.yaml")
    kf = res["keyframes"]
    g.check("i1", "two agents explore (keyframes per agent)", kf["agent0"] > 0 and kf["agent1"] > 0, kf)
    e0, e1 = ly(M0 / "eval.yaml")["map"], ly(M1 / "eval.yaml")["map"]
    g.check("i1", "each agent builds a local map alone (coverage of the GT free space)",
            e0["coverage"] > 0 and e1["coverage"] > 0,
            {"agent0_alone": e0["coverage"], "agent1_alone": e1["coverage"]})
    edges, merge = res["edges"], res["merge"]
    g.check("i1", "merge via >= 1 verified inter-agent closure",
            merge["ok"] == 1 and edges["sent"] >= 1 and edges["received_by_agent1"] == edges["sent"]
            and edges["correct"] >= 1 and finite(merge["merged_two_agent_ate_rmse_m"]),
            {"edges_sent": edges["sent"], "edges_correct_vs_gt": edges["correct"],
             "merged_ate_rmse_m": merge["merged_two_agent_ate_rmse_m"]})
    score = ly(RC / "score.yaml")
    renders = big / "render" / "renders.npz"
    g.check("i1", "a splat is rendered (remote GPU renders scored on held-out views)",
            renders.is_file() and score["views"] > 0 and finite(score["heldout_psnr_db"]),
            {"views": score["views"], "psnr_db": score["heldout_psnr_db"], "ssim": score["heldout_ssim"],
             "lpips": score["heldout_lpips"], "renders_sha256": sha(renders) if renders.is_file() else None})
    mres, mev = ly(M / "results.yaml"), ly(M / "eval.yaml")
    fe, first = mev["final_egress"], mev["first_valid_egress"]
    g.check("i1", "an exit route is produced, valid against GT at rho_responder",
            mres["time_to_first_valid_route_s"] >= 0 and fe["found"] and fe["valid"] and first["valid"],
            {"t_merge_s": mres["t_merge_s"], "first_valid_route_s": mres["time_to_first_valid_route_s"],
             "final_length_m": fe["length_m"], "gt_geodesic_m": fe["gt_geodesic_m"], "exit": fe["exit"]})
    manifests = [C / "manifest.yaml", F / "manifest.yaml", RC / "manifest_prepare.yaml", RC / "manifest_score.yaml",
                 M / "manifest.yaml", M0 / "manifest.yaml", M1 / "manifest.yaml"]
    commits = {str(p.relative_to(C.parent.parent)): ly(p).get("git_commit") if p.is_file() else None
               for p in manifests}
    g.check("i1", "one run: every stage's manifest records the same commit",
            len(set(commits.values())) == 1 and None not in commits.values(), commits)


def check_contracts(g: Gate, C: Path) -> None:
    c, res = ly(C / "contracts.yaml"), ly(C / "results.yaml")
    for k in (0, 1):
        a = c[f"agent{k}"]
        kfs = res["keyframes"][f"agent{k}"]
        g.check("contracts", f"§4.1 Frame delivered (agent {k})", a["frame_4_1"]["delivered"] > 0, a["frame_4_1"])
        g.check("contracts", f"§4.2 Keyframe records complete (agent {k})",
                a["keyframe_4_2"]["records"] == kfs and a["keyframe_4_2"]["complete"] == kfs, a["keyframe_4_2"])
        b = a["keyframe_bundle_18_2"]
        g.check("contracts", f"§18.2 KeyframeBundle built + spooled (agent {k})",
                b["built"] == kfs and b["spooled"] > 0, b)
    le = c["loop_edge_4_3"]
    g.check("contracts", "§4.3 LoopEdge encoded -> decoded identical -> merge edge received",
            le["encoded"] == res["edges"]["sent"] and le["decoded_identical"] == le["encoded"] > 0
            and le["merge_edge_received"] is True, le)
    br = c["byte_record_4_4"]
    msgs = sum(v["messages"] for v in res["bytes"].values() if isinstance(v, dict))
    g.check("contracts", "§4.4 ByteRecord: one ledger row per message, totals agree",
            br["ledger_rows"] == msgs and br["total_wire_bytes"] == res["bytes"]["total_wire_bytes"], br)
    g.check("contracts", "unset-by-design fields declared", bool(c.get("unset_by_design")), c.get("unset_by_design"))


def check_stages(g: Gate, G: Path, C: Path, F: Path, RP: Path) -> None:
    def txt(p: Path) -> str:
        return p.read_text() if p.is_file() else ""
    ct, pt = txt(G / "ctest.txt"), txt(G / "protocol_tests.txt")
    g.check("stages", "full ctest", "100% tests passed" in ct, ct.strip())
    m = re.search(r"(\d+)% tests passed, (\d+) tests failed out of (\d+)", pt)
    g.check("stages", "protocol tests (CollabProtocol.*, Serialization.*)",
            bool(m) and m.group(2) == "0" and int(m.group(3)) > 0, m.group(0) if m else pt[-200:])
    fi = txt(F / "fault_identity.txt")
    g.check("stages", "duplicate/out-of-order injection is idempotent (fault identity)",
            "VERDICT PASS" in fi and ly(F / "results.yaml")["inject_faults"] == 1,
            [l for l in fi.splitlines() if l.startswith(("fault run", "VERDICT"))])
    rc = txt(RP / "report_check.txt")
    g.check("stages", "every §30.2 metric emits a number (report check)", "VERDICT PASS" in rc,
            rc.strip().splitlines()[-1] if rc else "missing")
    cc = txt(G / "console_check.txt")
    g.check("stages", "console chain check", "VERDICT PASS" in cc, cc.strip().splitlines()[-1] if cc else "missing")


def compare_file(new: Path, ref: Path, excludes: list[dict]) -> dict:
    if not ref.exists():
        return {"status": "MISSING_REFERENCE"}
    if not new.exists():
        return {"status": "MISSING"}
    a, b = new.read_bytes(), ref.read_bytes()
    if a == b:
        return {"status": "identical"}
    if not excludes:
        return {"status": "DIFFERENT", "first_diff": first_diff(a, b)}
    pats = [re.compile(e["re"]) for e in excludes]

    def keep(data: bytes) -> tuple[list[str], int]:
        lines = data.decode().splitlines()
        kept = [l for l in lines if not any(p.match(l) for p in pats)]
        return kept, len(lines) - len(kept)
    ka, na = keep(a)
    kb, nb = keep(b)
    if ka == kb:
        return {"status": "identical_except", "excluded": [e["re"] for e in excludes], "lines_excluded": na}
    return {"status": "DIFFERENT", "first_diff": first_diff("\n".join(ka).encode(), "\n".join(kb).encode())}


def first_diff(a: bytes, b: bytes) -> str:
    try:
        la, lb = a.decode().splitlines(), b.decode().splitlines()
    except UnicodeDecodeError:
        return f"binary: {len(a)} vs {len(b)} bytes"
    for i, (x, y) in enumerate(zip(la, lb)):
        if x != y:
            return f"line {i + 1}: {x[:160]!r} != {y[:160]!r}"
    return f"length {len(la)} vs {len(lb)} lines"


def check_determinism(g: Gate, cfg: dict, runs: dict[str, Path], RC: Path) -> dict:
    root = Path(cfg["reference_root"])
    out = {}
    for key, files in cfg["compare"].items():
        for spec in files:
            for e in spec.get("exclude", []):
                if e.get("why") not in cfg["reasons"]:
                    raise SystemExit(f"phase1_gate: exclusion {e} on {key}/{spec['file']} has no named reason")
    for key, files in cfg["compare"].items():
        ref_dir = root / cfg["reference"][key]
        for spec in files:
            r = compare_file(runs[key] / spec["file"], ref_dir / spec["file"], spec.get("exclude", []))
            r["reference"] = str(ref_dir / spec["file"])
            out[f"{key}/{spec['file']}"] = r
            g.check("determinism", f"{key}/{spec['file']}", r["status"] in ("identical", "identical_except"),
                    r["status"] if r["status"] != "DIFFERENT" else r)
        if key in runs:
            new_only = sorted(p.name for p in runs[key].iterdir() if not (ref_dir / p.name).exists())
            if new_only:
                out[f"{key}/(new, not compared)"] = new_only

    def sums(p: Path) -> dict:
        d = {}
        for line in p.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            h, name = line.split(None, 1)
            d[Path(name).name] = h
        return d
    mine, ref = sums(RC / "checksums.sha256"), sums(root / cfg["reference"]["recon"] / "checksums.sha256")
    for name in cfg["checksums"]:
        ok = name in mine and mine.get(name) == ref.get(name)
        out[f"recon/{name} (sha256)"] = {"status": "identical" if ok else "DIFFERENT", "gate": mine.get(name),
                                         "reference": ref.get(name)}
        g.check("determinism", f"recon/{name} sha256", ok, mine.get(name))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--big", required=True)
    ap.add_argument("--config", required=True)
    a = ap.parse_args(argv)
    G, big, cfg = Path(a.gate), Path(a.big), ly(Path(a.config))
    sc = G / "sim_collab"
    F = one(sc, r"[0-9a-f]+-faults-\d{8}T\d{6}Z")
    C = one(sc, r"[0-9a-f]+-\d{8}T\d{6}Z")
    RC, RP = G / "recon" / C.name, G / "report" / C.name
    md = G / "mission"
    M = one(md, r"[0-9a-f]+-\d{8}T\d{6}Z")
    M0, M1 = one(md, r"[0-9a-f]+-agent0only-\d{8}T\d{6}Z"), one(md, r"[0-9a-f]+-agent1only-\d{8}T\d{6}Z")
    runs = {"sim_collab": C, "sim_collab_faults": F, "recon": RC, "mission": M, "mission_agent0": M0,
            "mission_agent1": M1, "report": RP}
    g = Gate()
    check_i1(g, C, F, RC, big, M, M0, M1)
    check_contracts(g, C)
    check_stages(g, G, C, F, RP)
    det = check_determinism(g, cfg, runs, RC)
    doc = {"gate": cfg["name"], "run": G.name, "verdict": "PASS" if g.ok else "FAIL",
           "checks": g.rows, "determinism": det,
           "chain": {k: str(v) for k, v in runs.items()}}
    (G / "gate.yaml").write_text("# Phase 1 exit gate (design PHASE1_GATE). Generated; do not hand-edit.\n"
                                 + yaml.dump(doc, Dumper=NoAliases, sort_keys=False, width=120, allow_unicode=True))
    for r in g.rows:
        print(f"{'PASS' if r['ok'] else 'FAIL'}  [{r['group']}] {r['check']}")
    print("VERDICT", "PASS" if g.ok else "FAIL")
    return 0 if g.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
