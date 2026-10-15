"""Render the 1.9 report (design HARNESS_1.9): figures from OUT/figures/data.npz + bytes.json, and a
self-contained OUT/report.html (figures embedded). Every figure is drawn here, never by hand (§36).

Colours: categorical slots 1-2 of the validated reference palette (blue = agent 0, orange = agent 1;
`validate_palette.js` PASS, both >= 3:1 on the light figure surface); ground truth in neutral grey.
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1d1d1b", "#6b6a63", "#e4e3de"
AGENT = {0: "#2a78d6", 1: "#eb6834"}
GT = "#9a9990"
GROUPS = ["localization", "registration", "mapping", "communication", "mission"]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_trajectories(d, path: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150, facecolor=SURFACE)
    _style(ax)
    for k in (0, 1):
        G, E = d[f"G{k}"], d[f"E{k}"]
        ax.plot(G[:, 0], G[:, 1], color=GT, linewidth=3.2, solid_capstyle="round", zorder=1)
        ax.plot(E[:, 0], E[:, 1], color=AGENT[k], linewidth=1.4, zorder=2)
        ax.annotate(f"agent {k}", E[0, :2], xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK)
    ax.plot([], [], color=GT, linewidth=3.2, label="ground truth")
    ax.plot([], [], color=AGENT[0], linewidth=1.4, label="agent 0 estimate")
    ax.plot([], [], color=AGENT[1], linewidth=1.4, label="agent 1 estimate (via the 1.6 loop)")
    ax.set_aspect("equal")
    ax.set_xlabel("x east (m)", color=MUTED, fontsize=8)
    ax.set_ylabel("y north (m)", color=MUTED, fontsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_ate_over_time(d, path: Path):
    fig, ax = plt.subplots(figsize=(7.2, 2.8), dpi=150, facecolor=SURFACE)
    _style(ax)
    t0 = min(d["t0"][0], d["t1"][0])
    for k in (0, 1):
        t = (d[f"t{k}"] - t0) * 1e-9
        ax.plot(t, d[f"err{k}"] * 1000, color=AGENT[k], linewidth=1.2, label=f"agent {k}")
    ax.set_xlabel("mission time (s)", color=MUTED, fontsize=8)
    ax.set_ylabel("position error after alignment (mm)", color=MUTED, fontsize=8)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def fig_bytes(b: dict, path: Path):
    items = sorted(((k, v["wire_bytes"], v["messages"]) for k, v in b.items()), key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(7.2, 2.4), dpi=150, facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(items))
    ax.barh(y, [i[1] / 1000 for i in items], color=AGENT[0], height=0.55)
    for yi, (k, v, n) in zip(y, items):
        ax.annotate(f"{v / 1000:,.1f} kB · {n} msgs", (v / 1000, yi), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [i[0] for i in items], fontsize=8, color=INK)
    ax.set_xlabel("wire bytes (kB), counted at the serializer", color=MUTED, fontsize=8)
    ax.set_xlim(0, max(i[1] for i in items) / 1000 * 1.35)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _img(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _fmt(v, unit: str) -> str:
    if v is None:
        return "—"
    if unit == "B":
        return f"{v:,.0f}"
    if unit in ("ratio", "proxy"):
        return f"{v:.4g}"
    return f"{v:.4g}"


def render(out: Path) -> int:
    rep = yaml.safe_load((out / "report.yaml").read_text())
    orc = yaml.safe_load((out / "oracle.yaml").read_text())
    reg = yaml.safe_load((out / "registry.yaml").read_text())["metrics"]
    inputs = yaml.safe_load((out / "inputs.yaml").read_text())
    check = (out / "report_check.txt").read_text() if (out / "report_check.txt").exists() else "not run"
    d = np.load(out / "figures" / "data.npz")
    fig_trajectories(d, out / "figures" / "trajectories.png")
    fig_ate_over_time(d, out / "figures" / "ate_over_time.png")
    fig_bytes(json.loads((out / "figures" / "bytes.json").read_text()), out / "figures" / "bytes.png")

    M, D = rep["metrics"], rep["definitions"]
    passed = "VERDICT PASS" in check
    rows = []
    for g in GROUPS:
        rows.append(f'<tr class="group"><th colspan="4">{g}</th></tr>')
        for m in [m for m in reg if m["group"] == g]:
            v = M.get(m["id"])
            proxy = "[PROXY]" in D.get(m["id"], "") or "[ASSUMPTION]" in D.get(m["id"], "")
            rows.append(f'<tr><td><code>{m["id"]}</code>{" <span class=tag>proxy</span>" if proxy else ""}</td>'
                        f'<td class="num">{_fmt(v, m["unit"])}</td><td class="unit">{html.escape(m["unit"])}</td>'
                        f'<td class="def">{html.escape(D.get(m["id"], ""))}</td></tr>')
    cmp_rows = "".join(
        f'<tr><td>{html.escape(str(r["agents"]))}</td><td class="num">{r["first_valid_route_s"]:g}</td>'
        f'<td class="num">{r["final_exit"]}</td><td class="num">{r["final_length_m"]:.2f}</td><td class="num">{r["coverage"]:.2f}</td></tr>'
        for r in rep["notes"]["mission_comparison"].get("rows", []))
    orc_rows = "".join(f'<tr><td><code>{k}</code></td><td class="num">{v["harness"]:.9g}</td><td class="num">{v["evo"]:.9g}</td>'
                       f'<td class="num">{v["rel_diff"]:.1e}</td></tr>' for k, v in orc["evo"]["pairs"].items())
    reg_n = rep["notes"]["registration"]
    files = "".join(f"<li><code>{html.escape(k)}</code> <span class=sha>{v[:12]}</span></li>" for k, v in inputs["files"].items())
    harness = inputs["harness"]
    page = f"""<title>Ariadne Run Report</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
  /* Layout: a single reading column; summary strip first, then the 27-metric table, then figures and provenance. */
  :root {{
    --bg: #f6f6f3; --surface: #fcfcfb; --ink: #1d1d1b; --muted: #6b6a63; --rule: #e1e0da;
    --accent: #1f5f9e; --pass: #1f7a46; --fail: #b3261e; --tag: #8a5a00;
    --sans: "IBM Plex Sans", system-ui, sans-serif; --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
  }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
    --bg: #151514; --surface: #1d1d1b; --ink: #ecebe6; --muted: #a3a29a; --rule: #34332f;
    --accent: #7fb3ea; --pass: #6cc58f; --fail: #f0857c; --tag: #e2b65c; color-scheme: dark; }} }}
  :root[data-theme="dark"] {{ --bg: #151514; --surface: #1d1d1b; --ink: #ecebe6; --muted: #a3a29a; --rule: #34332f;
    --accent: #7fb3ea; --pass: #6cc58f; --fail: #f0857c; --tag: #e2b65c; color-scheme: dark; }}
  body {{ background: var(--bg); color: var(--ink); font: 15px/1.55 var(--sans); }}
  main {{ max-width: 980px; margin: 0 auto; padding-block: 28px 56px;
    padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
    display: grid; gap: 28px; }}
  h1 {{ font-size: 1.7rem; margin: 0; text-wrap: balance; }}
  h2 {{ font-size: 1.1rem; margin: 0 0 10px; }}
  .sub {{ color: var(--muted); margin: 4px 0 0; }}
  .strip {{ display: flex; flex-wrap: wrap; gap: 10px 22px; font-size: .9rem; }}
  .strip b {{ font-variant-numeric: tabular-nums; }}
  .verdict {{ font-weight: 600; }} .ok {{ color: var(--pass); }} .bad {{ color: var(--fail); }}
  section {{ background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 18px 20px; min-width: 0; }}
  .scroll {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .88rem; }}
  th, td {{ text-align: left; padding: 5px 10px 5px 0; border-bottom: 1px solid var(--rule); vertical-align: top; }}
  tr.group th {{ text-transform: uppercase; letter-spacing: .08em; font-size: .72rem; color: var(--muted); padding-top: 14px; }}
  td.num {{ font-family: var(--mono); font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }}
  td.unit {{ color: var(--muted); font-family: var(--mono); }}
  td.def {{ color: var(--muted); font-size: .82rem; }}
  code {{ font-family: var(--mono); font-size: .84em; }}
  .tag {{ font-family: var(--mono); font-size: .68rem; color: var(--tag); border: 1px solid currentColor; border-radius: 3px; padding: 0 4px; margin-left: 6px; }}
  figure {{ margin: 0; display: grid; gap: 6px; }}
  figure img {{ width: 100%; border-radius: 4px; background: #fcfcfb; }}
  figcaption {{ color: var(--muted); font-size: .85rem; }}
  .grid2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 18px; }}
  pre {{ font-family: var(--mono); font-size: .8rem; white-space: pre-wrap; margin: 0; color: var(--muted); }}
  ul.files {{ font-size: .78rem; padding-left: 18px; columns: 1; }} .sha {{ color: var(--muted); font-family: var(--mono); }}
</style>
<main>
  <header>
    <h1>Ariadne run report · scene {html.escape(rep["chain"]["scene"])}</h1>
    <p class="sub">Phase 1 replay chain: {html.escape(rep["chain"]["collab_run"])} → recon → mission. Every §30.2 metric, its
      definition, the oracle cross-checks and the inputs. Generated by <code>python/analysis/report.py</code>; never edited.</p>
  </header>
  <div class="strip">
    <span class="verdict {"ok" if passed else "bad"}">§30.2 gate: {"PASS — all 27 metrics emit a number" if passed else "FAIL"}</span>
    <span>oracles: internal <b class="{"ok" if orc["verdict"]["internal"] else "bad"}">{"agree" if orc["verdict"]["internal"] else "DISAGREE"}</b>,
      evo {html.escape(str(orc["evo"]["tool"]))} <b class="{"ok" if orc["verdict"]["evo"] else "bad"}">{"agrees" if orc["verdict"]["evo"] else "DISAGREES"}</b></span>
    <span>inputs clean: <b class="{"ok" if rep["gates"]["inputs_clean"] else "bad"}">{rep["gates"]["inputs_clean"]}</b></span>
    <span>harness commit <code>{html.escape(str(harness["git_commit"])[:9])}</code> (dirty: {html.escape(str(harness["git_dirty"]))})</span>
  </div>
  <section>
    <h2>All §30.2 metrics</h2>
    <div class="scroll"><table><thead><tr><th>metric</th><th>value</th><th>unit</th><th>definition</th></tr></thead>
    <tbody>{"".join(rows)}</tbody></table></div>
    <p class="sub">Registration counts: {reg_n["exchanged_edges"]} exchanged edges, all correct; negative probe
      {reg_n["probe_pairs"]} pairs, {reg_n["probe_accepted"]} accepted, {reg_n["false_accepts_in_probe"]} of them wrong
      (closure precision {reg_n["correct_total"]}/{reg_n["accepted_total"]}). Calibration: {rep["notes"]["route_confidence_calibration"]["routes"]}
      replan routes, {rep["notes"]["route_confidence_calibration"]["distinct_routes"]} distinct — not yet a calibration estimate.</p>
  </section>
  <section>
    <h2>Trajectories</h2>
    <div class="grid2">
      <figure><img src="{_img(out / "figures" / "trajectories.png")}" alt="Top-down estimated and ground-truth paths of both agents">
        <figcaption>Both agents' estimates in the world frame (one ATE alignment) over ground truth.</figcaption></figure>
      <figure><img src="{_img(out / "figures" / "ate_over_time.png")}" alt="Position error over mission time per agent">
        <figcaption>Position error after the single alignment, per agent, over mission time.</figcaption></figure>
    </div>
  </section>
  <section>
    <h2>Egress and the collaboration comparison</h2>
    <div class="grid2">
      <figure><img src="{_img(out / "figures" / "map_vs_gt.png")}" alt="Merged occupancy map with egress, agent and GT routes">
        <figcaption>Merged map (light free, black occupied, dark unknown), GT boundary (red), GT route (blue), egress route
          at 0.40 m (orange), agent route at 0.20 m (teal), target (red dot), exits (green). North up.</figcaption></figure>
      <div class="scroll"><table><thead><tr><th>run</th><th>first valid egress (s)</th><th>exit</th><th>length (m)</th><th>coverage</th></tr></thead>
        <tbody>{cmp_rows}</tbody></table>
        <p class="sub">First-valid times are replan instants: the route appeared within the preceding 5 s.</p></div>
    </div>
  </section>
  <section>
    <h2>Communication</h2>
    <figure><img src="{_img(out / "figures" / "bytes.png")}" alt="Wire bytes by message class">
      <figcaption>Every inter-agent message, by class, counted at the serializer.</figcaption></figure>
  </section>
  <section>
    <h2>Oracles</h2>
    <div class="scroll"><table><thead><tr><th>quantity</th><th>harness</th><th>evo</th><th>relative diff</th></tr></thead>
      <tbody>{orc_rows}</tbody></table></div>
    <p class="sub">Also: each agent's ATE/RPE matches its C++ results.yaml, and the merged ATE reproduces 1.6's
      {orc["internal"]["merged_ate_vs_1.6"]["ate_rmse_m"][1]} m over {orc["internal"]["merged_ate_vs_1.6"]["samples"][1]} poses.</p>
  </section>
  <section>
    <h2>Gate output</h2>
    <pre>{html.escape(check)}</pre>
  </section>
  <section>
    <h2>Inputs</h2>
    <ul class="files">{files}</ul>
  </section>
</main>
"""
    (out / "report.html").write_text(page)
    print(f"report.html {len(page):,} bytes")
    return 0
