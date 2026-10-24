// Renders the trained splat for the demo video (6.x, M 2026-10-24) through orbit.html in a headless Chrome, over the
// DevTools protocol. Node >= 22 built-ins only (fetch, WebSocket); no npm packages.
//
//   node render_orbit.mjs --cdp http://127.0.0.1:9222 --page 'http://127.0.0.1:8765/orbit.html?base=splat&w=1920&h=1080'
//        --out DIR [--seconds 9] [--fps 30] [--elev-deg 35]
//
// DIR/orbit/frame_NNNN.png   a closed orbit around the mission world's centre (seconds x fps frames)
// DIR/stills/still_K.png      4 high viewpoints (60 deg elevation) no ground agent ever occupied
// DIR/align/agent_K.png, align/merged.png   one overview pose: each agent's submaps alone, then all together
// DIR/poses.json              every image's camera (pos, target, fov_deg, agents shown), the GL renderer, the label
// Poses are in mission world W (metres, z-up), the frame the table's T_W_anchor places submaps in.
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const cdp = arg("--cdp", "http://127.0.0.1:9222"), page = arg("--page"), out = arg("--out");
const seconds = +arg("--seconds", 9), fps = +arg("--fps", 30), elev = (+arg("--elev-deg", 35)) * Math.PI / 180;
if (!page || !out) { console.error("usage: --page URL --out DIR"); process.exit(2); }
const LABEL = "trained Gaussian splat, novel viewpoints (rendered)";

const target = await (await fetch(`${cdp}/json/new?${encodeURIComponent("about:blank")}`, { method: "PUT" })).json();
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((ok, bad) => { ws.onopen = ok; ws.onerror = bad; });
let id = 0;
const pending = new Map();
const pageErrors = [];
ws.onmessage = (m) => {
  const r = JSON.parse(m.data);
  if (r.id && pending.has(r.id)) { pending.get(r.id)(r); pending.delete(r.id); }
  else if (r.method === "Runtime.exceptionThrown") pageErrors.push(r.params.exceptionDetails.exception?.description || r.params.exceptionDetails.text);
  else if (r.method === "Log.entryAdded" && r.params.entry.level === "error") pageErrors.push(`${r.params.entry.text} ${r.params.entry.url || ""}`);
};
const send = (method, params = {}) => new Promise((ok, bad) => {
  const i = ++id;
  pending.set(i, (r) => (r.error ? bad(new Error(`${method}: ${JSON.stringify(r.error)}`)) : ok(r.result)));
  ws.send(JSON.stringify({ id: i, method, params }));
});
const evaluate = async (expr) => {
  const r = await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error(`${expr}: ${JSON.stringify(r.exceptionDetails)}`);
  return r.result.value;
};

const u = new URL(page);
const W = +(u.searchParams.get("w") || 1920), H = +(u.searchParams.get("h") || 1080);
await send("Page.enable");
await send("Runtime.enable");
await send("Log.enable");
await send("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: 1, mobile: false });
await send("Page.navigate", { url: page });
for (let t = 0; !(await evaluate("typeof window.renderView === 'function'")); t++) {   // the module loads from the CDN
  if (t > 180) { console.error("page API never appeared:", pageErrors.join(" | ")); process.exit(1); }
  await new Promise((r) => setTimeout(r, 500));
}
const info = await evaluate("window.loaded");
const gl = await evaluate("window.glInfo()");
console.log(`loaded ${info.submaps} submaps, agents ${info.agents}; GL: ${gl}`);

const { min, max } = info.bounds;
const c = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, min[2]];
const R = 0.75 * Math.max(max[0] - min[0], max[1] - min[1]) + 3;   // anchors bound the paths; the map extends past
const look = [c[0], c[1], c[2] + 0.3];
const poses = { label: LABEL, gl, frame: "mission world W (m, z-up)", size: [W, H], table_label: info.label,
                bounds: info.bounds, orbit: [], stills: [], align: [] };
const shoot = async (file, view) => {
  await evaluate(`window.renderView(${JSON.stringify(view)})`);
  const { data } = await send("Page.captureScreenshot", { format: "png" });
  writeFileSync(file, Buffer.from(data, "base64"));
};
const at = (az, el, r) => [c[0] + r * Math.cos(el) * Math.cos(az), c[1] + r * Math.cos(el) * Math.sin(az), c[2] + r * Math.sin(el)];

for (const d of ["orbit", "stills", "align"]) mkdirSync(join(out, d), { recursive: true });
const n = Math.round(seconds * fps);
for (let i = 0; i < n; i++) {   // closed loop: frame n would equal frame 0
  const v = { pos: at(2 * Math.PI * i / n, elev, R), target: look, fov_deg: 50, agents: null };
  const f = `orbit/frame_${String(i).padStart(4, "0")}.png`;
  await shoot(join(out, f), v);
  poses.orbit.push({ file: f, ...v });
  if (i % 30 === 0) console.log(`orbit ${i}/${n}`);
}
for (let k = 0; k < 4; k++) {   // 60 deg up: no ground agent's camera was ever there
  const v = { pos: at(Math.PI / 4 + k * Math.PI / 2, Math.PI / 3, 0.8 * R), target: look, fov_deg: 50, agents: null };
  const f = `stills/still_${k}.png`;
  await shoot(join(out, f), v);
  poses.stills.push({ file: f, ...v });
}
const over = { pos: at(-Math.PI / 2, 1.25, 1.1 * R), target: look, fov_deg: 50 };   // near-top-down overview
for (const a of info.agents) {
  const f = `align/agent_${a}.png`;
  await shoot(join(out, f), { ...over, agents: [a] });
  poses.align.push({ file: f, ...over, agents: [a] });
}
await shoot(join(out, "align/merged.png"), { ...over, agents: null });
poses.align.push({ file: "align/merged.png", ...over, agents: null });
writeFileSync(join(out, "poses.json"), JSON.stringify(poses, null, 1));
console.log(`wrote ${n} orbit frames, 4 stills, ${info.agents.length + 1} align images -> ${out}`);
ws.close();
process.exit(0);
