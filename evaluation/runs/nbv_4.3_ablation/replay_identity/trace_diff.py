import json, sys
VOL = {"mode", "executed", "plan_ms", "wall_ms", "snapshot_png", "stamp_ns", "elapsed_ms", "timing"}
def load(p):
    out = {}
    for l in open(p):
        r = json.loads(l)
        if "agent" in r and "t_s" in r: out[(r["t_s"], r["agent"])] = r
    return out
def strip(x):
    if isinstance(x, dict): return {k: strip(v) for k, v in x.items() if k not in VOL and not k.endswith("_ms")}
    if isinstance(x, list): return [strip(v) for v in x]
    return x
def diff(a, b, path=""):
    if type(a) != type(b): return [f"{path}: {str(a)[:80]} != {str(b)[:80]}"]
    if isinstance(a, dict):
        d = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b: d.append(f"{path}.{k}: only in {'new' if k in b else 'old'}"); continue
            d += diff(a[k], b[k], f"{path}.{k}")
        return d
    if isinstance(a, list):
        if len(a) != len(b): return [f"{path}: len {len(a)} != {len(b)}"]
        d = []
        for i, (x, y) in enumerate(zip(a, b)): d += diff(x, y, f"{path}[{i}]")
        return d
    return [] if a == b else [f"{path}: {a} != {b}"]
A, B = load(sys.argv[1]), load(sys.argv[2])
print(f"records old {len(A)} new {len(B)}")
for k in sorted(set(A) | set(B)):
    if k not in A or k not in B: print(k, "missing in", "old" if k not in A else "new"); continue
    d = diff(strip(A[k]), strip(B[k]))
    print(k, "IDENTICAL" if not d else f"{len(d)} diffs"); [print("   ", x) for x in d[:12]]
    if not d: continue
    print("    old command:", A[k].get("command"), "chosen", A[k].get("chosen"))
    print("    new command:", B[k].get("command"), "chosen", B[k].get("chosen"))
