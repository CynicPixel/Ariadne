#!/usr/bin/env python3
"""D123: fail on unsequenced random draws in a parenthesised call.

`f(n(g), n(g))` evaluates its arguments in an unspecified order (C++ [expr.call]), and the order differs between GCC
x86-64 (right to left) and aarch64 (left to right). So the same seed builds a different fixture on each architecture.
That is what made GravityPgo pass on x86 and fail on arm64 (evaluation/runs/arm64/20261023T233147Z-cpuarm-1e25b3).
A braced list `T{n(g), n(g)}` is evaluated left to right ([dcl.init.list]), so it is the fix.

Rule (line-based, as the 44-site sweep was; see `sites`): two draws `dist(gen)` that one call or one operator
combines. Generator names: g, gen, rng, engine, eng.
Usage: check_rng_sequencing.py [PATH...]   (default: tests; the runtime draws from explicit Philox streams); exit 1 and one line per site if any is found.
"""
import re
import sys
from pathlib import Path

DRAW = re.compile(r"\b[A-Za-z_]\w*\s*\(\s*(?:g|gen|rng|engine|eng)\s*\)")


def sites(line: str):
    """Yield the column of each unsequenced group on the line.

    A draw unit is a direct draw, or a nested bracket group that holds any draw. Two units are unordered when they
    sit (a) anywhere in one `( )` group: call arguments, or operands inside it; or (b) in one comma-separated element
    of a `{ }` list or of the statement itself, i.e. operands of one operator, e.g. `RandomAxis(g) * ang(g)` (an
    overloaded operator is a call). Braced-list elements and declarators are sequenced, so units in different
    elements are fine.
    """
    code = line.split("//", 1)[0]
    draws = {m.start() for m in DRAW.finditer(code)}
    root = ["", -1, [0], False]   # [bracket, column, units per sequenced element, control header]
    stack = [root]
    for i, ch in enumerate(code):
        top = stack[-1]
        if i in draws:
            top[2][-1] += 1
        if ch in "([{":
            header = ch == "(" and re.search(r"\b(for|if|while|switch)\s*$", code[:i]) is not None
            stack.append([ch, i, [0], header])
        elif ch in ")]}" and len(stack) > 1:
            b, col, units, header = stack.pop()
            call = b == "(" and not header
            if (call and sum(units) >= 2) or (not call and max(units) >= 2):
                yield col
            if header:
                stack[-1][2].append(0)   # the statement after a control header starts a new element
            elif sum(units) > 0:
                stack[-1][2][-1] += 1    # the whole group is one unit of its parent
        elif ch == "," or (ch == ";" and len(stack) > 1):
            top[2].append(0)
        elif ch in "?:" and code[i - 1:i + 1] != "::" and code[i:i + 2] != "::":
            top[2].append(0)             # ?: sequences its condition first and evaluates one branch
        elif ch == ";":
            if max(root[2]) >= 2:
                yield 0
            root[2] = [0]
    if len(stack) == 1 and max(root[2]) >= 2:
        yield 0


def main(argv):
    roots = [Path(p) for p in (argv or ["tests"])]
    found = 0
    for root in roots:
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.suffix in {".cpp", ".hpp", ".h", ".cc"})
        for f in files:
            for n, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                for col in sites(line):
                    print(f"{f}:{n}:{col + 1}: unsequenced random draws in one call: {line.strip()}")
                    found += 1
    print(f"check_rng_sequencing: {found} site(s)", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
