# Serialization.PreTaskConsensusBytesUnchanged on arm64: root cause (for M, owner of the serializer; fix post-freeze)

**Dumps** (`infra/eval/wire_dump/`, same recipe on both; built with test_serialization.cpp's own compile flags):
- `wire_dump_x86_64.txt`: local ubuntu:24.04, g++ 13.3. All four messages equal the test's goldens.
- `wire_dump_aarch64.txt`: run `20261024T074030Z-cpuarm-d60e34`, c8g spot, 2 min. Reproduces the failing ctest
  values of run `…f85185`.

| message | x86 | arm64 | first differing offset |
|---|---|---|---|
| heartbeat | 238 B 0x5d85433fd5441b89 | 238 B 0xc8b28c6f9ba11f89 | 202 (6 bytes differ) |
| loop_edge | 194 B 0x49bb277520bc078b | identical | — |
| pose_graph | 417 B 0xb6f2ab38c40f0161 | 419 B 0xdc8dcbc2eeaebf27 | 13 (zstd body; 366 bytes differ) |
| ack | 110 B 0x011ede3f1e7c8e79 | 110 B 0x052a83640e07845d | 82 (7 bytes differ) |

**Cause: unsequenced FlatBuffers builder calls in function arguments (D123's bug class, in `cpp/`).**
`FlatBufferBuilder` is stateful: the byte layout follows the order in which children are built. Two calls build two
children inside one call's argument list:
- `serializer.hpp:916` `wire::CreateAck(fbb, fbb.CreateVector(a.done), fbb.CreateVector(partial))`;
- `serializer.hpp:511` `wire::CreatePoseGraph(fbb, fbb.CreateVector(nodes), fbb.CreateVector(edges), …)`.

GCC x86-64 evaluates the arguments right to left, aarch64 left to right. In the ack dump at offset 82:
- x86 lays out `done` [3: 1,2,3] then the empty `partial`;
- arm64 lays out the empty `partial` first (the field offsets read 08/14 on x86 against 0c/04 on arm64).

The heartbeat embeds an Ack, hence offset 202. The pose graph's two vectors are swapped before zstd.
`BuildEdge` builds its children into named locals first, which is why the loop edge is identical.

**Consequences:**
- Decoding is unaffected: every round trip passes on both architectures, because FlatBuffers reads by offset.
- Byte counts can differ by alignment padding (pose graph +2 B after zstd).
- The goldens encode x86's order.

**Fix (M):**
- hoist both children into named locals in source order, as `BuildEdge` does;
- decide the canonical order;
- re-record the goldens;
- consider extending `check_rng_sequencing.py`'s rule to `fbb.Create*` / `Build*(fbb, …)` calls in one argument
  list. A scan finds exactly these 2 statements in `serializer.hpp`.
