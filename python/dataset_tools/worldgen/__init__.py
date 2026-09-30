"""Procedural world generation (arch §5.5); design: docs/design/SIM_1.1-1.3_WORLDGEN.md.

Pure numpy + pyyaml so it runs unchanged in the pinned ariadne-sim image, which is
the authoritative place worlds are generated.
"""
GENERATOR_VERSION = 1   # bump on ANY change that alters output for a fixed (seed, preset)
