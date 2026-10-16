# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# ariadne-oracle (arch §11.6, §37.5; design POSE_GRAPH_2.md §5): TEST ORACLES
# ONLY, never runtime (CLAUDE.md §1). GTSAM 4.3.0 (BSD, PyPI upload 2026-09-19).
# Pins: base by digest, every package exact. Repo at /work.
# ---------------------------------------------------------------------------
FROM python@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f
RUN pip install --no-cache-dir --root-user-action=ignore gtsam==4.3.0 numpy==2.5.4 pyyaml==6.0.3
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/work/python
WORKDIR /work
