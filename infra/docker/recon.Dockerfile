# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# ariadne-recon (arch §18.2, design reconstruction.md#first-reconstruction-loop): the Python reconstruction
# service's LOCAL (CPU) stages — spool validation, fusion, Gaussian init,
# geometry metrics, tests. No torch and no gsplat here: rasterization needs
# CUDA (SMOKE §4o) and runs remotely (Kaggle now, AWS GPU later, D44), from a
# render package this image writes. The repo is bind-mounted at /work.
# Pins: base by digest (resolved 2026-10-15), every package by exact version.
# ---------------------------------------------------------------------------
FROM python@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f
RUN pip install --no-cache-dir --root-user-action=ignore \
      numpy==2.5.4 scipy==1.18.1 opencv-python-headless==5.0.0.93 pyyaml==6.0.3 pytest==9.1.1
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/work/python/reconstruction
WORKDIR /work
