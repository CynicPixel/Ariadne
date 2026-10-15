# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# ariadne-analysis (design HARNESS_1.9): the evaluation harness + its oracle.
# The recon image's pinned stack plus evo 1.31.1, the trajectory oracle that
# reproduced 1.5's ATE exactly (D36), and matplotlib for the report figures.
# Pins: base by digest, every package by exact version. Repo at /work.
# ---------------------------------------------------------------------------
FROM python@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f
RUN pip install --no-cache-dir --root-user-action=ignore \
      numpy==2.5.4 scipy==1.18.1 opencv-python-headless==5.0.0.93 pyyaml==6.0.3 pytest==9.1.1 evo==1.31.1
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/work/python:/work/python/reconstruction MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl
WORKDIR /work
