# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# ariadne-console (arch §29.3, design CONSOLE_1.10): the operator console stub's
# FastAPI backend + static frontend. Pins: base by digest, every package exact.
# The repo is bind-mounted at /work (read-only is enough).
# ---------------------------------------------------------------------------
FROM python@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f
RUN pip install --no-cache-dir --root-user-action=ignore \
      numpy==2.5.4 pyyaml==6.0.3 fastapi==0.143.2 uvicorn==0.54.0 websockets==17.2 httpx==0.28.1 pytest==9.1.1
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/work/python
WORKDIR /work
