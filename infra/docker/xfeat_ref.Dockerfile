# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# ariadne-xfeat-ref (D15, design PERCEPTION_2.1): the PyTorch REFERENCE for the
# XFeat port. Offline tool only, never runtime: it (1) exports upstream's
# weights (verlab/accelerated_features, Apache-2.0) to Ariadne's tensor blob and
# (2) runs upstream's own model on fixed inputs so the OpenCV-primitive forward
# pass can be checked against it. CPU wheels; pins: base by digest, packages exact.
# ---------------------------------------------------------------------------
FROM python@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f
RUN pip install --no-cache-dir --root-user-action=ignore --index-url https://download.pytorch.org/whl/cpu \
      torch==2.5.1+cpu \
 && pip install --no-cache-dir --root-user-action=ignore numpy==2.1.3 pyyaml==6.0.3 opencv-python-headless==5.0.0.93 tqdm==4.67.1
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /work
