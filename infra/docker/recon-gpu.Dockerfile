# ---------------------------------------------------------------------------
# ariadne-recon-gpu: the live splat service's GPU backend (reconstruction.md §1, D190/D196/D231) for the judge box. The CPU
# recon image's packages (recon.Dockerfile) plus torch + gsplat, with gsplat's CUDA kernels COMPILED AHEAD OF TIME.
# gsplat's PyPI wheel is pure Python and JIT-compiles its kernels on first use (what the Kaggle smokes did:
# "pypi source (JIT CUDA build)"); a judge box must not compile at every start. Built from the 1.5.3 sdist, whose
# setup.py builds the CUDAExtension `gsplat.csrc`; gsplat's backend imports gsplat.csrc first and JIT-compiles only
# if that import fails, so the `from gsplat import csrc` checks below prove the image never compiles at run time.
# Pins = the stack validated on the Kaggle T4 (D196/D198): torch 2.11.0+cu128, gsplat 1.5.3, CUDA 12.8. Python 3.12
# (Ubuntu 24.04's; Kaggle ran 3.13 - the cu128 torch wheel exists for both). Arch list: T4 (g4dn) 7.5, A10G (g5)
# 8.6, L4 (g6) 8.9. torch's cu128 wheel carries the CUDA runtime libraries; the driver comes from the host (NVIDIA
# container toolkit). Bases by digest (resolved 2026-10-24); every package by exact version.
# Built on an AWS CPU slot (infra/scripts/remote_recon_gpu_build.sh), shipped as a tarball in S3, never pushed to ECR.
# ---------------------------------------------------------------------------
FROM nvidia/cuda@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66 AS build
# = nvidia/cuda:12.8.1-devel-ubuntu24.04 (nvcc for the ahead-of-time build)
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-dev ninja-build \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir setuptools==80.9.0 wheel==0.45.1 ninja==1.13.0 numpy==2.5.4 jaxtyping==0.3.3 \
       rich==14.2.0
ENV TORCH_CUDA_ARCH_LIST="7.5;8.6;8.9" MAX_JOBS=16
RUN pip install --no-cache-dir --no-build-isolation --no-binary gsplat gsplat==1.5.3 \
    && python -c "from gsplat import csrc; print('gsplat.csrc AOT:', csrc.__file__)"
# Every package goes into the venv HERE: the runtime base has no CA bundle, so it never runs pip (first build,
# run 20261023T191330Z-cpux86-763d65, failed exactly so).
RUN pip install --no-cache-dir scipy==1.18.1 opencv-python-headless==5.0.0.93 pyyaml==6.0.3 pytest==9.1.1

FROM ubuntu@sha256:68ee22004a9109b44973313b02ff7b1de250f1de0f5ecb628d73f579aaff33ac
# = ubuntu:24.04 (runtime: the venv's python3 is this image's /usr/bin/python3.12)
RUN apt-get update && apt-get install -y --no-install-recommends python3 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN python -c "import torch, gsplat, scipy, cv2, yaml; from gsplat import csrc; print('torch', torch.__version__, 'gsplat', gsplat.__version__, 'csrc', csrc.__file__, 'cv2', cv2.__version__)"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/work/python/reconstruction
WORKDIR /work
