# ariadne-recon-gpu build (6.x POC, D231)

The judge box's live-splat GPU image: `infra/docker/recon-gpu.Dockerfile` at commit `6ecfafc`.

| | |
|---|---|
| Image id | `sha256:a827de5514fe8f9314f17418c9f84edd4a1c53af907a6a6d7ea8456aeb2f0caf` |
| Tarball | `s3://ariadne-836505524290-eun1/runs/images/ariadne-recon-gpu-a827de5514fe.tar.zst` (4.0 GB; sha256 in the `.sha256` beside it and here) |
| Size | 4,208,422,000 B unpacked |
| Stack | torch 2.11.0+cu128, gsplat 1.5.3 (CUDA kernels compiled ahead of time for 7.5 / 8.6 / 8.9), Python 3.12, OpenCV 5.0.0 |
| Built on | AWS run `20261023T194812Z-cpux86-79ef88`, c7i.4xlarge spot, eu-north-1c, 1632 s, ~$0.0552 (slotq job 41) |

**No compile at run time.** `csrc_proof.txt`: `from gsplat import csrc` resolves to the compiled
`gsplat/csrc.so` in the build stage, in the runtime stage, and in a `docker run --network none` of the finished
image without a GPU. gsplat's backend tries that import first and JIT-compiles only if it fails.

**Not yet validated on a GPU.** That piggybacks on B's first g4dn demo run: nvidia-smi, the D196 micro-bench
against the Kaggle T4 (10.9 ms/iter at 640×360, 100k Gaussians), then a 20 s live run with `--backend gpu`.

The first attempt (`20261023T191330Z-cpux86-763d65`, ~$0.0420) compiled gsplat but failed at a runtime-stage pip
install (no CA bundle in ubuntu:24.04); fixed in `6ecfafc` by installing everything in the build stage.

Load on an instance:
`aws s3 cp s3://…/ariadne-recon-gpu-a827de5514fe.tar.zst - | zstd -dc | docker load` and check
`docker image inspect <tag> --format '{{.Id}}'` equals the image id above.
