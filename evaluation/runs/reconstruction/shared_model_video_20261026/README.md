# SharedModel video evidence, recovered 2026-10-26

The main checkout's completion register and results index still describe the 20 s
CPU-only seed preview. Later training and render records were found in the
`Ariadne-2.2-radiometric` checkout; copies are preserved here. The animation now
uses those later artifacts. No new mission, GPU job or training was launched.

## Sources

- Mission: `20261024T022434Z-cpux86-626abe`, 200 s, three agents, lit `maze_demo_0`.
- Training: `s3://ariadne-836505524290-eun1/runs/20261024T121121Z-gpu-ae604c/output/results/train/`.
  `training_remote.json` and `training_manifest.json` were retrieved directly from S3.
- Renders: `s3://ariadne-836505524290-eun1/runs/20261024T132853Z-gpu-1e9c60/output/results/`.
  All agent subsets, merged image, 270 orbit frames, four stills and camera metadata
  are downloaded under `build/video_assets/splat_626abe_render_20261024_aws/`.
- `TRAINING_RECORD.md` and `RENDER_RECORD.md` preserve the later checkout's records.

## What the scene shows

The batch budget trained all 1,476 keyframes (568 / 482 / 426 per agent),
100 iterations per keyframe, into 51 submaps and 2,590,315 Gaussians.
The scene restores the original three-dimensional WorldGen wall skeleton,
then assembles all 51 trained submaps in submap creation order. Each chunk
uses its recorded `T_W_anchor`; a deterministic sample of approximately
60,000 Gaussian centres retains actual trained colours and opacity. The
skeleton's scene id is checked against every training spool id. The full-model
counts come from the raw manifests, rather than the display sample.
The assembled model rotates inside the skeleton, then transitions to the
nine-second, 30 fps full gsplat orbit. The skeleton is labelled as WorldGen;
the assembly is labelled as a sampled display and assembly animation.

This is a recorded mission reconstructed incrementally from spooled keyframes.
The submap flight paths and reveal timing are editorial, not temporal training
snapshots. Trained final chunks are revealed piecewise; the outline is a
reference from WorldGen, not a reconstruction output.
The lower live budget took 349.7 s to process the 200 s mission; the scene must
not imply real-time training. The displayed model is the batch budget.
Placement uses briefing transforms, not evidence of live global pose correction.
No quality score is shown: 27.4 dB is a training-view diagnostic, with no
held-out protocol for these novel views. Holes and floaters are in the source.
The old 259,994 seeds / zero dropped keyframes / 92.9% geometry claims belong
to a different 20 s CPU run and are not used.

## Reproduce

```sh
bash python/video/tools/fetch_shared_model.sh
python/video/render.sh scenes/s10_splat.py SharedModel -qm
```

Preview: `python/video/media/videos/s10_splat/720p30/SharedModel.mp4`.
Captions: `python/video/media/captions/SharedModel.json`.

## Validation

The earlier panel-based draft was rendered and inspected, then superseded by
this restoration of the 3D skeleton and piecewise submap assembly.
All 280 source render objects matched S3 single-part ETag/MD5 checksums
(163 downloaded objects and 117 identical cached objects reused).
The trained chunk archive was fetched separately for the restored assembly.
All 54 local model files matched the downloaded archive byte for byte; its
SHA256 is `8bc2db12f2e66205200297c4a8409e55e2bb26b97ffbd5ecbc41587687431294`.
See `trained_chunk_verification.json`.

Restored assembly uses all 51 chunks, 59,999 deterministically sampled points,
recorded world transforms, and the matching WorldGen skeleton. Rendered with
pinned Manim CE 0.22.0 at 1280×720, 30 fps, 33.5 s. Visual review spans top view,
early and late assembly, completed 3D model, and the final Gaussian orbit.
Scene syntax, fetch-script shell syntax, and `git diff --check` pass.

## Latest visual revision

Trained submaps now fade at their final recorded positions with no translation.
The final orbit and in-place camera turn were removed. The ending crossfades
to a complete point cloud sampled from WorldGen textured surfaces, excluding
the roof to expose the interior. This is a hypothetical illustration of the
complete-map goal; it is not an observed result from an unlimited-time mission.
The measured badge and measured-model counts disappear before that crossfade.
No generated point is counted in measured reconstruction statistics.

## Gaussian display and final turnaround

The final visual now loads `worldgen_complete/worldgen_complete.splat`, a saved
standard 32-byte/Gaussian model of textured WorldGen surfaces. Every Gaussian
has a 3D mean, anisotropic scale, orientation quaternion, colour and opacity.
The source manifest includes per-surface geometry/texture checksums.
This remains generated geometry, not an additional measured reconstruction.

The scene uses projected Gaussian covariance and back-to-front alpha blending
for both recorded submaps and the complete model. Explicit smooth alpha ramps
replace PMobject fades, which are no-ops in pinned Manim; its point renderer
also overwrites pixels without blending. All layers stay in place during fades.
After completion fades in, a six-second full camera revolution shows the saved
Gaussian model. The assembly camera itself stays still.

Optional acceleration: `bash python/video/tools/build_gaussian_renderer.sh`.
The C++ compositor is compared with the numpy reference; the pure numpy path
remains available. Turnaround frames are cached by model and camera settings.

Validation of the Gaussian display: 187,405 generated Gaussians; accelerated
and numpy compositors agree exactly on the checked sample (maximum byte
difference 0). Explicit alpha ramps at 0, 0.25, 0.5, 0.75, 1 have strictly
increasing summed opacity and leave image positions unchanged. The generated
model manifest is preserved as `worldgen_gaussian_manifest.json`.

Final Gaussian-layer render: 1280×720, 30 fps, 34.13 s. Reviewed decoded
frames during the opacity ramp, assembled measured submaps, completed
Gaussian model and two turnaround angles. The complete model fades in at a
fixed position before its full six-second revolution. `git diff --check` passes.
