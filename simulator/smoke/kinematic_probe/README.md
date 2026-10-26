# Kinematic-mover and bag-size smoke probes (validation.md §4g)

Throwaway probes that decided the 1.1-1.3 sim design. Not production code.

- `KinematicProbe.cc` + `probe.sdf.in` + `run_probe.py`: writes `components::Pose` of a
  model every PreUpdate from an analytic trajectory, with and without the Physics system, and
  checks (a) cameras/depth/spot lamp follow the link, (b) render time vs pose time, (c) what
  gz's own IMU reports on a kinematically moved body.
- `KinematicRing.cc` + `run_bag.py`: two full rigs (stereo 1280x720 RGB + float32 depth,
  20 Hz) circling in a textured 8x12 m room; measures RTF and MCAP size, raw and zstd_fast.
  `run_bag.py` expects `bagsize.sdf` and `tex_*.png` beside it: run `gen_bag_textures.py`
  in this directory first (writes the five 1024^2 fBm albedo PNGs).
  Note: the §4g uncompressed-run RTF was lost to output truncation; the table's raw bytes are
  exact (per-message sums), the RTF 0.217 is the zstd_fast run (compressor competing for CPU).

Run (from the repo root, llvmpipe host):

    docker run --rm --init -e LIBGL_ALWAYS_SOFTWARE=1 -e GALLIUM_DRIVER=llvmpipe \
      -v $PWD/simulator/smoke/kinematic_probe:/work:ro --entrypoint bash ariadne/sim:dev \
      -lc 'source /opt/ros/jazzy/setup.bash && python3 /work/run_probe.py'
