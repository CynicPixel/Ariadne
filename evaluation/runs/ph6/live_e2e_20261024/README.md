# Live splat end to end (6.x POC, D230/D231), 2026-10-24 ~01:50-02:10 IST

Real runtime -> real service, no mock. B's 20 s 3-agent lit maze_demo_0 multiagent run (binaries feat/l1-a1p-llm
2a123e8, run dir Ariadne-llm build/live/splat_e2e_run, digest cd6a07aa6fdd000a = the no-tee reference: the tee is
determinism-neutral against the real consumer) teed over ARIADNE_RECON_SOCK into A2's service (feat/ph5-dve e2765ec, now on main 982046f;
ariadne-recon:1, cpu_init, --scene maze_demo_0).

- link: #t_w_o 3, #kf 71, dropped {} (runtime tee: 0 drops).
- 71 keyframes -> 4 submaps, 259,994 Gaussians, 8.3 MB, table v76.
- placement: B's #t_w_o == A2's briefing transform to <= 2.2e-16 for all 3 agents (qw-last, agent 2's 180 deg entrance).
- GT check (seeds in W vs the scene ESDF): wall submaps 92.9 % within 10 cm (agents 1, 2); agent 0's submaps 65.8 % and
  47.3 % (more floor: z median 0.68 / 0.01 m), the same pattern as the spool replay.
- /splat in headless Chromium on the Intel Iris Xe (ANGLE GL): table, chunks, GT mesh load; 2 canvases; screenshot.
Not yet: GPU training (cpu_init only), the 200 s recording.
