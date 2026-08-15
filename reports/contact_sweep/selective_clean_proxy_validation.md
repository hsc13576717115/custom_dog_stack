# Selective Collision Proxy Validation

The selective-collision asset now uses simplified thigh/calf collision proxies:

- thigh: radius `0.035 m`, length `0.170 m`, link-frame `z=-0.090 m`;
- calf: radius `0.022 m`, length `0.150 m`, link-frame `z=-0.090 m`;
- the foot collision meshes, visual meshes, inertias, and joint definitions are unchanged.

The same geometry contract is generated as URDF cylinders for Isaac and MuJoCo capsules for
sim2sim.  The cleaned MuJoCo model was evaluated with the accepted `SC_740` routed policy over the
15-command Stage-A grid for 10 seconds per command, including a 2-second warmup and 8-second metric
window.

Results are stored in `selective_clean_proxy_grid_10s.json` and `selective_clean_proxy_grid_10s.csv`:

- `15/15` absolute and all-gate commands passed;
- maximum errors: `vx=0.038 m/s`, `vy=0.019 m/s`, `wz=0.029 rad/s`;
- standing mean height `0.323 m`, standing maximum tilt `2.63 deg`;
- maximum hip outward `11.6 deg`;
- pure-yaw XY drift `0.0159 m/s`, yaw-bias rate `0.0294 rad/s`;
- dynamic sim2sim samples reported `0` self-collision and `0` illegal non-foot ground contacts.

This is a MuJoCo-only admission result.  Isaac selective-collision runtime validation must still
be rerun with GPU access; its old report remains invalid and cannot authorize Stage B or recovery
training.

The grid was rerun from the current generator on 2026-08-15 after the sim2sim recovery-state
extensions.  `selective_clean_proxy_grid_10s_rerun.json` reproduces every summary value above
exactly.  The selective Isaac URDF used as the source contract has SHA-256
`71d54ef145a36f3459740395788d491d2d3be0c6deb867ec1af9b38f9bc522b5`.  This repeatability evidence
does not relax the separate Isaac runtime gate.
