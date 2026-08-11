# Environment Manifests

- `training-extra.txt`: packages directly used by this project after Isaac
  Sim/Isaac Lab installation.
- `mujoco.txt`: isolated MuJoCo sim2sim environment.
- `conda-linux-64.lock`: exact Conda package snapshot of the working x86_64
  WSL environment.
- `host-pip-freeze-x86_64.txt`: audit snapshot of installed pip packages.

Use `scripts/setup_ubuntu2204.sh` for installation. The two snapshots are for
diagnostics and exact-version comparison; installing the entire pip freeze is
not the primary path because Isaac Sim owns many transitive packages.
