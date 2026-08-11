# Ubuntu 22.04 Migration

This repository stores the custom source, robot assets, reviewed ONNX policy,
dependency commit pins and the local Unitree patches. NVIDIA drivers, Isaac Sim
binary packages and ROS 2 are installed on the destination machine because they
are platform packages and cannot be redistributed as ordinary repository files.

## 1. Before replacing WSL

Git preserves source and reviewed releases. It does not preserve the ignored
`logs/`, raw checkpoints, caches or every experimental `deploy/candidates/`
directory. Copy any raw run that you may want to resume to separate storage.

The currently promoted baseline is:

```text
deploy/releases/omni45-axis4698/
```

## 2. Clone on Ubuntu

```bash
sudo apt update
sudo apt install -y git git-lfs
git lfs install
git clone --recurse-submodules git@github.com:hsc13576717115/custom_dog_stack.git
cd custom_dog_stack
git lfs pull
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

`third_party/Dog-control` is also private. Do not upload the private SSH key to
Git. Add a new Ubuntu SSH public key to GitHub (or transfer the existing private
key through secure offline storage) before cloning the recursive submodules.

## 3. Host prerequisites

Install the NVIDIA driver and verify that `nvidia-smi` sees the training GPU.
Install Miniconda. Install ROS 2 Humble using the official Ubuntu 22.04 packages
when ROS visualization or controller builds are needed.

The installer targets the x86_64 training PC. Do not run it on Jetson Orin NX;
the Orin consumes the ONNX release and ROS 2 packages, not Isaac Sim training.

## 4. Reconstruct the training environment

The user has already accepted the NVIDIA Omniverse EULA. Keep that explicit on
the new host:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
./scripts/setup_ubuntu2204.sh --install-system-deps
```

The script performs these reproducible steps:

1. initializes the three pinned submodules;
2. applies the recorded Unitree RL Lab and Unitree MuJoCo local patches;
3. creates the Python 3.11 Conda environment;
4. installs Isaac Sim 5.1.0, CUDA PyTorch 2.7.0 and pinned Isaac Lab;
5. installs Unitree RL Lab, this project and the MuJoCo evaluation environment;
6. builds and installs the pinned Unitree SDK2 under `${HOME}/.local/unitree_robotics`;
7. downloads pinned MuJoCo 3.3.6 and builds the native Unitree simulator/controller;
8. checks dependency commits and GPU visibility.

After installation:

```bash
source scripts/activate_env.sh
./scripts/validate.sh
./scripts/train_smoke.sh
```

## 5. ROS 2 verification

```bash
source /opt/ros/humble/setup.bash
./scripts/build_ros2.sh
```

The repository currently contains the policy contract and deployment state
machine, but the real GO-M8010-6 RS485 transport, encoder-zero calibration and
hardware emergency-stop path still require implementation and hardware testing.
