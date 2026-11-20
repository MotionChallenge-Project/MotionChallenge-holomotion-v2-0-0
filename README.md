# MotionChallenge · holomotion-v2-0-0

G1 29-DoF motion tracking with GPU-batched policy inference and MJWarp simulation.
[Model source](https://github.com/HorizonRobotics/HoloMotion) · [Checkpoint](https://huggingface.co/HorizonRobotics/HoloMotion_models/resolve/e3c90dd1f7483d02d3b4d2e5fb3a359c6ff9c67a/HoloMotion_motion_tracking_model_v2.0.0/exported/model_10500.onnx) · [Provenance](provenance.json)

## Run

Use Linux, Python 3.12 and a CUDA GPU. This repository pins Core as its Git submodule; Core pins the simulation and inference dependencies.

```bash
git clone --recurse-submodules https://github.com/MotionChallenge-Project/MotionChallenge-holomotion-v2-0-0.git
cd MotionChallenge-holomotion-v2-0-0
uv sync --locked
uv run evaluate.py --dataset lafan --data /path/to/datasets/lafan --device cuda:0
uv run evaluate.py --dataset seed --data /path/to/datasets/seed/g1.tar.gz --device cuda:0
```

Download [LAFAN G1 CSVs](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) or the original [BONES-SEED G1 archive](https://huggingface.co/datasets/bones-studio/seed) and keep the files locally. SEED uses Core's single public inclusion list. The command prepares these local motion files, downloads the public checkpoint and required auxiliary model files, verifies their SHA-256 hashes, converts the weights into the cache, completes evaluation and independently verifies every recorded pose trace on CPU. `--data-format lerobot-v3` selects a locally prepared LeRobot v3 dataset.

`--cache PATH` selects the cache; `--output PATH` selects the result directory. `--weights PATH --offline` uses a local original checkpoint. The converter extracts the original tensors to safetensors and checks outputs against original ONNX inference. The default world count is `min(motion count, 4096)`: 40 for the LAFAN suite and 4096 for the SEED suite. `--slots N` overrides it. Inference runs the complete world batch in one forward call. `--verify-jobs N` controls independent CPU trace verification.

## Policy and evaluation

HoloMotion uses a recurrent transformer with a 32-position per-world key/value cache, independent RoPE steps and a top-two mixture of experts. The cache resets at each motion boundary and at the configured RoPE limit. The WholeBody-14 reference and proprioceptive body features use this repository's G1 FK asset and the supplied root/joint poses.

The policy receives joint positions and velocities, root orientation, angular velocity, gravity and its own action history. Actual global position and actual linear velocity are excluded. Motion references supply root/joint information, timing and reference derivatives; the adapter computes its additional features.

Control and scoring run every 20 ms, with four 5 ms physics steps. Contact sliding friction is 1.0 and the Newton solver uses at most 50 iterations. Core precomputes reference FK for scoring and reads actual body poses from the simulator. Each full motion reports success, completion, joint error and MPKPE with world, XY and XY+yaw alignment, for both full-motion and first-failure-prefix scopes.

BONES-SEED evaluation uses a feasibility-filtered subset of the original G1 motion data. Download the original archive; Core applies one public inclusion list of 129,561 motions shared by every model.

## Validation files

[Current validation results](validation/README.md) list inference evidence, full dataset scores and clean-run comparisons.

- [Runtime](runtime.mjwarp.json) and [inference admission](admission.mjwarp.json) bind checkpoint, configuration and validation hashes.
- [holo_torch.py](holo_torch.py) and [convert_weights.py](convert_weights.py) provide this model's batched network and conversion hook.
- [Network parity](validation/inference-network-proof.json) compares extracted weights with original ONNX outputs on CPU and GPU.
- [Observation parity](validation/tensor-observation-parity.json) checks original deployment features, policy memory, partial resets and action targets.
- [4096-world check](validation/full-batch-4096.json) records full-batch inference time and GPU tensor memory.

`result.json` and pose traces are saved to the selected output directory. Independent trace verification checks metrics against the recorded trajectories. A separate clean-cache run and comparison establish reproduction; its dataset-specific report is kept separately from inference admission.

<!-- current-evaluation-evidence -->

## Current evaluation evidence

[LAFAN1 full evaluation](validation/lafan-evaluation.json) · [BONES-SEED full evaluation](validation/seed-evaluation.json)

## Metric

MPKPE (Mean Per-Keypoint Position Error) measures the 3D Euclidean distance between actual and reference keypoint positions. The protocol defines these keypoints at robot body-frame origins: 14 points for G1. Errors are averaged over keypoints and scoring samples within each motion, then motions are weighted equally. Results report full-motion and first-failure-prefix scores with no alignment, XY alignment, and XY + yaw alignment. XY alignment removes pelvis horizontal translation while retaining height error. Leaderboard values are reported in millimetres (mm).
