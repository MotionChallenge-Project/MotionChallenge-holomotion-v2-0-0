# HoloMotion v2.0.0

[Author project](https://horizonrobotics.github.io/holomotion/) · [Checkpoint](https://huggingface.co/HorizonRobotics/HoloMotion_models/blob/e3c90dd1f7483d02d3b4d2e5fb3a359c6ff9c67a/HoloMotion_motion_tracking_model_v2.0.0/exported/model_10500.onnx) · [Provenance](provenance.json)

## Install and run

```bash
git clone --recurse-submodules https://github.com/MotionChallenge-Project/MotionChallenge-holomotion-v2-0-0.git
cd MotionChallenge-holomotion-v2-0-0
uv sync
uv run evaluate.py
```

One full rollout per motion produces success, completion and all six MPBPE values
in one verified `result.json`. MuJoCo 3.17 CPU batch; `--jobs 8` sets rollout and
verification parallelism. Progress updates every five seconds.

Missing data and weights are downloaded and hash-checked. Cached files can be
selected with `--weights PATH --data PATH --offline`. Results and traces stay
outside Git.

This adapter uses the released WholeBody-14 controller. The root `motion-challenge`
submodule owns datasets, physics, metrics and verification. See `--help` for options.
