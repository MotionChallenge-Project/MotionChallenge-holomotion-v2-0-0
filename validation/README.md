# Native inference comparison

`native-parity.json` compares 160 states, four resets, 20 endpoint contexts and
the 8128-step recurrent-cache boundary. It runs the pinned author's CUDA Warp
reference FK, measured FK and 1378-dimensional observation builder, plus the
original heading calibration, recurrent ONNX runtime and PD-target conversion.
Joint order is permuted; measured poses include nearly vertical orientations.

Only ROS and unused training import glue are omitted. All native files are
checked against `native-sources.json`; the checkpoint configuration is checked
against its saved SHA256. The canonical robot's body poses are also compared
with the author's FK asset. Physics, reference data and failure rules are not changed.

```bash
uv run --with torch --with warp-lang validate_inference.py \
  --source /path/to/pinned/HoloMotion \
  --weights /path/to/model_10500.onnx \
  --output /path/outside/git/native-parity.json
```

CUDA is needed only for this comparison, not for `evaluate.py`. On CUDA 12 drivers,
use NVIDIA's official [Warp +cu12 wheel](https://github.com/NVIDIA/warp/releases/tag/v1.18.0).
The recorded audit used Warp 1.18.0+cu12 on an A100 and ONNX Runtime CPU.
It checks CPU adapter parity with the author's observation and runtime logic;
it does not claim CPU/CUDA ONNX provider equivalence.
