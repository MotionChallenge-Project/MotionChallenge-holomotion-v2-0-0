# Validation

Inference admission binds the original checkpoint, model configuration, tensor implementation and fixed parity evidence. Full dataset evaluation and clean-run comparisons have separate reports.

| Check | Status | Evidence |
| --- | --- | --- |
| Extracted-network parity | passed | [Original ONNX versus CPU and GPU tensor inference](inference-network-proof.json) |
| Deployment parity | passed | [Original observations, history or recurrent state, motion resets and joint targets](tensor-observation-parity.json) |
| Full-batch inference | passed | [4096 worlds in one forward call](full-batch-4096.json) |
| Fresh-cache conversion | passed | [Unified converter entrypoint; repository files unchanged](conversion-cache-check.json) |
| ONNX Runtime cross-check | passed | [Original ONNX Runtime 1.30.0 CPU outputs versus converted GPU outputs](onnxruntime-crosscheck.json) |
| Independent delivery | passed | [Model-local production modules, pinned Core dependency, locked environment and current commands](delivery-check.json) |
| Checkpoint origin | author_release_verified | [Publisher release identity, original-file hashes and official-entry eligibility](checkpoint-origin.json) |

The recorded 4096-world actor check takes 36.16 ms per warm forward call and allocates at most 10.709 GiB of PyTorch tensors. This measures policy inference separately from complete simulation.

## Full dataset evaluation

Control and scoring use 20 ms periods, four 5 ms physics steps, contact sliding friction 1.0 and Newton maximum 50 iterations. The default batch contains `min(motion count, 4096)` worlds. Full-motion and first-failure-prefix scores are recorded separately.

| Dataset | Motions | Successful motions | Mean completion | Rollout seconds | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| LAFAN | 40 | 24 | 0.78155523 | 203.62 | [Independent trace receipt](lafan-evaluation.json) |
| SEED | 129561 | 126402 | 0.98561777 | 1023.97 | [Independent trace receipt](seed-evaluation.json) |

Rollout time includes policy inference, physics, scoring, slot refill and trace recording; it excludes preparation and initial reset. Independent CPU verification recomputes reference interpolation, FK, failures and scores from every recorded sample.

## Reproduction

Run the dataset command from the repository README with a fresh `--cache` and `--output`. The command downloads and checks the original model files, prepares the selected original dataset, converts the model, completes the rollout and independently verifies its traces. Compare two complete results with:

```bash
uv run python -m motion_challenge.evaluation.reproduction /path/to/first/result.json /path/to/second/result.json --output /path/to/comparison.json
```

The comparison checks dataset, model, physics and rollout identities and both trace receipts. Its pipeline status and numerical differences are separate fields.

No same-identity clean-run comparison is included in this repository yet.
