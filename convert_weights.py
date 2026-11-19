"""Convert and independently verify this repository's HoloMotion checkpoint."""
import json
from pathlib import Path
import numpy as np
import torch
from onnx.reference import ReferenceEvaluator
from onnx.reference.op_run import OpRun


class GatherElements(OpRun):
    def _run(self,data,indices,axis=0):
        return (np.take_along_axis(data,indices,axis=axis),)

from holo_torch import HoloNetwork,convert
from motion_challenge.validation import sha256_file


def prepare(root,runtime,source,output,artifacts,device):
    root,source,output=Path(root),Path(source),Path(output)
    previous_path=output/"parity.json"
    if previous_path.exists() and (output/"graph.json").exists():
        previous=json.loads(previous_path.read_text())
        if previous.get("converter_sha256")==sha256_file(root/"holo_torch.py") and previous.get("hook_sha256")==sha256_file(__file__) and previous.get("source_sha256")==runtime["weights_sha256"] and previous.get("graph_sha256")==sha256_file(output/"graph.json"):
            HoloNetwork(output,"cpu")
            return output
    metadata=convert(source,output,runtime["weights_sha256"])
    reference=ReferenceEvaluator(str(source),new_ops=[GatherElements])
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rng=np.random.default_rng(20261119);cases=[];maxima={"actions":0.,"memory":0.,"logits":0.}
    names=["actions","present_key_values","moe_layer_0_expert_indices","moe_layer_0_expert_logits"]
    for batch in (1,7,40,129):
        obs=rng.normal(0,.2,(batch,metadata["input_dimension"])).astype(np.float32)
        memory=rng.normal(0,.01,(batch,2,32,4,64)).astype(np.float32)
        steps=np.resize(np.array([0,1,2,31,32,255,8127],np.int64),batch)
        expected=[[] for _ in range(4)]
        for i in range(batch):
            outputs=reference.run(names,{"obs":obs[i:i+1],"past_key_values":memory[i:i+1].reshape(1,2,1,32,4,64),"step_idx":steps[i:i+1]})
            for j,value in enumerate(outputs):expected[j].append(value)
        expected=[np.concatenate(expected[0]),np.stack([v.reshape(2,32,4,64) for v in expected[1]]),np.stack([v.reshape(2) for v in expected[2]]),np.stack([v.reshape(1024) for v in expected[3]])]
        for target_device in dict.fromkeys(("cpu",str(device))):
            model=HoloNetwork(output,target_device).eval()
            with torch.inference_mode():actual=model(torch.tensor(obs,device=target_device),torch.tensor(memory,device=target_device),torch.tensor(steps,device=target_device))
            actual=[v.cpu().numpy() for v in actual]
            np.testing.assert_array_equal(actual[2],expected[2])
            errors={}
            for j,key in [(0,"actions"),(1,"memory"),(3,"logits")]:
                np.testing.assert_allclose(actual[j],expected[j],atol=5e-5,rtol=2e-5)
                errors[key]=float(np.max(abs(actual[j]-expected[j])));maxima[key]=max(maxima[key],errors[key])
            cases.append({"batch":batch,"device":target_device,"steps":sorted(set(steps.tolist())),"max_abs_error":errors,"expert_indices":"exact"})
            del model,actual
    report={"status":"passed","source_sha256":runtime["weights_sha256"],"reference":"original ONNX ReferenceEvaluator per sample; independent per-world cache and step",
            "cases":cases,"max_abs_error":maxima,"oracle_GatherElements":"numpy.take_along_axis","tolerance":{"absolute":5e-5,"relative":2e-5,"expert_indices":"exact"},
            "converter_sha256":sha256_file(root/"holo_torch.py"),"hook_sha256":sha256_file(__file__),"graph_sha256":sha256_file(output/"graph.json")}
    previous_path.write_text(json.dumps(report,indent=2)+"\n")
    return output
