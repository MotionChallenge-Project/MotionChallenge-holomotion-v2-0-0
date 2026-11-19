"""Exercise one complete 4096-world policy batch on the target GPU."""
import argparse,importlib,json,time
from pathlib import Path
import torch
from motion_challenge.sdk import DeployableStateBatch,MotionReferenceBatch
from motion_challenge.validation import sha256_file


def validate(weights,device,batch):
    root=Path(__file__).resolve().parent
    runtime=json.loads((root/"runtime.mjwarp.json").read_text())
    cfg=json.loads((root/"policy.json").read_text());names=tuple(cfg["policy_joint_names"])
    entry=runtime["tensor_entrypoint"].split(":")
    policy_class=getattr(importlib.import_module(entry[0]),entry[1])
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.empty(0,device=device);torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
    policy=policy_class(root,runtime,names,batch,device,weights)
    policy.reset(torch.arange(batch,device=device))
    q=policy.default.expand(batch,29).clone();zero=torch.zeros_like(q)
    quat=torch.zeros((batch,4),device=device,dtype=torch.float64);quat[:,-1]=1
    angular=torch.zeros((batch,3),device=device,dtype=torch.float64)
    state=DeployableStateBatch(names,torch.zeros(batch,device=device),q,zero,quat,angular)
    offsets=torch.tensor(runtime["reference_offsets_s"],device=device,dtype=torch.float64)
    times=offsets[None].expand(batch,-1)
    count=len(offsets);rp=torch.zeros((batch,count,3),device=device,dtype=torch.float64);rp[:,:,2]=.78
    rq=quat[:,None].expand(batch,count,4);rj=q[:,None].expand(batch,count,29)
    refs=MotionReferenceBatch(times,times,rj,rp,rq,torch.zeros_like(rj),torch.zeros_like(rp),torch.zeros_like(rp))
    durations=[]
    with torch.inference_mode():
        for step in range(4):
            torch.cuda.synchronize(device);start=time.perf_counter()
            output=policy.act(state,refs)
            torch.cuda.synchronize(device);durations.append(time.perf_counter()-start)
            if output.shape!=(batch,5,29) or not torch.isfinite(output).all():raise AssertionError("Invalid full-batch PD output")
    report={"status":"passed","batch":batch,"device":device,"forward_calls":4,"environment_chunks":1,
            "inference_seconds":durations,"warm_mean_seconds":sum(durations[1:])/3,
            "torch_peak_allocated_bytes":torch.cuda.max_memory_allocated(device),"torch_peak_reserved_bytes":torch.cuda.max_memory_reserved(device),
            "adapter_sha256":sha256_file(root/"tensor_adapter.py"),"graph_sha256":sha256_file(Path(weights)/"graph.json")}
    (root/"validation/full-batch-4096.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--weights",required=True,type=Path);p.add_argument("--device",default="cuda:0");p.add_argument("--batch",default=4096,type=int)
    validate(**vars(p.parse_args()))
