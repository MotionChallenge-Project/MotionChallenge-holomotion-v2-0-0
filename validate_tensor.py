"""Independent source-adapter comparison for HoloMotion observations and state."""
import argparse,json,sys,types
from pathlib import Path
import mujoco
import numpy as np
import torch
from onnx.reference import ReferenceEvaluator
from convert_weights import GatherElements
from motion_challenge.sdk import DeployableStateBatch,DeployableStatePacket,MotionReferenceBatch,ReferenceFrame
from motion_challenge.simulation.tensor_reference import TensorMotionBank
from motion_challenge.simulation.tensor_math import rotate
from motion_challenge.resources import resource_root
from motion_challenge.simulation.physics import World
from tensor_adapter import TensorHoloMotion


def validate(prepared,weights,source,device):
    root=Path(__file__).resolve().parent
    runtime=json.loads((root/"runtime.mjwarp.json").read_text());cfg=json.loads((root/"policy.json").read_text())
    meta=json.loads((Path(prepared)/"dataset.json").read_text());names=tuple(meta["joint_names"]);slots=7
    bank=TensorMotionBank(np.array(np.load(Path(prepared)/"poses.npy",mmap_mode="r")),meta["starts"],meta["lengths"],np.full(len(meta["names"]),meta["fps"]),meta["names"],device)
    tensor=TensorHoloMotion(root,runtime,names,slots,device,weights);tensor.reset(torch.arange(slots,device=device))
    sys.modules.setdefault("onnxruntime",types.ModuleType("onnxruntime"))
    from adapter import HoloMotion
    from motion_challenge.adapters.onnx import vector
    oracle=ReferenceEvaluator(str(source),new_ops=[GatherElements]);original=[]
    world=World(resource_root()/"robots/g1-29dof-mjwarp-v1.json")
    for _ in range(slots):
        a=HoloMotion.__new__(HoloMotion);a.cfg=cfg;a.runtime=runtime;a.names=cfg["policy_joint_names"];a.default=vector(cfg["default_joint_pos"],a.names)
        if "link_names" in cfg:a.links=cfg["link_names"]
        original.append(a)
    def bodies(position,quat,joints):
        world.data.qpos[:]=world.model.qpos0;world.data.qpos[:3]=position
        world.data.qpos[3:7]=[quat[3],*quat[:3]];world.data.qpos[world.qadr]=joints
        mujoco.mj_kinematics(world.model,world.data)
        return tuple(types.SimpleNamespace(name=n,position_world_m=world.data.xpos[world.model.body(n).id].copy(),orientation_xyzw=world.data.xquat[world.model.body(n).id,[1,2,3,0]].copy()) for n in cfg.get("link_names",[]))
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rng=np.random.default_rng(19120);maximum=0.;action_error=0.;memory_error=0.;target_error=0.
    for step in range(16):
        offsets=runtime["reference_offsets_s"]
        request=torch.tensor(np.arange(slots)*.031+step*.02,device=device,dtype=torch.float64)[:,None]+torch.tensor(offsets,device=device,dtype=torch.float64)[None]
        ids=torch.arange(slots,device=device)[:,None].expand_as(request)
        t,p,q,j,linear,omega,jv=bank.state(ids,request);refs=MotionReferenceBatch(request,t,j,p,q,jv,linear,rotate(q,omega))
        joints=rng.normal(0,.1,(slots,29));vel=rng.normal(0,.1,(slots,29));quat=rng.normal(size=(slots,4));quat/=np.linalg.norm(quat,axis=-1,keepdims=True);angular=rng.normal(0,.1,(slots,3))
        if step==11:tensor.reset(torch.tensor([1,4],device=device))
        if step==5:tensor.steps[2]=8127;tensor.steps[3]=8128
        state=DeployableStateBatch(names,torch.full((slots,),step*.02,device=device),*[torch.tensor(x,device=device) for x in (joints,vel,quat,angular)])
        obs,memory,index=tensor.observations(state,refs)
        cpu={n:getattr(refs,n).cpu().numpy() for n in ("requested_time_s","source_time_s","joint_position_rad","root_position_world_m","root_orientation_xyzw","joint_velocity_rad_s","root_linear_velocity_world_m_s","root_angular_velocity_world_rad_s")}
        expected_actions=[];expected_memory=[]
        for row,a in enumerate(original):
            packet=DeployableStatePacket(step*.02,names,tuple(joints[row]),tuple(vel[row]),tuple(quat[row]),tuple(angular[row]))
            if "link_names" in cfg:
                packet=types.SimpleNamespace(time_s=step*.02,joint_names=names,joint_position_rad=joints[row],joint_velocity_rad_s=vel[row],root_orientation_xyzw=quat[row],root_angular_velocity_world_rad_s=angular[row],root_position_world_m=np.zeros(3),bodies=bodies(np.zeros(3),quat[row],joints[row]))
            frames=[]
            for k in range(len(offsets)):
                values=[tuple(cpu[n][row,k]) if cpu[n][row,k].ndim else float(cpu[n][row,k]) for n in cpu]
                frames.append(ReferenceFrame(*values,bodies=bodies(values[3],values[4],values[2])))
            if step==0 or (step==11 and row in [1,4]):a.reset(packet,frames,0)
            if step==5 and row in [2,3]:a.step=8127+(row==3)
            expected=a.observations(packet,frames)
            wanted=expected["obs"].astype(np.float32)
            maximum=max(maximum,float(abs(obs[row:row+1].cpu().numpy()-wanted).max()))
            np.testing.assert_allclose(obs[row:row+1].cpu().numpy(),wanted,atol=2e-5,rtol=2e-5)
            np.testing.assert_array_equal(memory[row].cpu().numpy(),expected["past_key_values"].reshape(2,32,4,64))
            if int(index[row])!=int(expected["step_idx"][0]):raise AssertionError("Independent RoPE step differs")
            results=oracle.run(["actions","present_key_values"],{**expected,"obs":wanted})
            expected_actions.append(results[0].reshape(29));expected_memory.append(results[1].reshape(2,32,4,64))
            a.last=results[0].reshape(29).astype(np.float32);a.memory=results[1]
        with torch.inference_mode():actions,new_memory,_,_=tensor.network(obs,memory,index)
        actions_np=actions.cpu().numpy();memory_np=new_memory.cpu().numpy()
        expected_actions=np.stack(expected_actions);expected_memory=np.stack(expected_memory)
        action_error=max(action_error,float(abs(actions_np-expected_actions).max()));memory_error=max(memory_error,float(abs(memory_np-expected_memory).max()))
        np.testing.assert_allclose(actions_np,expected_actions,atol=1e-4,rtol=2e-5)
        np.testing.assert_allclose(memory_np,expected_memory,atol=5e-5,rtol=2e-5)
        target_error=max(target_error,float(abs(tensor.scale.cpu().numpy()*(actions_np-expected_actions)).max()))
        tensor.last.copy_(torch.tensor(expected_actions,device=device));tensor.memory.copy_(torch.tensor(expected_memory,device=device));tensor.steps.add_(1)
    report={"status":"passed","steps":16,"batch":slots,"partial_reset_step":11,"partial_reset_slots":[1,4],"rope_boundary_steps":[8127,8128],"observation_max_abs_error":maximum,"network_max_abs_error":action_error,"memory_max_abs_error":memory_error,"joint_target_max_abs_error_rad":target_error,"history_action_inputs":"identical source-network outputs and caches"}
    (root/"validation/tensor-observation-parity.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True);return report


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--prepared",required=True,type=Path);p.add_argument("--weights",required=True,type=Path);p.add_argument("--source",required=True,type=Path);p.add_argument("--device",default="cuda:0")
    a=p.parse_args();validate(**vars(a))
