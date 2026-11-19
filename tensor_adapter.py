"""HoloMotion recurrent GPU adapter with independent per-world policy state."""
import json,re
from pathlib import Path
import torch
from holo_torch import HoloNetwork
from motion_challenge.simulation.tensor_math import normalize,conjugate,multiply,rotate,matrix
from motion_challenge.validation import sha256_file


def vector(value,names,device):
    if isinstance(value,dict):
        result=[]
        for name in names:
            found=[v for pattern,v in value.items() if re.fullmatch(pattern,name)]
            if len(found)>1:raise ValueError("Ambiguous joint pattern")
            result.append(found[0] if found else 0.)
        value=result
    t=torch.as_tensor(value,device=device,dtype=torch.float64)
    return t.expand(len(names)) if t.ndim==0 else t


def yaw(q):
    m=matrix(q);return torch.atan2(m[...,1,0],m[...,0,0])


def yaw_quaternion(angle):
    z=torch.zeros_like(angle);return torch.stack((z,z,(angle/2).sin(),(angle/2).cos()),dim=-1)


class ReferenceKinematics:
    """Adapter-owned G1 FK, using only supplied root/joint poses."""
    def __init__(self,root,runtime,names,links,size,device):
        import mujoco,mujoco_warp as mjw,warp as wp
        self.wp,self.mjw=wp,mjw;self.device=torch.device(device)
        path=Path(root)/"assets/reference-g1.xml"
        if sha256_file(path)!=runtime["reference_kinematics_xml_sha256"]:raise ValueError("Reference asset hash differs")
        model=mujoco.MjModel.from_xml_path(str(path))
        with wp.ScopedDevice(str(device)),wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            self.model=mjw.put_model(model);self.data=mjw.make_data(model,nworld=size,nconmax=1,njmax=1)
            self.qpos=wp.to_torch(self.data.qpos);self.qpos[:]=torch.tensor(model.qpos0,device=device,dtype=self.qpos.dtype)
        self.joints=[int(model.joint(n).qposadr[0]) for n in names]
        self.bodies=[model.body(n).id for n in links]

    def __call__(self,positions,rotations,joints):
        self.qpos[:,:3]=positions;self.qpos[:,3:7]=rotations[:,[3,0,1,2]];self.qpos[:,self.joints]=joints.float()
        with self.wp.ScopedDevice(str(self.device)),self.wp.ScopedStream(self.wp.stream_from_torch(torch.cuda.current_stream(self.device))):
            self.mjw.kinematics(self.model,self.data)
        return self.wp.to_torch(self.data.xpos)[:,self.bodies],self.wp.to_torch(self.data.xquat)[:,self.bodies][:,:,[1,2,3,0]]


class TensorHoloMotion:
    def __init__(self,root,runtime,joint_names,slots,device,converted_weights):
        self.device=torch.device(device)
        self.cfg=cfg=json.loads((Path(root)/"policy.json").read_text())
        self.network=HoloNetwork(converted_weights,device).eval()
        if self.network.metadata["source_sha256"]!=runtime["weights_sha256"]:raise ValueError("Source identity mismatch")
        self.dimension=self.network.metadata["input_dimension"]
        names=cfg["policy_joint_names"]
        self.order=[joint_names.index(n) for n in names];self.inverse=[names.index(n) for n in joint_names]
        self.default=vector(cfg["default_joint_pos"],names,device);self.scale=vector(cfg["action_scale"],names,device)
        self.kp=vector(cfg["joint_kp"],names,device)[self.inverse];self.kd=vector(cfg["joint_kd"],names,device)[self.inverse]
        self.clip=cfg.get("action_clip");self.reset_at=cfg.get("rope_reset_at",8128)
        self.float_motor_targets="input_dimension" in cfg
        self.memory=torch.zeros((slots,2,32,4,64),device=device)
        self.steps=torch.zeros(slots,device=device,dtype=torch.long)
        self.heading=torch.zeros((slots,4),device=device,dtype=torch.float64)
        self.last=torch.zeros((slots,29),device=device);self.fresh=torch.ones(slots,device=device,dtype=torch.bool)
        if self.dimension==1378:self.kinematics=ReferenceKinematics(root,runtime,joint_names,cfg["link_names"],slots*6,device)

    def reset(self,slots):
        self.last[slots]=0;self.memory[slots]=0;self.steps[slots]=0;self.fresh[slots]=True

    def observations(self,state,reference):
        q=state.joint_position_rad[:,self.order].double();qd=state.joint_velocity_rad_s[:,self.order].double()
        actual=normalize(state.root_orientation_xyzw.double());inv=conjugate(actual)
        down=torch.tensor([0.,0.,-1.],device=self.device,dtype=torch.float64)
        gravity=rotate(inv,down);omega=rotate(inv,state.root_angular_velocity_world_rad_s.double())
        rotations=normalize(reference.root_orientation_xyzw.double());positions=reference.root_position_world_m.double()
        self.heading.copy_(torch.where(self.fresh[:,None],yaw_quaternion(yaw(actual)-yaw(rotations[:,0])),self.heading))
        self.fresh.zero_()
        if self.dimension==1378:
            count=len(q)
            fk_pos=torch.cat((torch.zeros_like(positions[:,:1]),positions[:,1:]),dim=1)
            fk_rot=torch.cat((actual[:,None],rotations[:,1:]),dim=1)
            fk_joints=torch.cat((state.joint_position_rad[:,None],reference.joint_position_rad[:,1:]),dim=1)
            bp,bq=self.kinematics(fk_pos.reshape(-1,3),fk_rot.reshape(-1,4),fk_joints.reshape(-1,29))
            bp=bp.double().reshape(count,6,14,3);bq=normalize(bq.double().reshape(count,6,14,4))
            local_position=rotate(inv[:,None],bp[:,0])
            local_rotation=multiply(inv[:,None],bq[:,0])
            relative=multiply(inv,self.heading)
            target_positions=rotate(relative[:,None,None],bp[:,1:]-positions[:,0,None,None])
            target_rotations=multiply(relative[:,None,None],bq[:,1:])
            errors=multiply(target_rotations,conjugate(local_rotation[:,None]))
            def xz(q):
                m=matrix(q);return torch.cat((m[..., :,0],m[..., :,2]),dim=-1)
            obs=torch.cat((gravity,omega,q-self.default,qd,self.last,positions[:,0,2:3],
                torch.ones((count,19),device=self.device,dtype=torch.float64),target_positions.flatten(1),
                (target_positions-local_position[:,None]).flatten(1),xz(target_rotations).flatten(1),xz(errors).flatten(1),
                torch.arange(1,6,device=self.device,dtype=torch.float64).expand(count,5)),dim=-1)
        else:
            refq=reference.joint_position_rad[:,:,self.order].double()
            grav=rotate(conjugate(rotations),down)
            linear=rotate(conjugate(rotations),reference.root_linear_velocity_world_m_s.double())
            angular=rotate(conjugate(rotations),reference.root_angular_velocity_world_rad_s.double())
            current=[grav[:,0],linear[:,0],angular[:,0],refq[:,0],positions[:,0,2:3]]
            future=[refq[:,1:].flatten(1),positions[:,1:,2],grav[:,1:].flatten(1),linear[:,1:].flatten(1),angular[:,1:].flatten(1)]
            if self.dimension==604:
                heading_error=yaw(multiply(self.heading,rotations[:,0]))-yaw(actual)
                current.append(torch.stack((heading_error.sin(),heading_error.cos()),dim=-1))
                ref_yaw=yaw(rotations);delta=ref_yaw[:,1:]-ref_yaw[:,:1]
                relative=matrix(multiply(multiply(inv,self.heading)[:,None],rotations[:,1:]))[...,:,:2]
                future.extend((torch.stack((delta.sin(),delta.cos()),dim=-1).flatten(1),relative.flatten(1)))
            current.extend((gravity,omega,q-self.default,qd,self.last))
            obs=torch.cat(current+future,dim=-1)
        reset=self.steps>=self.reset_at
        self.memory.masked_fill_(reset[:,None,None,None,None],0)
        self.steps.masked_fill_(reset,0)
        return obs.float(),self.memory,self.steps

    @torch.inference_mode()
    def act(self,state,reference):
        raw,memory,_,_=self.network(*self.observations(state,reference))
        self.memory.copy_(memory);self.steps.add_(1)
        action=raw.clamp(-self.clip,self.clip) if self.clip is not None else raw
        self.last.copy_(action)
        target=(self.default.float()+self.scale.float()*action) if self.float_motor_targets else (self.default+self.scale*action)
        target=target[:,self.inverse];zero=torch.zeros_like(target)
        return torch.stack((target,zero,self.kp.expand_as(target),self.kd.expand_as(target),zero),dim=1).float()
