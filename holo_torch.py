"""Dynamic-batch HoloMotion recurrent attention and top-two expert inference."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file,save_file
from motion_challenge.validation import sha256_file


class HoloNetwork(torch.nn.Module):
    def __init__(self,directory,device="cpu"):
        super().__init__();directory=Path(directory)
        self.metadata=json.loads((directory/"graph.json").read_text())
        if sha256_file(directory/"weights.safetensors")!=self.metadata["weights_sha256"]:raise ValueError("Weight checksum differs")
        for k,v in load_file(str(directory/"weights.safetensors"),device=device).items():self.register_buffer(k,v)

    def norm(self,x,w):return x*(1./torch.sqrt(torch.mean(x**2,dim=-1,keepdim=True)+1e-6))*w
    def silu(self,x):return x*torch.sigmoid(x)
    def mlp(self,x,prefix):
        x=F.linear(x,getattr(self,prefix+"1_weight"),getattr(self,prefix+"1_bias"))
        return F.linear(self.silu(x),getattr(self,prefix+"2_weight"),getattr(self,prefix+"2_bias"))
    def rope(self,x,cos,sin):return x*cos+torch.cat((-x[...,32:],x[...,:32]),dim=-1)*sin

    def _chunk(self,obs,memory,steps):
        batch=len(obs)
        obs=((obs-self.obs_mean)/self.obs_std).clamp(-10.,10.)
        x=self.mlp(obs,"obs_")
        router=self.mlp(obs[:,self.router_indices],"router_")
        z=self.norm(x,self.norm1)
        q=(z@self.q_proj).reshape(batch,8,1,64)
        kv=(z@self.kv_proj).reshape(batch,1,2,4,64)
        k=kv[:,:,0].transpose(1,2);v=kv[:,:,1]
        q=self.norm(q,self.q_norm);k=self.norm(k,self.k_norm)
        cos=self.rope_cos[steps][:,None,None,:];sin=self.rope_sin[steps][:,None,None,:]
        q=self.rope(q,cos,sin);k=self.rope(k,cos,sin)
        indices=(steps%32)[:,None,None,None].expand(batch,1,4,64)
        keys=memory[:,0].scatter(1,indices,k.transpose(1,2))
        values=memory[:,1].scatter(1,indices,v)
        kk=keys.transpose(1,2).repeat_interleave(2,dim=1)
        vv=values.transpose(1,2).repeat_interleave(2,dim=1)
        scores=(q*self.attention_scale)@(kk.transpose(-1,-2)*self.attention_scale)
        valid=self.cache_positions[None,:]<(steps+1).clamp_max(32)[:,None]
        scores=scores+torch.where(valid[:,None,None,:],0.,torch.finfo(torch.float32).min)
        attn=torch.softmax(scores,dim=-1)@vv
        gate=torch.sigmoid(z@self.attention_gate)[:,:,None,None]
        attn=(attn*gate).transpose(1,2).reshape(batch,512)
        x=x+attn@self.o_proj
        z=self.norm(x,self.norm2)
        shared=z@self.shared_up+self.shared_up_bias
        shared=self.silu(shared[:,:256])*shared[:,256:]
        shared=shared@self.shared_down+self.shared_down_bias
        logits=router@self.router_proj
        _,chosen=torch.topk(logits,2,dim=-1,largest=True,sorted=True)
        weights=torch.softmax(logits,dim=-1).gather(1,chosen)
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-20)
        up=torch.matmul(z[:,None,None,:].expand(batch,2,1,512),self.experts_up[chosen]).squeeze(2)
        hidden=self.silu(up[:,:,:256])*up[:,:,256:]
        expert=torch.matmul(hidden[:,:,None,:],self.experts_down[chosen]).squeeze(2)
        result=x+(shared+(expert*weights[:,:,None]).sum(1))
        action=self.mlp(self.norm(result,self.norm_f),"action_")
        return action,torch.stack((keys,values),dim=1),chosen,logits

    def forward(self,obs,memory,steps):
        return self._chunk(obs,memory,steps)


def convert(source,directory,expected_sha256):
    import onnx
    from onnx import numpy_helper,helper
    if sha256_file(source)!=expected_sha256:raise ValueError("Source hash differs")
    model=onnx.load(source)
    initial={t.name:numpy_helper.to_array(t).copy() for t in model.graph.initializer}
    nodes={n.name:n for n in model.graph.node}
    constants={n.output[0]:numpy_helper.to_array(helper.get_attribute_value(n.attribute[0])).copy() for n in model.graph.node if n.op_type=="Constant"}
    values={**initial,**constants};weights={}
    direct={"obs_mean":"obs_normalizer._mean","norm1":"actor_module.layers.0.norm1.weight","norm2":"actor_module.layers.0.norm2.weight","norm_f":"actor_module.norm_f.weight","q_norm":"actor_module.layers.0.attn.q_norm.weight","k_norm":"actor_module.layers.0.attn.k_norm.weight","experts_up":"actor_module.layers.0.gate_up_proj","experts_down":"actor_module.layers.0.down_proj","shared_up_bias":"actor_module.layers.0.shared_experts.gate_up_proj.bias","shared_down_bias":"actor_module.layers.0.shared_experts.down_proj.bias"}
    for name,original in direct.items():weights[name]=torch.from_numpy(values[original])
    for prefix,original in [("obs_","obs_embed"),("router_","router_obs_embed"),("action_","action_mu_head")]:
        for i,j in [(1,0),(2,2)]:
            for kind in ("weight","bias"):
                weights[f"{prefix}{i}_{kind}"]=torch.from_numpy(values[f"actor_module.{original}.{j}.{kind}"])
    mappings={"q_proj":"q_proj","kv_proj":"kv_proj","attention_gate":"gate_proj","o_proj":"o_proj","shared_up":"shared_experts/gate_up_proj","shared_down":"shared_experts/down_proj","router_proj":"router"}
    for name,node in mappings.items():weights[name]=torch.from_numpy(values[nodes[f"/actor_module/{node}/MatMul"].input[1]])
    weights["obs_std"]=torch.from_numpy(values[nodes["/Div"].input[1]])
    weights["router_indices"]=torch.from_numpy(values[nodes["/actor_module/Gather"].input[1]])
    weights["rope_cos"]=torch.from_numpy(values[nodes["/actor_module/Gather_1"].input[0]])
    weights["rope_sin"]=torch.from_numpy(values[nodes["/actor_module/Gather_2"].input[0]])
    weights["attention_scale"]=torch.tensor(np.sqrt(np.float32(.125)),dtype=torch.float32)
    weights["cache_positions"]=torch.arange(32)
    if tuple(weights["experts_up"].shape)!=(1024,512,512) or tuple(weights["experts_down"].shape)!=(1024,256,512):raise ValueError("Unknown expert architecture")
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    save_file(weights,str(directory/"weights.safetensors"))
    meta={"source_sha256":expected_sha256,"weights_sha256":sha256_file(directory/"weights.safetensors"),"converter_sha256":sha256_file(__file__),"input_dimension":weights["obs_mean"].shape[-1],"experts":1024,"top_k":2,"context":32,"query_heads":8,"key_value_heads":4,"head_dim":64,"layers":1}
    (directory/"graph.json").write_text(json.dumps(meta,indent=2)+"\n")
    return meta
