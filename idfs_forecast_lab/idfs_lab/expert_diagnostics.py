"""Inference-only fixed-weight group removal. Not a trained candidate."""
import torch
from .masked_moe import observation_mask


def routed_prediction(model,x,remove_group=-1):
    if model.mode!='moe' or remove_group not in (-1,0,1,2):
        raise ValueError('learned MoE and one optional group index required')
    z=(torch.sign(x)*torch.log1p(x.abs())-model.mean)/model.std*observation_mask(x)
    for j in [34]+[38+4*i for i in range(15)]+[115+4*i for i in range(8)]:z[:,:,j]=x[:,:,j]
    a,_=model.short(z[:,:,:35]);b,_=model.long(z[:,:,35:112]);c,_=model.spatial(z[:,:,112:])
    weights=torch.softmax(model.upper(z),dim=-1)
    if remove_group>=0:
        keep=torch.ones_like(weights);keep[:,:,remove_group]=0
        weights=weights*keep;weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
    latent=a*weights[:,:,:1]+b*weights[:,:,1:2]+c*weights[:,:,2:]
    correction=torch.cat((model.node_head(latent).squeeze(-1)+model.node_bias,model.cluster_head(latent.reshape(-1,128))),dim=1)
    last=x[:,:,2];anchor=torch.cat((last,last.sum(1,keepdim=True)),dim=1)
    return torch.expm1(torch.log1p(anchor)+correction*model.target_log_std).clamp_min(0),weights
