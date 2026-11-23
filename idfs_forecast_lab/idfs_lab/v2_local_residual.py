"""History-local output scaling and residual head; input statistics never fit here."""
import numpy as np
import torch
from torch import nn

from .r2_scaling import checked_input,transform

LOCAL_NAMES=('local_base_over_training_s','local_scale_over_training_s',
             'latest_minus_local_mean_over_local_scale','local_zero_history')


def local_context(raw,training_scales):
    x=np.asarray(raw,dtype=np.float64)
    scales=np.asarray(training_scales,dtype=np.float64)
    if x.ndim!=3 or x.shape[1:]!=(8,258) or scales.shape!=(8,) or not np.isfinite(scales).all() or (scales<1).any():
        raise ValueError('Fixed second/node/C258 axes and eight existing training scales required')
    history=x[:,:, [2,5,8,11,14]]
    if not np.isfinite(history).all() or (history<0).any():
        raise ValueError('Complete five-second nonnegative historical total counts required')
    mean=history.mean(axis=2)
    base=np.maximum(mean,1.0)
    scale=np.maximum(np.maximum(history.std(axis=2,ddof=0),1.0),.05*scales[None,:])
    descriptors=np.stack((base/scales,scale/scales,(history[:,:,0]-mean)/scale,(mean==0).astype(float)),axis=2)
    return base,scale,descriptors


def prepare_features(raw,names,preprocessing):
    x=np.asarray(raw,dtype=np.float64)
    checked_input(x.reshape(-1,258),names)
    state=preprocessing['scaling']
    base,scale,context=local_context(x,preprocessing['target_scales']['s'])
    fixed=transform(x.reshape(-1,258),names,state).reshape(x.shape)
    features=np.concatenate((fixed,context),axis=2)
    if features.shape[2]!=262 or not np.isfinite(features).all():
        raise ValueError('Invalid fixed local-context inputs')
    return features,base,scale


class LocalResidualMLP(nn.Module):
    """Same hidden64 layers, signed correction and nonnegative final prediction."""

    def __init__(self,training_scales):
        super().__init__()
        scales=torch.as_tensor(training_scales,dtype=torch.float32).detach().cpu().clone()
        if scales.shape!=(8,) or not torch.isfinite(scales).all() or (scales<1).any():
            raise ValueError('Eight existing training target scales required')
        self.register_buffer('training_scales',scales)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(2026)
            self.node_embedding=nn.Embedding(8,4,device='cpu')
            self.backbone=nn.Sequential(nn.Linear(266,64,device='cpu'),nn.ReLU(),
                                        nn.Linear(64,64,device='cpu'),nn.ReLU(),
                                        nn.Linear(64,1,device='cpu'))
        nn.init.zeros_(self.backbone[-1].weight)
        nn.init.zeros_(self.backbone[-1].bias)

    def forward(self,features,node_indices):
        if features.ndim!=2 or features.shape[1]!=262 or node_indices.shape!=(len(features),) or node_indices.dtype!=torch.long:
            raise ValueError('C258 plus four local descriptors and one node ID required')
        target_scale=self.training_scales.index_select(0,node_indices)
        baseline=features[:,-4]*target_scale
        local_scale=features[:,-3]*target_scale
        correction=self.backbone(torch.cat((features,self.node_embedding(node_indices)),dim=1)).squeeze(-1)
        return torch.relu(baseline+local_scale*correction)
