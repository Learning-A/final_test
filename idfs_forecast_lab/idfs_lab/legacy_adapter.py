"""Scaling around the unmodified historical sequence core."""
import torch
from torch import nn


class LegacyAdapter(nn.Module):
    def __init__(self,core,window,mean=0.,std=1.):
        super().__init__();self.core=core;self.window=window
        self.register_buffer('global_mean',torch.tensor(float(mean)))
        self.register_buffer('global_std',torch.tensor(float(std)))

    @torch.jit.export
    def normalized(self,x):
        if self.window:
            mean=x.mean(1,keepdim=True);std=torch.sqrt(x.var(1,unbiased=False,keepdim=True)+1e-5)
        else:
            mean=torch.ones_like(x[:,:1])*self.global_mean;std=torch.ones_like(x[:,:1])*self.global_std
        p=self.core((x-mean)/std).squeeze(-1)
        return p,mean[:,0,0],std[:,0,0]

    def forward(self,x):
        p,mean,std=self.normalized(x)
        return (p*std+mean).clamp_min(0)
