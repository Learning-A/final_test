"""Masked144 hierarchical experts with causal history-residual outputs."""
import torch
from torch import nn


def observation_mask(x):
    obs=torch.ones_like(x)
    for i in range(15):
        obs[...,35+4*i:38+4*i]=x[...,38+4*i:39+4*i]
    for i in range(8):
        obs[...,112+4*i:115+4*i]=x[...,115+4*i:116+4*i]
    return obs


def training_scaler(x):
    """Only training tensors; masked placeholders contribute neither moments."""
    obs=observation_mask(x);v=torch.sign(x)*torch.log1p(x.abs())
    count=obs.sum(dim=(0,1)).clamp_min(1)
    mean=(v*obs).sum(dim=(0,1))/count
    std=(((v-mean).square()*obs).sum(dim=(0,1))/count).sqrt().clamp_min(.25)
    for j in [34]+[38+4*i for i in range(15)]+[115+4*i for i in range(8)]:
        mean[j]=0;std[j]=1
    return mean,std


class ExpertGroup(nn.Module):
    def __init__(self,dim,uniform):
        super().__init__();self.uniform=uniform
        self.first=nn.Sequential(nn.Linear(dim,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU())
        self.second=nn.Sequential(nn.Linear(dim,32),nn.Tanh(),nn.Linear(32,16),nn.SiLU())
        self.gate=nn.Linear(dim,2)

    def forward(self,x):
        weights=torch.softmax(self.gate(x),dim=-1)
        if self.uniform:weights=torch.ones_like(weights)*.5
        value=weights[...,:1]*self.first(x)+weights[...,1:]*self.second(x)
        return value,weights


class MaskedForecast(nn.Module):
    def __init__(self,mean,std,target_log_std,mode='moe'):
        super().__init__()
        if mode not in ('single','uniform','moe'):raise ValueError('unknown mode')
        self.mode=mode
        self.register_buffer('mean',mean);self.register_buffer('std',std)
        self.register_buffer('target_log_std',target_log_std)
        self.short=ExpertGroup(35,mode=='uniform');self.long=ExpertGroup(77,mode=='uniform');self.spatial=ExpertGroup(32,mode=='uniform')
        self.upper=nn.Linear(144,3)
        self.single=nn.Sequential(nn.Linear(144,64),nn.SiLU(),nn.Linear(64,16),nn.SiLU())
        self.node_head=nn.Linear(16,1)
        self.node_bias=nn.Parameter(torch.zeros(8))
        self.cluster_head=nn.Sequential(nn.Linear(128,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.node_head.weight);nn.init.zeros_(self.node_head.bias)
        nn.init.zeros_(self.cluster_head[-1].weight);nn.init.zeros_(self.cluster_head[-1].bias)

    @torch.jit.export
    def details(self,x):
        obs=torch.ones_like(x)
        for i in range(15):obs[:,:,35+4*i:38+4*i]=x[:,:,38+4*i:39+4*i]
        for i in range(8):obs[:,:,112+4*i:115+4*i]=x[:,:,115+4*i:116+4*i]
        z=(torch.sign(x)*torch.log1p(x.abs())-self.mean)/self.std*obs
        # Preserve raw mask semantics, rather than log(1+mask).
        z[:,:,34]=x[:,:,34]
        for i in range(15):z[:,:,38+4*i]=x[:,:,38+4*i]
        for i in range(8):z[:,:,115+4*i]=x[:,:,115+4*i]
        a,ga=self.short(z[:,:,:35]);b,gb=self.long(z[:,:,35:112]);c,gc=self.spatial(z[:,:,112:])
        upper=torch.softmax(self.upper(z),dim=-1)
        if self.mode=='uniform':upper=torch.ones_like(upper)/3
        latent=a*upper[:,:,:1]+b*upper[:,:,1:2]+c*upper[:,:,2:]
        if self.mode=='single':latent=self.single(z)
        correction=torch.cat((self.node_head(latent).squeeze(-1)+self.node_bias,self.cluster_head(latent.reshape(-1,128))),dim=1)
        last=x[:,:,2]
        anchor=torch.cat((last,last.sum(dim=1,keepdim=True)),dim=1)
        log_pred=torch.log1p(anchor)+correction*self.target_log_std
        prediction=torch.expm1(log_pred).clamp_min(0)
        return prediction,torch.stack((ga,gb,gc),dim=2),upper

    def forward(self,x):
        return self.details(x)[0]


def objective(pred,truth,lower,upper,scale,relative_floor,combo=False,regularize=False):
    from torch.nn import functional as F
    log=F.smooth_l1_loss(torch.log1p(pred),torch.log1p(truth))
    value=log
    if combo:
        raw=F.smooth_l1_loss(pred/scale,truth/scale)
        relative=F.smooth_l1_loss((pred-truth)/truth.clamp_min(relative_floor),torch.zeros_like(pred))
        consistency=((pred[:,:8].sum(1)-pred[:,8]).abs()/scale[8]).mean()
        value=.5*log+.25*raw+.25*relative+.05*consistency
    if regularize:
        balance=(lower.mean((0,1))-.5).square().mean()+(upper.mean((0,1))-1/3).square().mean()
        value=value+.001*balance
    return value
