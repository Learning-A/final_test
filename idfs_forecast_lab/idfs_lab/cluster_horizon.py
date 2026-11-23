"""Pure target aggregation and equal-capacity direct multi-horizon models."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SECOND=1_000_000_000
GRID=100_000_000
HORIZONS=(1,3,5,10)


def collect_future_counts(index,start,hosts,recording_start):
    """Labels only: complete next 100 nominal buckets, no missing-as-zero."""
    if start+10*SECOND>recording_start+600*SECOND:
        return None,'target_beyond_recording_workload_range'
    counts=[]
    for tick in range(100):
        at=start+tick*GRID;total=0.
        for host in hosts:
            row=index.get(('host_oss',host,at))
            if row is None or not row['sample_ok']:
                return None,'future_target_incomplete'
            total+=row['total_count']
        counts.append(total)
    return np.asarray(counts,dtype=np.float64),None


def seconds_from_counts(counts):
    a=np.asarray(counts,dtype=np.float64)
    if a.shape[-1]!=100 or not np.isfinite(a).all() or (a<0).any():
        raise ValueError('100 finite nonnegative future count buckets required')
    return a.reshape(*a.shape[:-1],10,10).sum(axis=-1)


def window_means(seconds):
    a=np.asarray(seconds,dtype=np.float64)
    if a.shape[-1]!=10 or not np.isfinite(a).all() or (a<0).any():
        raise ValueError('10 finite nonnegative future per-second rates required')
    return np.stack([a[...,:h].mean(axis=-1) for h in HORIZONS],axis=-1)


def cluster_features(scaled_nodes):
    a=np.asarray(scaled_nodes)
    if a.ndim!=3 or a.shape[1:]!=(8,258):raise ValueError('node C258 order required')
    return np.concatenate((a[:,:,:112].reshape(len(a),-1),a[:,0,112:144],a[:,:,144:].reshape(len(a),-1)),axis=-1)


class ClusterHorizonMLP(nn.Module):
    """Identical weights for 100ms versus 1s targets; fixed time/width coordinates."""
    def __init__(self,scale,steps):
        super().__init__()
        if steps not in (10,100) or not np.isfinite(scale) or scale<1:raise ValueError('scale/steps')
        width=10./steps
        coords=torch.tensor([[(i+.5)*width/10.,width] for i in range(steps)],dtype=torch.float32)
        self.register_buffer('coords',coords)
        self.register_buffer('scale',torch.tensor(float(scale),dtype=torch.float32))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(2026)
            self.encoder=nn.Sequential(nn.Linear(1840,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
            self.decoder=nn.Sequential(nn.Linear(66,32),nn.ReLU(),nn.Linear(32,1))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.constant_(self.decoder[-1].bias,float(np.log(np.expm1(1.))))

    def forward(self,x):
        if x.ndim!=2 or x.shape[1]!=1840:raise ValueError('cluster1840 required')
        latent=self.encoder(x)
        n,steps=len(x),len(self.coords)
        joined=torch.cat((latent[:,None,:].expand(n,steps,64),self.coords[None,:,:].expand(n,steps,2)),dim=-1)
        return F.softplus(self.decoder(joined).squeeze(-1))*self.scale
