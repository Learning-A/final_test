"""Offline GRU/TCN adapters, imported only inside an isolated lmj GPU job."""
import copy
import json
import time


def normalized_sequence_output(head, features, mean, std, residual=False):
    """Optional skip in normalized log-target space; raw feature0 is newest lag."""
    if residual:
        import torch
        return head+(torch.log1p(features[:,0].clamp_min(0))-mean)/std
    return head


def fit_sequence(x,y,vx,vy,name,artifact,epochs=100,threads=4,residual_output=False):
    from .placement import require_training_host
    require_training_host()
    import torch
    from torch import nn
    from torch.nn import functional as F
    if not torch.cuda.is_available():raise RuntimeError("selected GPU is unavailable; no silent CPU training fallback")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
    device=torch.device("cuda:0")
    xt=torch.tensor(x,dtype=torch.float32,device=device)
    yt=torch.log1p(torch.tensor(y,dtype=torch.float32,device=device))
    xv=torch.tensor(vx,dtype=torch.float32,device=device)
    yv=torch.log1p(torch.tensor(vy,dtype=torch.float32,device=device))
    mean=yt.mean().item();std=max(yt.std(unbiased=False).item(),1e-6)

    class SequenceModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.is_gru=name=="gru"
            if self.is_gru:self.encoder=nn.GRU(1,32,batch_first=True)
            else:
                self.conv1=nn.Conv1d(1,32,3)
                self.conv2=nn.Conv1d(32,32,3)
            self.head=nn.Linear(32,1)
            if residual_output:
                # Zero correction starts exactly at the last available value.
                nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

        def normalized(self,features):
            sequence=torch.log1p(features[:,:5].clamp_min(0)).flip(1).unsqueeze(-1)
            if self.is_gru:
                output,_=self.encoder(sequence);encoded=output[:,-1,:]
            else:
                z=sequence.transpose(1,2)
                z=F.relu(self.conv1(F.pad(z,(2,0))))
                z=F.relu(self.conv2(F.pad(z,(2,0))))
                encoded=z[:,:,-1]
            return normalized_sequence_output(self.head(encoded).reshape(-1),features,mean,std,residual_output)

        def forward(self,features):
            return torch.expm1((self.normalized(features)*std+mean).clamp(max=60)).clamp_min(0)

    model=SequenceModel().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    best=None;best_loss=float("inf");stale=0;history=[]
    for epoch in range(epochs):
        model.train();order=torch.randperm(len(xt),device=device);loss_sum=0
        for start in range(0,len(xt),128):
            indices=order[start:start+128]
            optimizer.zero_grad(set_to_none=True)
            loss=F.smooth_l1_loss(model.normalized(xt[indices]),(yt[indices]-mean)/std)
            loss.backward();optimizer.step();loss_sum+=float(loss.detach().item())*len(indices)
        model.eval()
        with torch.inference_mode():validation=float(F.smooth_l1_loss(model.normalized(xv),(yv-mean)/std).item())
        history.append({"epoch":epoch+1,"train_loss":loss_sum/len(xt),"validation_loss":validation})
        print(json.dumps({"event":"epoch","model":name,**history[-1]}),flush=True)
        if validation<best_loss:
            best_loss=validation;best=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if stale>=12:break
    if best is None:raise RuntimeError("no finite validation checkpoint")
    model.load_state_dict(best);model.eval()
    with torch.inference_mode():prediction=model(xv).cpu().tolist()
    cpu=model.cpu().eval()
    traced=torch.jit.trace(cpu,torch.tensor(vx[:min(4,len(vx))],dtype=torch.float32),check_trace=True)
    traced.save(str(artifact))
    with torch.inference_mode():cpu_prediction=traced(torch.tensor(vx,dtype=torch.float32)).reshape(-1).tolist()
    parity=max((abs(a-b)/max(1,abs(a)) for a,b in zip(prediction,cpu_prediction)),default=0)
    if parity>1e-5:raise RuntimeError("GPU to CPU inference parity failed")
    return cpu_prediction,{"torch":torch.__version__,"cuda_build":torch.version.cuda,"device":"single_visible_gpu",
                       "gpu_cpu_max_relative_error":parity,"gpu_cpu_parity_passed":True,
                       "epochs_completed":len(history),"history":history,"target_log_mean":mean,"target_log_std":std,
                       "output_structure":"history_log_residual_v1" if residual_output else "absolute_log_v1",
                       "head_initialization":"zero_correction" if residual_output else "torch_default"}
