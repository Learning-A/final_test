"""CPU-only inference for exported numeric models. This module contains no fitting."""
import math
from pathlib import Path

from .io import load


def tree_value(node, features):
    while "leaf" not in node:
        value=features[node["feature"]]
        node=node["left"] if value<=node["threshold"] else node["right"]
    return float(node["leaf"])


class Predictor:
    def __init__(self, model_path, cpu_threads=1):
        self.path=Path(model_path)
        self.model=load(model_path)
        self.kind=self.model["kind"]
        self.torch_model=None
        if self.kind=="torchscript":
            import torch
            torch.set_num_threads(cpu_threads)
            self.torch_model=torch.jit.load(str(self.path.parent/self.model["artifact"]),map_location="cpu")
            self.torch_model.eval()

    def predict(self, features):
        x=[float(v) for v in features]
        if len(x)!=int(self.model.get("feature_count",6)) or not all(math.isfinite(v) for v in x):
            raise ValueError("model feature count / finite-value check failed")
        if self.kind=="last_value":result=x[0]
        elif self.kind=="rolling_mean":result=x[5]
        elif self.kind=="linear":
            result=float(self.model["intercept"])+sum(a*b for a,b in zip(self.model["coefficients"],x))
        elif self.kind=="tree_sum":
            result=float(self.model.get("base",0))+sum(tree_value(t,x) for t in self.model["trees"])
        elif self.kind=="torchscript":
            import torch
            with torch.inference_mode():result=float(self.torch_model(torch.tensor([x],dtype=torch.float32)).item())
        else:raise ValueError("unsupported exported model: "+self.kind)
        if not math.isfinite(result):raise ValueError("non-finite inference result")
        return max(0,result)

    def diagnostics(self,features):
        if self.model.get("diagnostics")!="moe_read_write_gates":return None
        import torch
        with torch.inference_mode():
            values=self.torch_model.details(torch.tensor([features],dtype=torch.float32))[0].tolist()
        return {"total_iops":values[0],"read_iops":values[1],"write_iops":values[2],
                "expert_weights":dict(zip(("temporal","spatial","cross_layer"),values[3:6]))}

    def predict_with_diagnostics(self, features):
        """The MoE's details graph already computes the scalar prediction."""
        if self.model.get("diagnostics") != "moe_read_write_gates":
            return self.predict(features), None
        if len(features) != int(self.model["feature_count"]) or not all(
                math.isfinite(value) for value in features):
            raise ValueError("model feature count / finite-value check failed")
        details = self.diagnostics(features)
        value = details["total_iops"]
        if not math.isfinite(value):
            raise ValueError("non-finite inference result")
        return max(0., value), details


def check_runtime(directory):
    import hashlib
    from .io import save
    directory=Path(directory);deployment=load(directory/"deployment.json")
    for name,expected in deployment["files"].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=expected:raise ValueError("runtime artifact hash mismatch")
    predictor=Predictor(directory/"cpu_model.json")
    vectors=load(directory/"parity_vectors.json")
    errors=[abs(predictor.predict(r["features"])-r["expected"]) for r in vectors]
    tolerance=deployment["equivalence"]["relative_tolerance"]
    ok=all(error<=tolerance*max(1,abs(row["expected"])) for error,row in zip(errors,vectors))
    result={"passed":ok,"vectors":len(vectors),"max_abs_error":max(errors,default=0),"training_performed":False}
    if not ok:raise ValueError("target CPU runtime parity failed")
    return result


if __name__=="__main__":
    import argparse
    import json
    from .io import save
    p=argparse.ArgumentParser();p.add_argument("--check",required=True);p.add_argument("--out")
    a=p.parse_args();r=check_runtime(a.check)
    if a.out:save(a.out,r)
    print(json.dumps(r))
