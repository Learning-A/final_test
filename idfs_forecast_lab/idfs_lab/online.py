"""Inference-only stdio service; counter input is forwarded by the external collector."""
import argparse
import getpass
import hashlib
import json
import math
from pathlib import Path
import queue
import sys
import threading
import time

from .counters import GRID_NS,counter_delta
from .dataset import build_windows,feature_rows
from .evaluation import score
from .io import append_event,canonical_hash,load,save,BufferedJSONLog
from .journal import PredictionJournal
from .runtime import Predictor

NS=1000000000


class LiveState:
    def __init__(self,expected_ids,jitter_ms=25,site=None):
        self.expected=set(str(v) for v in expected_ids)
        self.jitter_ms=jitter_ms;self.previous={};self.groups={};self.ticks={};self.windows={}
        self.site=site

    def ingest(self,row,received_ns):
        identity=(row["scope"],row["entity"])
        previous=self.previous.get(identity);self.previous[identity]=row
        if previous is None:return []
        bucket=counter_delta(previous,row,jitter_ns=int(self.jitter_ms*1e6))
        # Availability is when the inference host actually received the data,
        # not a Windows receipt time or the earlier source sampling time.
        bucket["available_epoch_ns"]=max(received_ns,row["read_end_epoch_ns"])
        fresh=self._bucket(bucket)
        if bucket["scope"]=="oss":
            end=bucket["bucket_end_ns"];group=self.groups.setdefault(end,{})
            if str(bucket["entity"]) in group:raise ValueError("duplicate OSS bucket")
            group[str(bucket["entity"])]=bucket
            if set(group)==self.expected:
                combined={"scope":"cluster_oss","entity":"all","bucket_start_ns":end-GRID_NS,
                          "bucket_end_ns":end,"sample_ok":all(v["sample_ok"] for v in group.values()),
                          "available_epoch_ns":max(v["available_epoch_ns"] for v in group.values())}
                if combined["sample_ok"]:
                    for component in ("read","write","total"):
                        combined[component+"_count"]=sum(v[component+"_count"] for v in group.values())
                fresh+=self._bucket(combined)
                del self.groups[end]
        # Old incomplete groups remain a gap, never a zero-valued cluster.
        cutoff=row["scheduled_epoch_ns"]-10*NS
        self.groups={t:g for t,g in self.groups.items() if t>=cutoff}
        if self.site:
            for window in list(fresh):
                if window["scope"]!="oss":continue
                second=window["window_start_ns"]
                for node in self.site["storage_nodes"]:
                    if int(window["entity"]) not in node["oss_ids"]:continue
                    key=("host_oss",node["hostname"],second)
                    members=[self.windows.get(("oss",str(i),second)) for i in node["oss_ids"]]
                    if key in self.windows or any(r is None for r in members):continue
                    from .joint import host_windows
                    combined=next(r for r in host_windows(members,{"storage_nodes":[node]}) if r["window_start_ns"]==second)
                    self.windows[key]=combined;fresh.append(combined)
        return fresh

    def _bucket(self,bucket):
        start=bucket.get("bucket_start_ns",bucket["bucket_end_ns"]-GRID_NS)
        second=start//NS*NS;key=(bucket["scope"],bucket["entity"],second)
        ticks=self.ticks.setdefault(key,{})
        if start in ticks:raise ValueError("duplicate source bucket")
        ticks[start]=bucket
        if len(ticks)<10:return []
        window=build_windows(list(ticks.values()))[0]
        if key in self.windows:raise ValueError("duplicate actual window")
        self.windows[key]=window;del self.ticks[key]
        return [window]

    def feature(self,target,start):
        # Build the same causal history as offline training, including the
        # one-second safety lag when the newest boundary has not arrived.
        series=[self.windows[(target["scope"],str(target["entity"]),start-i*NS)]
                for i in range(1,8) if (target["scope"],str(target["entity"]),start-i*NS) in self.windows]
        dummy={"scope":target["scope"],"entity":target["entity"],"window_start_ns":start,
               "window_end_ns":start+NS,"available_epoch_ns":start+NS,
               "complete":True,"n_ticks":10,"read_iops":0,"write_iops":0,"total_iops":0}
        rows=list(feature_rows(series+[dummy]))
        found=[r for r in rows if r["window_start_ns"]==start and r["component"]==target["component"]]
        return found[0] if len(found)==1 else None


def verify_deployment(directory):
    directory=Path(directory);deployment=load(directory/"deployment.json")
    for name,digest in deployment["files"].items():
        path=(directory/name).resolve()
        if directory.resolve() not in path.parents:raise ValueError("model file outside export")
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError("inference artifact checksum mismatch")
    if not deployment["equivalence"]["passed"]:raise ValueError("export gate failed")
    return deployment


def serve(model_dir,protocol_path,output,selection_path=None,source=None,clock_path=None):
    deployment=verify_deployment(model_dir);protocol=load(protocol_path);ph=canonical_hash(protocol)
    if selection_path:
        selection=load(selection_path)
        if selection["target"]!=deployment["target"] or selection["model_sha256"]!=deployment["source_model_sha256"] or selection["protocol_hash"]!=ph:
            raise ValueError("frozen selection and deployed model differ")
    target=deployment["target"]
    joint_site=load(Path(model_dir)/"site.json") if deployment.get("feature_schema")=="joint_context_v1" else None
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    if selection_path:save(output/"selection.json",selection)
    if clock_path:save(output/"clock_alignment.json",load(clock_path))
    predictor=Predictor(Path(model_dir)/"cpu_model.json",cpu_threads=1)
    raw_log=BufferedJSONLog(output/"raw_counters.jsonl")
    window_log=BufferedJSONLog(output/"windows.jsonl")
    journal=PredictionJournal(output/"predictions.jsonl")
    state=LiveState(protocol["expected_oss_ids"],protocol["max_sample_jitter_ms"],joint_site)
    events=queue.Queue(maxsize=10000)
    source=source or sys.stdin

    def reader():
        try:
            for line in source:
                if line.strip():events.put((json.loads(line),time.time_ns()))
        except Exception as exc:events.put(({"type":"input_error","error":str(exc)},time.time_ns()))
        finally:events.put((None,time.time_ns()))
    thread=threading.Thread(target=reader,daemon=True);thread.start()
    t0=None;next_start=None;finished=False;predictions={};settled=set();pairs=[]
    ewma=0.;residuals=0;pending=[]
    save(output/"service_manifest.json",{"role":"inference_only","model":deployment,"protocol_hash":ph,"training_enabled":False})
    try:
        while not finished or not events.empty():
            try:
                message,received=events.get(timeout=0.01)
                if message is None:finished=True
                elif message.get("type")=="input_error":raise RuntimeError(message["error"])
                elif message.get("type") in ("t0","observation_start"):
                    if t0 is not None:raise ValueError("t0 already set")
                    smoke_anchor=message.get("type")=="observation_start"
                    if smoke_anchor and protocol.get("experiment_kind")!="smoke_not_acceptance":
                        raise ValueError("observation anchor is allowed only for non-acceptance smoke")
                    t0=int(message["epoch_ns"])
                    anchor_event = (message.get("evidence") or {}).get("details", {}).get("event", "tool_first_io")
                    if not smoke_anchor and anchor_event not in ("tool_first_io", "recording_start"):
                        raise ValueError("unsupported recording anchor")
                    save(output/("observation_start.json" if smoke_anchor else "first_io.json"),
                         {"event":"observation_start_not_tool_first_io" if smoke_anchor else anchor_event,
                          "epoch_ns":t0,"evidence":message.get("evidence")})
                    wanted=t0+protocol["evaluation_from_s"]*NS
                    next_start=((wanted+NS-1)//NS)*NS
                elif message.get("type")=="sample":
                    row=message["record"];raw_log.append(dict(row,received_inference_epoch_ns=received))
                    fresh=state.ingest(row,received)
                    for window in fresh:
                        window_log.append(window)
                        key=window["window_start_ns"]
                        if (window["scope"]==target["scope"] and str(window["entity"])==str(target["entity"]) and
                                key in predictions and key not in settled and window["complete"]):
                            actual=window[target["component"]+"_iops"];pred=predictions[key]
                            # Apply each calibration residual only to future predictions.
                            residual=actual-pred
                            alpha=protocol["ewma"]["alpha"]
                            ewma=residual if residuals==0 else alpha*residual+(1-alpha)*ewma
                            residuals+=1;settled.add(key);pairs.append((pred,actual))
                            append_event(output/"settlement.jsonl",{"window_start_ns":key,"target":target,
                                "prediction_iops":pred,"actual_iops":actual,"settled_epoch_ns":time.time_ns()})
                            save(output/"accuracy_provisional.json",dict(score(pairs,protocol["mask"]),provisional=True))
                            save(output/"calibration_state.json",{"residuals":residuals,"ewma":ewma})
                else:raise ValueError("unknown inference input message")
            except queue.Empty:pass
            raw_log.flush();window_log.flush()
            now=time.time_ns()
            if next_start is not None and now>=next_start-int(protocol.get("issue_margin_ms",30)*1e6):
                if next_start+NS>t0+protocol["evaluation_to_s"]*NS:
                    next_start=None;continue
                if joint_site:
                    from .joint import causal_features
                    feat=causal_features(state.windows,next_start,joint_site,protocol.get("issue_margin_ms",30),target=target)
                else:
                    feat=state.feature(target,next_start)
                if feat is None:
                    append_event(output/"quality_events.jsonl",{"event":"missing_causal_features","window_start_ns":next_start,"at_ns":now})
                else:
                    features=feat["features"] if joint_site else feat["lags"]+[feat["rolling_mean"]]
                    correction=ewma if residuals>=protocol["ewma"]["warmup_windows"] else 0
                    def forecast(x):
                        raw=predictor.predict(x)
                        return {"raw":raw,"prediction":max(0,raw+correction),
                                "mode":"online" if residuals>=protocol["ewma"]["warmup_windows"] else "cold_uncalibrated",
                                "calibration":{"applied":residuals>=protocol["ewma"]["warmup_windows"],"ewma":ewma,"residuals":residuals},
                                "diagnostics":predictor.diagnostics(x)}
                    event=journal.issue(forecast,features,target,
                        next_start,next_start+NS,feat["feature_end_ns"],feat["features_available_ns"],
                        deployment["source_model_sha256"],ph)
                    predictions[next_start]=event["prediction_iops"]
                next_start+=NS
    finally:
        raw_log.__exit__();window_log.__exit__()
        save(output/"service_finished.json",{"ended_epoch_ns":time.time_ns(),"predictions":len(predictions),
             "settlements":len(settled),"fit_performed":False})


def main():
    p=argparse.ArgumentParser(description="CPU在线预测，stdin接收只读计数和真实首I/O事件")
    p.add_argument("--model",required=True);p.add_argument("--protocol",required=True)
    p.add_argument("--out",required=True);p.add_argument("--selection")
    p.add_argument("--clock")
    a=p.parse_args()
    if getpass.getuser()!="hust":raise RuntimeError("cluster inference must run as the isolated hust user")
    output=Path(a.out).resolve();root=Path("/home/hust/idfs_forecast_inference").resolve()
    if (Path("/home/hust/idfs_forecast_inference").is_symlink() or
        Path("/home/hust").resolve() not in root.parents or root not in output.parents):
        raise ValueError("inference logs must stay under the isolated hust service root")
    serve(a.model,a.protocol,a.out,a.selection,clock_path=a.clock)


if __name__=="__main__":main()
