"""Single counter stream, shared state, eight trained CPU models. No fitting."""
import argparse
import json
from pathlib import Path
import queue
import sys
import threading
import time
from .io import load,save,append_event,BufferedJSONLog,canonical_hash
from .joint import causal_features
from .online import LiveState,verify_deployment
from .runtime import Predictor
from .journal import PredictionJournal

NS=10**9


def serve(model_entries,protocol,output,source=None):
    if protocol.get('experiment_kind')!='smoke_not_acceptance':raise ValueError('smoke service requires explicit smoke protocol')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    models={};site=None
    for entry in model_entries:
        path=Path(entry['model']);metadata=verify_deployment(path);target=metadata['target'];name=target['entity']
        if name!=entry['host'] or target['scope']!='host_oss' or name in models:raise ValueError('invalid/duplicate storage target')
        predictor=Predictor(path/'cpu_model.json',cpu_threads=1)
        # Warm the actual batch-one forward and diagnostics graphs before READY.
        for vector in load(path/'parity_vectors.json')[:4]:
            predictor.predict_with_diagnostics(vector['features'])
        model_site=load(path/'site.json')
        if site is not None and canonical_hash(site)!=canonical_hash(model_site):raise ValueError('model topology mismatch')
        site=model_site
        dest=output/name;dest.mkdir()
        models[name]={'target':target,'predictor':predictor,'metadata':metadata,'journal':PredictionJournal(dest/'predictions.jsonl'),
                      'predictions':{},'settled':set(),'ewma':0.,'residuals':0,'directory':dest}
    if len(models)!=8:raise ValueError('all eight storage nodes required')
    state=LiveState(protocol['expected_oss_ids'],protocol['max_sample_jitter_ms'],site)
    events=queue.Queue(maxsize=10000);source=source or sys.stdin
    def reader():
        try:
            for line in source:
                if line.strip():
                    message=json.loads(line);received=time.time_ns()
                    if message.get('type')=='samples':
                        for row in message['records']:events.put(({'type':'sample','record':row},received))
                    else:events.put((message,received))
        except Exception as exc:events.put(({'type':'input_error','error':str(exc)},time.time_ns()))
        finally:events.put((None,time.time_ns()))
    threading.Thread(target=reader,daemon=True).start()
    ph=canonical_hash(protocol);save(output/'protocol.json',protocol)
    save(output/'service_manifest.json',{'role':'eight_CPU_models_shared_counter_state','training_enabled':False,'models':model_entries,'protocol_hash':ph})
    print(json.dumps({'event':'ready','models':list(models),'warmup_complete':True}),flush=True)
    anchor=None;next_start=None;finished=False
    with BufferedJSONLog(output/'raw_counters.jsonl') as raw,BufferedJSONLog(output/'windows.jsonl') as windows:
        try:
            while not finished or not events.empty():
                try:
                    message,received=events.get(timeout=0.005)
                    if message is None:finished=True
                    elif message.get('type')=='input_error':raise RuntimeError(message['error'])
                    elif message.get('type')=='observation_start':
                        if anchor is not None:raise ValueError('anchor already assigned')
                        anchor=int(message['epoch_ns']);wanted=anchor+protocol['evaluation_from_s']*NS
                        next_start=((wanted+NS-1)//NS)*NS
                        save(output/'observation_start.json',{'event':'observation_start_not_tool_first_io','epoch_ns':anchor})
                    elif message.get('type')=='sample':
                        row=message['record'];raw.append(dict(row,received_inference_epoch_ns=received))
                        for window in state.ingest(row,received):
                            windows.append(window)
                            if window['scope']!='host_oss' or not window['complete']:continue
                            model=models[window['entity']];start=window['window_start_ns']
                            if start not in model['predictions'] or start in model['settled']:continue
                            event=model['predictions'][start];actual=window['total_iops']
                            residual=actual-event['model_raw_iops'];alpha=protocol['ewma']['alpha']
                            model['ewma']=residual if model['residuals']==0 else alpha*residual+(1-alpha)*model['ewma']
                            model['residuals']+=1;model['settled'].add(start)
                            append_event(model['directory']/'settlement.jsonl',{'window_start_ns':start,'actual_iops':actual,
                                'prediction_iops':event['prediction_iops'],'settled_epoch_ns':time.time_ns(),'target':model['target']})
                    else:raise ValueError('unknown input')
                except queue.Empty:pass
                raw.flush();windows.flush();now=time.time_ns()
                if next_start is None or now<next_start-int(protocol['issue_margin_ms']*1e6):continue
                if next_start+NS>anchor+protocol['evaluation_to_s']*NS:
                    next_start=None;continue
                for name,model in models.items():
                    feat=causal_features(state.windows,next_start,site,protocol['issue_margin_ms'],target=model['target'])
                    if feat is None:
                        append_event(model['directory']/'quality_events.jsonl',{'event':'missing_causal_features','window_start_ns':next_start});continue
                    ready=model['residuals']>=protocol['ewma']['warmup_windows'];correction=model['ewma'] if ready else 0.
                    def forecast(x):
                        raw_value, diagnostics = model['predictor'].predict_with_diagnostics(x)
                        return {'raw':raw_value,'prediction':max(0.,raw_value+correction),'diagnostics':diagnostics,
                                'mode':'online' if ready else 'cold_uncalibrated',
                                'calibration':{'applied':ready,'ewma':model['ewma'],'residuals':model['residuals']}}
                    event=model['journal'].issue(forecast,feat['features'],model['target'],next_start,next_start+NS,
                        feat['feature_end_ns'],feat['features_available_ns'],model['metadata']['source_model_sha256'],ph)
                    model['predictions'][next_start]=event
                next_start+=NS
        finally:
            save(output/'service_finished.json',{'ended_epoch_ns':time.time_ns(),'training_performed':False,
                'nodes':{name:{'predictions':len(m['predictions']),'settlements':len(m['settled'])} for name,m in models.items()}})


def main():
    p=argparse.ArgumentParser();p.add_argument('--models',required=True);p.add_argument('--protocol',required=True);p.add_argument('--out',required=True)
    a=p.parse_args()
    import getpass
    if getpass.getuser()!='hust' or Path('/home/hust/idfs_forecast_inference') not in Path(a.out).resolve().parents:
        raise ValueError('private hust inference directory required')
    serve(load(a.models),load(a.protocol),a.out)


if __name__=='__main__':main()
