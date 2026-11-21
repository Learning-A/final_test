import argparse
import datetime
import hashlib
import json
from pathlib import Path

from .io import canonical_hash, load, save

ROOT=Path(__file__).resolve().parent.parent


def main(argv=None):
    parser=argparse.ArgumentParser(description="IDFS 本地工程：只读探测、计数适配、训练准备、日志复算")
    parser.add_argument("--site",default=str(ROOT/"config/site.json"))
    parser.add_argument("--protocol",default=str(ROOT/"config/protocol.json"))
    parser.add_argument("--training-host",default=str(ROOT/"config/training_host.json"))
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("probe",help="短时只读探测，证据仅落本地")
    p.add_argument("--out",required=True)
    p=sub.add_parser("clock-check",help="不改时钟，测量相对客户端控制节点的偏移")
    p.add_argument("--out",required=True)
    p=sub.add_parser("collect",help="默认只显示计划；--execute 才运行stdout只读采样")
    p.add_argument("--out",required=True);p.add_argument("--seconds",type=float,default=660);p.add_argument("--execute",action="store_true")
    p=sub.add_parser("process");p.add_argument("--input",required=True);p.add_argument("--out",required=True)
    p=sub.add_parser("export-features");p.add_argument("--windows",required=True);p.add_argument("--out",required=True);p.add_argument("--run-id",required=True)
    p=sub.add_parser("verify");p.add_argument("--run",required=True);p.add_argument("--out",required=True)
    p=sub.add_parser("summarize");p.add_argument("reports",nargs="+");p.add_argument("--out",required=True)
    p=sub.add_parser("inventory",help="本地模型清单，不会部署或训练")
    p=sub.add_parser("gpu-probe",help="只读探查lmj训练服务器")
    p.add_argument("--out",required=True)
    p=sub.add_parser("gpu-submit",help="默认只生成lmj隔离任务计划，--execute才传输并训练")
    p.add_argument("--spec",required=True);p.add_argument("--out",required=True);p.add_argument("--execute",action="store_true")
    p=sub.add_parser("gpu-fetch",help="下载本工程lmj任务的模型和结果")
    p.add_argument("--job-id",required=True);p.add_argument("--out",required=True);p.add_argument("--execute",action="store_true")
    p=sub.add_parser("stage-inference",help="仅部署CPU推理包，不含训练代码，不自动启动")
    p.add_argument("--bundle",required=True);p.add_argument("--execute",action="store_true")
    p=sub.add_parser("relay-live",help="默认只显示计划，连接已训练推理服务并转发只读计数")
    p.add_argument("--release",required=True);p.add_argument("--selection",required=True);p.add_argument("--clock",required=True)
    p.add_argument("--first-io",required=True);p.add_argument("--out",required=True);p.add_argument("--run-id",required=True)
    p.add_argument("--round-index",type=int,required=True);p.add_argument("--seconds",type=float,default=660)
    p.add_argument("--formal",action="store_true");p.add_argument("--execute",action="store_true")
    p=sub.add_parser("freeze-selection",help="在验证数据选定后冻结正式目标和模型")
    p.add_argument("--model-manifest",required=True);p.add_argument("--workload",required=True)
    p.add_argument("--workload-commit",required=True);p.add_argument("--out",required=True)
    p=sub.add_parser("train",help="以后对显式训练/验证数据训练，不自动安装依赖")
    p.add_argument("--train",nargs="+",required=True);p.add_argument("--validation",nargs="+",required=True)
    p.add_argument("--scope",choices=["oss","cluster_oss","client_nfs_rpc"],required=True)
    p.add_argument("--entity",required=True);p.add_argument("--component",choices=["read","write","total"],default="total")
    p.add_argument("--model",required=True);p.add_argument("--out",required=True)
    args=parser.parse_args(argv)
    protocol=load(args.protocol)
    if args.command=="probe":
        from .probe import refresh
        result=refresh(load(args.site),args.out)
        shown={k:result[k] for k in ["sample_records","good_records","observed_oss_ids","observed_clients"]}
    elif args.command=="clock-check":
        from .transport import SiteConnection
        from .probe import targets
        site=load(args.site)
        output=Path(args.out)
        if output.exists():raise ValueError("do not overwrite clock evidence")
        with SiteConnection(site) as connection:
            control={"kind":"client_nfs_rpc","host":site["control_host"]}
            ref=connection.clock_probe(control)
            observations=[]
            for target in targets(site):
                try:
                    item=connection.clock_probe(target)
                    item["relative_offset_ms"]=(item["offset_ns"]-ref["offset_ns"])/1e6
                    item["uncertainty_ms"]=(item["rtt_ns"]+ref["rtt_ns"])/2e6
                except Exception as exc:item={"host":target["host"],"error":str(exc)}
                observations.append(item)
            after=connection.clock_probe(control)
        success=[r for r in observations if "error" not in r]
        shown={"reference_host":site["control_host"],"reference_before":ref,"reference_after":after,
               "observed_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"observations":observations,
               "max_relative_offset_ms":max(abs(r["relative_offset_ms"]) for r in success) if success else None,
               "uncertainty_ms":max(r["uncertainty_ms"] for r in success) if success else None}
        drift=abs(after["offset_ns"]-ref["offset_ns"])/1e6
        shown["reference_change_ms"]=drift
        shown["verified"]=(len(success)==len(observations) and
            shown["max_relative_offset_ms"]<=protocol["max_clock_offset_ms"] and
            shown["uncertainty_ms"]<=protocol["max_clock_uncertainty_ms"] and drift<=protocol["max_clock_uncertainty_ms"])
        save(output,shown)
    elif args.command=="collect":
        from .probe import collect
        shown=collect(load(args.site),protocol,args.out,args.seconds,args.execute)
    elif args.command=="process":
        from .dataset import process
        shown=process(args.input,args.out,protocol)
    elif args.command=="export-features":
        from .dataset import export_features
        shown=export_features(args.windows,args.out,args.run_id)
    elif args.command=="inventory":shown=load(ROOT/"models/inventory.json")
    elif args.command=="gpu-probe":
        from .gpu import probe
        shown=probe(load(args.training_host),args.out)
    elif args.command=="gpu-submit":
        from .workflow import submit_job
        shown=submit_job(load(args.training_host),load(args.spec),args.protocol,args.out,args.execute)
    elif args.command=="gpu-fetch":
        from .workflow import fetch_results
        shown=fetch_results(load(args.training_host),args.job_id,args.out,args.execute)
    elif args.command=="stage-inference":
        from .workflow import stage_inference
        shown=stage_inference(load(args.site),args.bundle,args.execute)
    elif args.command=="relay-live":
        from .live_relay import relay
        shown=relay(load(args.site),protocol,args.release,args.selection,args.clock,args.first_io,args.out,
                    args.run_id,args.round_index,args.seconds,args.formal,args.execute)
    elif args.command=="freeze-selection":
        from .selection import freeze_selection
        shown=freeze_selection(args.model_manifest,args.workload,args.workload_commit,protocol,args.out)
    elif args.command=="train":
        from .training import train
        shown=train(args.train,args.validation,{"scope":args.scope,"entity":args.entity,"component":args.component},args.model,args.out,protocol)
    elif args.command=="verify":
        from .evaluation import verify_files
        shown=verify_files(args.run,protocol);save(args.out,shown)
    elif args.command=="summarize":
        from .evaluation import summarize_rounds
        shown=summarize_rounds([load(p) for p in args.reports],protocol);save(args.out,shown)
    print(json.dumps(shown,ensure_ascii=False,indent=2,allow_nan=False))
    if args.command in ("verify","summarize"):
        return 0 if shown["passed"] else 1
    if args.command=="clock-check":return 0 if shown["verified"] else 1
    return 0


if __name__=="__main__":
    raise SystemExit(main())
