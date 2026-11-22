"""Read-only receipt; no collection, fitting, SSH or extraction."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
from idfs_lab.fine_receipt import verify_delivery


def main():
    p=argparse.ArgumentParser()
    for name in ('directory','manifest-sha256','windows','windows-sha256','dataset-id','run-id','source-role'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--hosts',nargs=8,required=True)
    p.add_argument('--event-start-ns',type=int,action='append',required=True)
    a=p.parse_args()
    try:
        result=verify_delivery(a.directory,a.manifest_sha256,a.windows,a.windows_sha256,a.dataset_id,
                               a.run_id,a.source_role,a.hosts,a.event_start_ns)
    except (ValueError,KeyError,OSError,TypeError) as e:
        print(json.dumps({'status':'needs_fix','reason':str(e),'fitting_performed':False}))
        raise SystemExit(2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
