"""Read-only coverage audit, no model fitting or SSH."""
import argparse,collections,hashlib,json,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
from idfs_lab.masked_inputs import build_masked_input,SECOND
from idfs_lab.joint import index_windows


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    dest=Path(a.output)
    if dest.exists():raise ValueError('refuse overwrite')
    sha=hashlib.sha256(Path(a.archive).read_bytes()).hexdigest()
    if sha!='f28fc45ba4e6d80c80da815b06d4e05573a7e9c788cbe199e0a1f01898b40b49':raise ValueError('archive checksum')
    hosts=['idfs'+str(i) for i in (44,45,46,47,64,65,66,67)]
    out={'schema':'masked_host_multiscale_v1','archive_sha256':sha,'fitting_performed':False,'formal_acceptance':False,'time_boundary_review':'pending','timeline':'same15s eligible timeline as first E2E; coverage only','workloads':{}}
    with zipfile.ZipFile(a.archive) as z:
        for d in json.loads(z.read('dataset_index.json'))['datasets']:
            prefix=d['dataset_id']+'/'
            raw=z.read(prefix+'handoff_manifest.json')
            if hashlib.sha256(raw).hexdigest()!=d['handoff_manifest_sha256']:raise ValueError('manifest checksum')
            manifest=json.loads(raw)
            if manifest['role']!='train' or manifest['fault_affected']:raise ValueError('wrong role/fault')
            for item in manifest['files']:
                raw=z.read(prefix+item['path'])
                if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('member checksum')
            rows=[json.loads(l) for l in z.read(prefix+'processed/windows.jsonl').splitlines()]
            index=index_windows(rows);t0=json.loads(z.read(prefix+'recording_start.json'))['epoch_ns']
            counts=collections.Counter();hist=collections.Counter()
            starts=sorted({r['window_start_ns'] for r in rows if r['scope']=='host_oss' and t0<=r['window_start_ns'] and r['window_end_ns']<=t0+600*SECOND})
            for start in starts:
                part=next(((name,lo) for name,lo,hi in [('train',0,360),('development',420,600)] if start-16*SECOND>=t0+lo*SECOND and start+SECOND<=t0+hi*SECOND),None)
                if part is None:counts['purged_or_outside']+=1;continue
                labels=[index.get(('host_oss',h,start)) for h in hosts]
                if any(r is None or not r['complete'] for r in labels):counts['incomplete_labels']+=1;continue
                floor=((t0+part[1]*SECOND+SECOND-1)//SECOND)*SECOND
                fs=[build_masked_input(index,start,h,hosts,d['dataset_id']+':'+d['handoff_manifest_sha256'],floor) for h in hosts]
                if any(f is None for f in fs):counts['incomplete_or_late_short_history']+=1;continue
                counts['accepted_'+part[0]]+=1
                for f in fs:
                    if f['features_available_ns']>start-30000000:raise ValueError('late consumed input')
                    hist[str(f['long_observed_count'])]+=1
            out['workloads'][d['workload']]={'source':d,'coverage':dict(counts),'observed_long_seconds_histogram_per_accepted_node':dict(hist)}
    dest.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({w:d['coverage'] for w,d in out['workloads'].items()}))


if __name__=='__main__':main()
