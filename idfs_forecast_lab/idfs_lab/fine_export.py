"""Read-only reduction of existing processed100ms buckets; no filling labels."""
import math
from .counters import GRID_NS


class FineAccumulator:
    def __init__(self,topology,client='client7414'):
        self.groups={n['entity']:set(map(str,n['oss_ids'])) for n in topology['storage_nodes']}
        flat=[v for ids in self.groups.values() for v in ids]
        if len(self.groups)!=8 or any(len(s)!=12 for s in self.groups.values()) or len(flat)!=96 or set(flat)!=set(map(str,range(96))):raise ValueError('complete eight by twelve topology required')
        self.owner={v:h for h,ids in self.groups.items() for v in ids};self.client=client;self.frames={};self.clients={};self.ticks=set()

    def add(self,r):
        start=r.get('bucket_start_ns',r['bucket_end_ns']-GRID_NS)
        if start%GRID_NS:raise ValueError('unaligned bucket')
        self.ticks.add(start)
        scope=r['scope'];entity=str(r['entity'])
        if scope=='client_nfs_rpc' and entity==self.client:
            if start in self.clients:raise ValueError('duplicate client bucket')
            self.clients[start]=r
        if scope!='oss':return
        if entity not in self.owner:raise ValueError('unknown OSS')
        key=(self.owner[entity],start);s=self.frames.setdefault(key,{'members':set(),'ok':True,'read':0,'write':0,'available':[],'observed':[]})
        if entity in s['members']:raise ValueError('duplicate OSS bucket')
        s['members'].add(entity);s['ok']=s['ok'] and bool(r['sample_ok'])
        if r['sample_ok']:
            values=[r['read_count'],r['write_count']]
            if any(not math.isfinite(v) or v<0 for v in values):raise ValueError('invalid count')
            s['read']+=values[0];s['write']+=values[1];s['available'].append(r['available_epoch_ns']);s['observed'].append(r['observed_interval_ns'])

    def rows(self):
        for start in sorted(self.ticks):
            for host,expected in self.groups.items():
                s=self.frames.get((host,start));ok=bool(s and s['ok'] and s['members']==expected)
                r={'scope':'host_oss','entity':host,'bucket_start_ns':start,'bucket_end_ns':start+GRID_NS,'sample_ok':ok,'expected_oss':12,'observed_oss':len(s['members']) if s else 0,'unit':'operations per nominal100ms bucket'}
                if ok:r.update(read_count=s['read'],write_count=s['write'],total_count=s['read']+s['write'],available_epoch_ns=max(s['available']),observed_interval_ns_min=min(s['observed']),observed_interval_ns_max=max(s['observed']))
                else:r['error']='missing_or_invalid_member_oss'
                yield r
            c=self.clients.get(start)
            if c is None:yield {'scope':'client_nfs_rpc','entity':self.client,'bucket_start_ns':start,'bucket_end_ns':start+GRID_NS,'sample_ok':False,'error':'missing_client_bucket'}
            else:yield c
