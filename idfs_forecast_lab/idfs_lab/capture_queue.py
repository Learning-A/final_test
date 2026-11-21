"""One durable development attempt per invocation; safe for repeated heartbeats.

No daemon, OSS repair, process signals or hidden background jobs. OS locks are
released on process exit. Interrupted attempts are preserved and retried only
after an idle/healthy remote preflight, in a new directory.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import time

from .collect_workloads import development_rounds, run
from .io import canonical_hash, load, save
from .transport import SiteConnection


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def exclusive_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('a+b')
    acquired = False
    try:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b'0');stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError('another experiment controller owns the lock') from None
        acquired = True
        yield
    finally:
        if acquired:
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def preflight(site):
    with SiteConnection(site) as c:
        health = c.run(dict(kind='oss', host=site['storage_nodes'][0]['management_ip']),
                       'timeout 12 idfs -s', timeout=15)
        tasks = c.run(dict(kind='client_nfs_rpc', host=site['control_host']),
            "ps -u hust -o comm=,args= | awk '$1 == \"java\" || $1 == \"vdbench\" || "
            "($1 ~ /^python/ && ($0 ~ /idfs_forecast_inference|idfs[-_]workloads?|python3 -B -/)) {print}'",
            timeout=15)
    healthy = health['exit'] == 0 and 'HEALTH_OK' in health['stdout'] and '96 up' in health['stdout']
    idle = tasks['exit'] == 0 and not tasks['stdout'].strip()
    return dict(healthy=healthy, idle=idle, checked_at=now(), health=health, tasks=tasks)


def initial_state(fingerprint, rounds):
    return dict(schema_version=1, config_hash=fingerprint, created_at=now(), state='ready',
                rounds=[dict(spec=r, state='pending', attempts=[]) for r in rounds])


def choose_round(state):
    # Preserve order: independent validation never overtakes training or a failed round.
    return next((r for r in state['rounds'] if r['state'] != 'captured_pending_qa'), None)


def tick(root, site_path, protocol_path, workloads_path, lock_path, execute=False,
         max_attempts=3, cooldown_s=300, stable_checks=2, interval_s=2,
         check=preflight, capture=run, clock=time.time, sleep=time.sleep):
    if max_attempts < 1 or cooldown_s < 0 or stable_checks < 1 or interval_s < 0:
        raise ValueError('invalid recovery policy')
    root = Path(root).resolve()
    site, protocol, workloads = load(site_path), load(protocol_path), load(workloads_path)
    rounds = development_rounds()
    fingerprint = canonical_hash(dict(site=site, protocol=protocol, workloads=workloads, rounds=rounds))
    if not execute:
        return dict(state='plan_only', root=str(root), rounds=len(rounds),
                    max_attempts=max_attempts, network_accessed=False, remote_work_started=False)
    with exclusive_lock(lock_path):
        root.mkdir(parents=True, exist_ok=True)
        path = root/'queue.json'
        state = load(path) if path.exists() else initial_state(fingerprint, rounds)
        if state['config_hash'] != fingerprint:
            raise ValueError('queue configuration changed: use a new batch root')
        item = choose_round(state)
        if item is None:
            state.update(state='captured_pending_qa', updated_at=now());save(path, state)
            return dict(state=state['state'])
        # A running marker survived process exit. Never mark it successful from
        # partial files, and never overwrite its directory. Remote checks follow.
        for old in item['attempts']:
            if old['state'] == 'running':
                old.update(state='interrupted', ended_at=now())
                item['state'] = 'retry_pending'
        if len(item['attempts']) >= max_attempts:
            state.update(state='retry_limit', current=item['spec']['run_id'], updated_at=now())
            save(path, state);return dict(state='retry_limit', run_id=item['spec']['run_id'])
        remaining = state.get('next_check_epoch', 0)-clock()
        if remaining > 0:
            save(path, state)
            return dict(state='cooldown', seconds_remaining=remaining)
        evidence = []
        try:
            for i in range(stable_checks):
                evidence.append(check(site))
                if not evidence[-1]['healthy'] or not evidence[-1]['idle']:
                    break
                if i+1 < stable_checks:
                    sleep(interval_s)
        except Exception as exc:
            evidence.append(dict(healthy=False, idle=False, error_type=type(exc).__name__))
        save(root/'last_preflight.json', evidence)
        if len(evidence) != stable_checks or not all(e['healthy'] and e['idle'] for e in evidence):
            state.update(state='waiting_environment', current=item['spec']['run_id'],
                         next_check_epoch=clock()+cooldown_s, updated_at=now())
            save(path, state)
            return dict(state='waiting_environment', run_id=item['spec']['run_id'])
        number = len(item['attempts'])+1
        # Include batch ID in the remote namespace. The inner legacy capture
        # runner derives remote output paths from this attempt directory name.
        attempt_root = root/(root.name+'__'+item['spec']['run_id']+'__attempt_%02d' % number)
        if attempt_root.exists():
            raise ValueError('unregistered attempt directory: do not overwrite evidence')
        attempt = dict(number=number, state='running', root=str(attempt_root), started_at=now())
        item['attempts'].append(attempt);item['state']='running'
        state.update(state='running', current=item['spec']['run_id'], updated_at=now())
        save(path, state)
        try:
            outcome = capture(attempt_root, site_path=site_path, protocol_path=protocol_path,
                              workloads_path=workloads_path, rounds=[item['spec']])
            attempt.update(state='captured_pending_qa', outcomes=outcome, ended_at=now())
            item['state']='captured_pending_qa'
            state.update(state='between_rounds', updated_at=now(), next_check_epoch=0)
        except Exception as exc:
            attempt.update(state='failed', error_type=type(exc).__name__, error=str(exc), ended_at=now())
            item['state']='retry_pending'
            state.update(state='cooldown', updated_at=now(), next_check_epoch=clock()+cooldown_s)
        save(path, state)
        return dict(state=state['state'], run_id=item['spec']['run_id'], attempt=number)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--site', required=True)
    parser.add_argument('--protocol', default='config/protocol.json')
    parser.add_argument('--workloads', default='config/workloads.json')
    parser.add_argument('--lock', default='.local/experiment_A.lock')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument('--cooldown-seconds', type=float, default=300)
    args = parser.parse_args()
    import json
    print(json.dumps(tick(args.out, args.site, args.protocol, args.workloads, args.lock,
                         args.execute, args.max_attempts, args.cooldown_seconds), ensure_ascii=False))
