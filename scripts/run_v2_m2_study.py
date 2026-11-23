"""Fit the registered M2 workload adapter with the original L0 training loop."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import run_r2_loss_study as r2
from run_v2_i1_study import screen, summary
from idfs_lab.r2_scaling import transform
from idfs_lab.v2_workload_adapter import WorkloadAdapterMLP


JOB = 'b19_v2_m2_workload_adapter_v1_a01'
WORKLOADS = ('ai_inference_ses', 'ai_training_ses', 'bigdata_baleen', 'graph_zipf', 'hpc_wrf')
BUDGET = 1800


def check_registration(job):
    config = json.loads((job / 'registration.json').read_bytes())
    r2.check(
        job.name == JOB and config['job_id'] == JOB and config['budget_seconds'] == BUDGET
        and config['code_sha'] == os.environ['E2E_CODE_SHA']
        and config['owner_thread'] == r2.OWNER and os.environ['CUDA_VISIBLE_DEVICES'] == r2.GPU
        and os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8',
        'Registered M2 job, source, owner, budget and GPU required',
    )
    paths, hashes = {}, {}
    for name, entry in config['inputs'].items():
        r2.check(Path(entry['file']).name == entry['file'], 'Input must be a named member')
        paths[name] = job / 'input' / entry['file']
        hashes[name] = entry['sha256']
        r2.check(r2.file_sha(paths[name]) == hashes[name], f'Input pin differs: {name}')
    return config, paths, hashes


def worker(job, config, paths, hashes):
    started = time.monotonic()
    np, torch = r2.np, r2.torch
    lease = json.loads((r2.COORD / (JOB + '.json')).read_bytes())
    r2.check(lease['state'] == 'running' and lease['pid'] == os.getpid()
             and lease['owner_thread'] == r2.OWNER and lease['gpu_uuid'] == r2.GPU
             and lease['code_sha'] == config['code_sha'] and lease['data_sha256'] == hashes
             and lease['budget_seconds'] == BUDGET, 'Matching active M2 lease required')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    r2.check(torch.cuda.device_count() == 1, 'Exactly one GPU required')
    device = torch.device('cuda:0')
    old = json.loads(paths['r2_report'].read_bytes())
    prep = json.loads(paths['preprocessing'].read_bytes())
    keys = json.loads(paths['samples'].read_bytes())
    meta = json.loads(paths['metadata'].read_bytes())
    names = r2.feature_contract()['names']
    r2.check(str(torch.__version__) == old['torch'] and np.__version__ == old['numpy']
             and prep['feature_contract'] == r2.feature_contract()
             and hashes['preprocessing'] == old['preprocessing_sha256']
             and hashes['baseline'] == old['arms']['L0']['predictions_sha256'],
             'Original runtime, C contract and L0 evidence must match')
    r2.check(not set(map(tuple, keys['fit'])) & set(map(tuple, keys['internal_dev'])),
             'Fitting and held-out keys overlap')
    parts = {}
    with np.load(paths['arrays'], allow_pickle=False) as arrays:
        for part, n, runs in (('fit', 5328, (1, 2)), ('internal_dev', 2661, (3,))):
            sample = keys[part]
            r2.check(len(sample) == len(set(map(tuple, sample))) == n
                     and all(meta[k[0]]['role'] == 'train' and meta[k[0]]['index'] in runs for k in sample),
                     'Sample identities or training roles differ')
            core, actual = arrays[part + '_features'], arrays[part + '_actual']
            r2.check(core.shape == (n, 8, 258) and actual.shape == (n, 8)
                     and np.isfinite(actual).all() and (actual >= 0).all(), 'C or label shape differs')
            r2.check(all(int(t) <= k[1] - 80_000_000 for t, k in zip(arrays[part + '_available_ns'], sample, strict=True)),
                     'Source cutoff differs')
            workloads = [meta[k[0]]['workload'] for k in sample]
            workload_indices = np.repeat([WORKLOADS.index(w) for w in workloads], 8)
            scaled = transform(core.reshape(-1, 258), names, prep['scaling'])
            x = np.column_stack((scaled, workload_indices)).astype(np.float32)
            parts[part] = dict(keys=sample, actual=actual, workloads=workloads,
                               x=torch.from_numpy(x).to(device), nodes=torch.arange(8, device=device).repeat(n))
    fit, dev = parts['fit'], parts['internal_dev']
    weights = r2.run_equal_node_weights([k[0] for k in fit['keys']])
    r2.check(abs(weights.sum() - 1) < 1e-10, 'Original weighting required')
    fit['weights'] = torch.from_numpy(weights.reshape(-1).astype(np.float32)).to(device)
    fit['y'] = torch.from_numpy(fit['actual'].reshape(-1).astype(np.float32)).to(device)
    with gzip.open(paths['baseline'], 'rt', encoding='utf-8') as stream:
        originals = list(map(json.loads, stream))
    r2.check([(row['dataset_id'], row['window_start_ns']) for row in originals] == list(map(tuple, dev['keys'])),
             'Frozen baseline and internal keys must match exactly')
    actual = np.asarray([row['actual'] for row in originals])
    frozen = np.asarray([row['final_prediction'] for row in originals])
    r2.check(np.array_equal(actual, dev['actual']), 'Original labels differ')
    baseline = summary(np, frozen, actual, dev['workloads'])
    r2.check(abs(baseline['macro_mape_percent'] - old['arms']['L0']['macro_mape']) < 1e-9, 'L0 scores differ')
    ts = prep['target_scales']
    scales = torch.as_tensor(ts['s'], dtype=torch.float32, device=device)
    floors = torch.as_tensor(ts['tau'], dtype=torch.float32, device=device)
    model = WorkloadAdapterMLP(ts['s']).to(device)
    r2.check(sum(parameter.numel() for parameter in model.parameters()) == 21414, 'M2 parameter count differs')
    initial = r2.state_on_cpu(model)
    initial_predictions = r2.predict(model, dev['x'], dev['nodes'])
    initial_hash = hashlib.sha256(initial_predictions.tobytes()).hexdigest()
    r2.check(initial_hash == old['epoch0_prediction_sha256'], 'Zero adapters must reproduce original epoch zero')
    orders, batch_hash = r2.make_orders(len(fit['x']))
    r2.check(batch_hash == old['batch_permutations_sha256'], 'Original batch sequence required')
    groups = r2.selection_masks(actual, dev['workloads'])
    dest = job / 'results'
    dest.mkdir()
    result = r2.train_arm('L0', model, initial, initial_hash, orders, fit, dev, scales, floors, groups, dest)
    (dest / 'L0').rename(dest / 'M2')
    with gzip.open(dest / 'M2' / 'predictions.jsonl.gz', 'rt', encoding='utf-8') as stream:
        prediction = np.asarray([row['final_prediction'] for row in map(json.loads, stream)])
    candidate = summary(np, prediction, actual, dev['workloads'])
    report = dict(state='completed', job_id=JOB, code_sha=config['code_sha'], budget_seconds=BUDGET,
        elapsed_seconds=time.monotonic()-started, inputs=config['inputs'], parameters=21414, new_fit_count=1,
        new_preprocessing_fit_count=0, baseline_fit_count=0, external_validation_evaluations=0,
        online_time_eligibility=False, conditional_known_workload=True, workload_order=WORKLOADS,
        batch_permutations_sha256=batch_hash, epoch0_prediction_sha256=initial_hash,
        arms={'M0_reused': baseline, 'M2': candidate}, comparison=screen(candidate, baseline), result=result)
    r2.write_json(dest / 'report.json', report)
    print('COMPLETED', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    job = r2.require_training_host()
    cfg, paths, hashes = check_registration(job)
    if args.worker:
        worker(job, cfg, paths, hashes)
    else:
        r2.check(not (job / 'results').exists(), 'Do not repeat M2')
        r2.run_leased([sys.executable, '-B', str(Path(__file__).resolve()), '--worker'], job_id=JOB,
            owner_thread=r2.OWNER, code_sha=cfg['code_sha'], data_hashes=hashes, budget_seconds=BUDGET, use_gpu=True)


if __name__ == '__main__':
    main()
