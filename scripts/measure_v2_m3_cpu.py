"""Measure the M0 and M3 model components after M3 training exits."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'idfs_forecast_lab'))
from idfs_lab.r2_loss_controls import R2SharedMLP
from idfs_lab.r2_scaling import feature_contract, transform
from idfs_lab.v2_short_tcn import ShortCausalTCN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    job = args.job
    target = job / 'results' / 'cpu_forward_timing.json'
    with target.open('x', encoding='utf-8') as output:
        report = json.loads((job / 'results' / 'report.json').read_bytes())
        if report['state'] != 'completed':
            raise ValueError('A completed training report is required')
        inputs = job / 'input'
        old = json.loads((inputs / 'r2_report.json').read_bytes())
        prep = json.loads((inputs / 'r2_preprocessing.json').read_bytes())
        before = old['arms']['L0']['checkpoint_sha256']
        after = report['result']['checkpoint_sha256']
        current = job / 'results' / 'M3' / 'checkpoint.pt'
        for path, pin in ((args.baseline, before), (current, after)):
            if hashlib.sha256(path.read_bytes()).hexdigest() != pin:
                raise ValueError('Checkpoint differs from recorded hash')
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        ids = json.loads((inputs / 'sample_ids.json').read_bytes())['internal_dev']
        meta = json.loads((inputs / 'metadata.json').read_bytes())
        positions = [i for i, key in enumerate(ids) if meta[key[0]]['workload'] == 'graph_zipf'][:100]
        if len(positions) != 100:
            raise ValueError('Required Graph sample count differs')
        with np.load(inputs / 'C_source_inputs.npz', allow_pickle=False) as arrays:
            raw = arrays['internal_dev_features'][positions]
        features = transform(raw.reshape(-1, 258), feature_contract()['names'], prep['scaling'])
        features = torch.as_tensor(features.astype(np.float32)).reshape(100, 8, 258)
        nodes = torch.arange(8)
        models = {'M0': R2SharedMLP(prep['target_scales']['s']), 'M3': ShortCausalTCN(prep['target_scales']['s'])}
        for name, path in (('M0', args.baseline), ('M3', current)):
            models[name].load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            models[name].eval()
        records = {name: [] for name in models}
        with torch.inference_mode():
            for x in features[:10]:
                for model in models.values():
                    model(x, nodes)
            for x in features:
                for name, model in models.items():
                    start = perf_counter_ns()
                    model(x, nodes)
                    records[name].append((perf_counter_ns() - start) / 1_000_000)
        json.dump(dict(scope='CPU forward for eight nodes on precomputed input, excludes feature preparation and transport',
                       cpu_threads=1, warmup_samples=10, graph_samples=100,
                       elapsed_ms=records,
                       percentiles={k: {'p50_ms': float(np.median(v)), 'p95_ms': float(np.percentile(v, 95))}
                                    for k, v in records.items()}), output, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
