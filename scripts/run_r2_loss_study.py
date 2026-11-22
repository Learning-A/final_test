"""Train the four registered R2 loss controls in one supervised b507 job."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

LAB = Path(__file__).resolve().parents[1] / 'idfs_forecast_lab'
sys.path.insert(0, str(LAB))
from idfs_lab.placement import require_training_host
from idfs_lab.r2_loss_controls import ARMS, SEED, R2SharedMLP, weighted_loss
from idfs_lab.r2_scaling import (
    HOSTS, feature_contract, fit_observed_scaling, fit_target_scales,
    run_equal_node_weights, transform,
)
from idfs_lab.resource_slots import COORD, OWNERS, run_leased
from prepare_r2_inputs import FILE_PINS, WORKLOADS, check, file_sha, load_r2_inputs
from run_r1_input_study import group_metrics


OWNER = '01a203f5-3aa6-7b21-8a1f-8fece4787432'
GPU = OWNERS[OWNER][1]
JOB_ID = 'b19_r2_loss_controls_v1_a01'
BUDGET_SECONDS = 1800
BATCH = 128
MAX_EPOCH = 60
PATIENCE = 8
LR = .001
DECAY = .0001
GRADIENT_CLIP = 5.0
INPUT_NAMES = {
    'train': 'train.zip', 'archive': 'r1-train-core.zip', 'reference_archive': 'graph-r0.zip',
    'original_index': 'original-index.json', 'index': 'core-index.json',
    'sample_ids': 'sample_ids.json', 'schema_index': 'schema-index.json',
}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def training_root():
    job = require_training_host()
    check(job.name == JOB_ID, 'R2 job identity differs')
    check(os.environ.get('CUDA_VISIBLE_DEVICES') == GPU, 'assigned B GPU required')
    lease_path = COORD / (job.name + '.json')
    lease = json.loads(lease_path.read_bytes())
    check(lease['state'] == 'running' and lease['pid'] == os.getpid()
          and lease['owner_thread'] == OWNER and lease['gpu_uuid'] == GPU
          and lease['code_sha'] == os.environ['E2E_CODE_SHA']
          and lease['data_sha256'] == FILE_PINS and lease['budget_seconds'] == BUDGET_SECONDS,
          'registered live R2 lease required')
    return job, lease


def configuration():
    return {
        'loss_arms': list(ARMS), 'seed': SEED, 'batch': BATCH, 'max_epoch': MAX_EPOCH,
        'patience': PATIENCE, 'learning_rate': LR, 'weight_decay': DECAY,
        'grad_clip': GRADIENT_CLIP, 'budget_seconds': BUDGET_SECONDS,
        'gpu_uuid': GPU, 'cpu_threads': 4, 'data_loader_workers': 0,
        'hosts': list(HOSTS), 'node_feature_count': 258, 'processing_margin_ms': 50,
        'checkpoint_metric': 'equal-weight macro MAPE over node and workload, earlier epoch wins ties',
    }


def configuration_sha():
    payload = json.dumps(configuration(), sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def supervise():
    job = require_training_host()
    if job.name != JOB_ID or (job / 'results').exists():
        raise ValueError('a fresh registered R2 job is required')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('deterministic CUDA workspace configuration required')
    for field, name in INPUT_NAMES.items():
        check(file_sha(job / 'input' / name) == FILE_PINS[field], f'pinned {field} checksum differs')
    run_leased([sys.executable, '-B', str(Path(__file__).resolve()), '--worker'],
               job_id=JOB_ID, owner_thread=OWNER, code_sha=os.environ['E2E_CODE_SHA'],
               data_hashes=FILE_PINS, budget_seconds=BUDGET_SECONDS, use_gpu=True)


def make_orders(population):
    # One epoch permutation shared by all arms, including an identical short final batch.
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    orders = [torch.randperm(population, generator=generator) for _ in range(MAX_EPOCH)]
    digest = hashlib.sha256()
    for order in orders:
        digest.update(order.numpy().astype('<i8', copy=False).tobytes())
    return orders, digest.hexdigest()


def selection_masks(actual, workloads):
    labels = np.asarray(workloads)
    check(actual.ndim == 2 and actual.shape == (len(labels), len(HOSTS)), 'bad metric dimensions')
    check(set(labels) == WORKLOADS, 'the original five workload groups are required')
    positive_groups = []
    for workload in sorted(set(labels)):
        for node in range(len(HOSTS)):
            rows = np.flatnonzero((labels == workload) & (actual[:, node] > 0))
            check(len(rows) > 0, f'MAPE undefined for {workload}/{HOSTS[node]}')
            positive_groups.append((rows, node, actual[rows, node]))
    return positive_groups


def macro_mape(prediction, groups):
    return float(np.mean([
        np.mean(np.abs(prediction[rows, node] - observed) / observed) * 100
        for rows, node, observed in groups
    ]))


def keep_checkpoint(score, best_score):
    # A tie keeps the earlier epoch, including epoch 0.
    if not np.isfinite(score):
        raise ValueError('nonfinite checkpoint score')
    return score < best_score


def inspect_predictions(prediction, actual, workloads):
    labels = np.asarray(workloads)
    result = {}
    for workload in sorted(set(labels)):
        mask = labels == workload
        true = actual[mask]
        predicted = prediction[mask]
        result[str(workload)] = {
            'nodes': {host: group_metrics(predicted[:, node], true[:, node])
                      for node, host in enumerate(HOSTS)},
            'cluster_sum': group_metrics(predicted.sum(axis=1), true.sum(axis=1)),
        }
        ordinary = true > 100
        if ordinary.any():
            diff = predicted[ordinary] - true[ordinary]
            result[str(workload)]['ordinary_actual_gt_100'] = {
                'n': int(ordinary.sum()), 'bias_iops': float(diff.mean()),
                'mae_iops': float(np.abs(diff).mean()),
                'wape_percent': float(100 * np.abs(diff).sum() / true[ordinary].sum()),
            }
    return result


def state_on_cpu(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def predict(model, features, nodes):
    model.eval()
    with torch.inference_mode():
        predictions = torch.cat([model(x, n) for x, n in zip(features.split(4096), nodes.split(4096))])
    output = predictions.cpu().numpy().astype(np.float64).reshape(-1, len(HOSTS))
    check(np.isfinite(output).all() and np.all(output >= 0), 'invalid predictions')
    return output


def save_predictions(path, keys, actual, prediction):
    check(len(keys) == len(actual) == len(prediction), 'prediction rows differ from the pinned sample list')
    with gzip.open(path, 'wt', encoding='utf8') as stream:
        for (capture, start), truth, forecast in zip(keys, actual, prediction):
            stream.write(json.dumps({'dataset_id': capture, 'window_start_ns': start,
                                     'actual': truth.tolist(), 'raw_prediction': forecast.tolist(),
                                     'final_prediction': forecast.tolist()}, allow_nan=False)+'\n')


def train_arm(arm, model, init_state, initial_hash, orders, fit, dev, scales, floors, groups, dest):
    started = time.monotonic()
    arm_dir = dest / arm
    arm_dir.mkdir()
    model.load_state_dict(init_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=DECAY)
    epoch_zero = predict(model, dev['x'], dev['nodes'])
    check(hashlib.sha256(epoch_zero.tobytes()).hexdigest() == initial_hash,
          'epoch-zero predictions differ between arms')
    best_epoch, best_score = 0, macro_mape(epoch_zero, groups)
    torch.save(init_state, arm_dir / 'checkpoint.pt')
    logs = [{'epoch': 0, 'dev_macro_mape': best_score, 'checkpoint_selected': True}]
    write_json(arm_dir / 'epochs.json', logs)

    population = len(fit['x'])
    for epoch, order in enumerate(orders, start=1):
        model.train()
        totals = {}
        gradient_norm_sum = torch.zeros((), device=fit['x'].device)
        component_norms = {}
        for batch_number, index in enumerate(order.to(fit['x'].device).split(BATCH)):
            nodes = fit['nodes'][index]
            prediction = model(fit['x'][index], nodes)
            loss, parts = weighted_loss(arm, prediction, fit['y'][index], scales[nodes],
                                        floors[nodes], fit['weights'][index], population_size=population)
            check(bool(torch.isfinite(loss)), f'{arm} epoch {epoch} has nonfinite loss')
            optimizer.zero_grad(set_to_none=True)
            if arm == 'L3' and batch_number == 0:
                parameters = tuple(model.parameters())
                for name, term in parts.items():
                    gradients = torch.autograd.grad(term, parameters, retain_graph=True)
                    norm = float(torch.stack([g.detach().square().sum() for g in gradients]).sum().sqrt())
                    check(np.isfinite(norm), f'nonfinite {name} gradient norm')
                    component_norms[name] = norm
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP, error_if_nonfinite=True)
            optimizer.step()
            gradient_norm_sum += norm.detach()
            for name, term in parts.items():
                totals[name] = totals.get(name, 0.0) + term.detach() * (len(index) / population)

        forecast = predict(model, dev['x'], dev['nodes'])
        score = macro_mape(forecast, groups)
        improved = keep_checkpoint(score, best_score)
        if improved:
            best_score, best_epoch = score, epoch
            checkpoint = arm_dir / 'checkpoint.pt'
            temporary = arm_dir / 'checkpoint.tmp'
            torch.save(state_on_cpu(model), temporary)
            temporary.replace(checkpoint)
        totals = {name: float(total) for name, total in totals.items()}
        component_norm_total = sum(component_norms.values())
        logs.append({'epoch': epoch, 'dev_macro_mape': score, 'checkpoint_selected': improved,
                     'train_loss_components': totals, 'train_loss': sum(totals.values()),
                     'mean_gradient_norm_before_clip': float(gradient_norm_sum) / (batch_number + 1),
                     'first_batch_component_gradient_norms': component_norms,
                     'first_batch_norm_share': {k: v / component_norm_total if component_norm_total else 0.0
                                                for k, v in component_norms.items()}})
        write_json(arm_dir / 'epochs.json', logs)
        print(json.dumps({'arm': arm, 'epoch': epoch, 'dev_macro_mape': score, 'selected': improved}), flush=True)
        if epoch - best_epoch >= PATIENCE:
            break

    checkpoint = arm_dir / 'checkpoint.pt'
    selected = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(selected)
    forecast = predict(model, dev['x'], dev['nodes'])
    check(abs(macro_mape(forecast, groups) - best_score) <= 1e-8, 'selected checkpoint score differs')
    predictions = arm_dir / 'predictions.jsonl.gz'
    save_predictions(predictions, dev['keys'], dev['actual'], forecast)
    scores = inspect_predictions(forecast, dev['actual'], dev['workloads'])
    result = {
        'best_epoch': best_epoch, 'macro_mape': best_score, 'epochs_trained': logs[-1]['epoch'],
        'internal_checkpoint_evaluations': len(logs), 'checkpoint_sha256': file_sha(checkpoint),
        'predictions_sha256': file_sha(predictions), 'epochs_sha256': file_sha(arm_dir / 'epochs.json'),
        'workload_metrics': scores, 'elapsed_seconds': time.monotonic() - started,
    }
    write_json(arm_dir / 'result.json', result)
    return result


def apply_screen(results):
    def summary(result):
        grouped = result['workload_metrics']
        worst = max(node['mape_percent'] for group in grouped.values() for node in group['nodes'].values())
        averages = {workload: float(np.mean([node['mae_iops'] for node in group['nodes'].values()]))
                    for workload, group in grouped.items()}
        count = sum(node['mape_percent'] < 10 for group in grouped.values() for node in group['nodes'].values())
        return worst, averages, count

    baseline = results['L0']
    worst, averages, _ = summary(baseline)
    for arm, results_arm in results.items():
        current_worst, current_averages, count = summary(results_arm)
        improvement = 1 - results_arm['macro_mape'] / baseline['macro_mape'] if baseline['macro_mape'] else None
        mae_pass = all(current_averages[workload] <= 1.05 * mae for workload, mae in averages.items())
        results_arm['screen'] = {'macro_mape_relative_improvement': improvement, 'worst_group_mape': current_worst,
                                 'workload_node_mean_mae': current_averages, 'groups_mape_lt_10': count,
                                 'passes': arm != 'L0' and improvement is not None and improvement >= .05
                                 and current_worst <= 1.05 * worst and mae_pass}


def train():
    job, lease = training_root()
    dest = job / 'results'
    dest.mkdir()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    check(torch.cuda.device_count() == 1, 'exactly one visible GPU required')
    device = torch.device('cuda:0')
    splits, evidence = load_r2_inputs({key: job / 'input' / name for key, name in INPUT_NAMES.items()})
    write_json(dest / 'input_evidence.json', evidence)
    names = feature_contract()['names']
    fit, dev = splits['fit'], splits['internal_dev']
    role = {'source_role': 'train', 'partition': 'fit_run01_02'}
    scaling = fit_observed_scaling(fit['features'].reshape(-1, 258), names, **role)
    target = fit_target_scales(fit['actual'], **role)
    states = {'scaling': scaling, 'target_scales': target, 'feature_contract': feature_contract()}
    write_json(dest / 'preprocessing.json', states)
    for split in (fit, dev):
        split['x'] = torch.from_numpy(transform(split.pop('features').reshape(-1, 258), names, scaling)
                                     .astype(np.float32)).to(device)
        check(bool(torch.isfinite(split['x']).all()), 'nonfinite float32 input')
        split['nodes'] = torch.arange(len(HOSTS), device=device).repeat(len(split['keys']))
    fit['y'] = torch.from_numpy(fit['actual'].reshape(-1).astype(np.float32)).to(device)
    check(bool(torch.isfinite(fit['y']).all()), 'nonfinite float32 target')
    fit_capture_ids = np.asarray([capture for capture, _ in fit['keys']])
    weights = run_equal_node_weights(fit_capture_ids)
    check(abs(float(weights.sum()) - 1) <= 1e-10, 'training weights must sum to one')
    run_weight = {capture: float(weights[fit_capture_ids == capture].sum())
                  for capture in sorted(set(fit_capture_ids))}
    check(all(abs(weight - .1) <= 1e-10 for weight in run_weight.values()), 'training run weights differ')
    workload_weight = {}
    for capture, weight in run_weight.items():
        workload = evidence['coverage'][capture]['workload']
        workload_weight[workload] = workload_weight.get(workload, 0.0) + weight
    check(len(workload_weight) == 5 and all(abs(weight - .2) <= 1e-10 for weight in workload_weight.values()),
          'training workload weights differ')
    fit['weights'] = torch.from_numpy(weights.reshape(-1).astype(np.float32)).to(device)
    dev['workloads'] = [evidence['coverage'][capture]['workload'] for capture, _ in dev['keys']]
    groups = selection_masks(dev['actual'], dev['workloads'])
    scales = torch.as_tensor(target['s'], dtype=torch.float32, device=device)
    floors = torch.as_tensor(target['tau'], dtype=torch.float32, device=device)
    check(bool(torch.isfinite(floors).all()), 'nonfinite float32 target floor')
    model = R2SharedMLP(target['s']).to(device)
    initial = state_on_cpu(model)
    initial_prediction_hash = hashlib.sha256(predict(model, dev['x'], dev['nodes']).tobytes()).hexdigest()
    orders, order_sha = make_orders(len(fit['x']))
    torch.save(initial, dest / 'epoch0.pt')
    results = {}
    metadata = {**configuration(), 'configuration_sha256': configuration_sha(),
                'code_sha': lease['code_sha'], 'job_id': lease['job_id'],
                'python': sys.version.split()[0], 'torch': str(torch.__version__), 'numpy': np.__version__,
                'input_sha256': FILE_PINS, 'preprocessing_sha256': file_sha(dest / 'preprocessing.json'),
                'epoch0_checkpoint_sha256': file_sha(dest / 'epoch0.pt'),
                'epoch0_prediction_sha256': initial_prediction_hash, 'batch_permutations_sha256': order_sha,
                'internal_dev_sample_count': len(dev['keys']), 'fit_sample_count': len(fit['keys']),
                'source_role': 'train',
                'fit_run_weights': run_weight, 'fit_workload_weights': workload_weight,
                'clock_uncertainty_ns': None,
                'external_validation_evaluations': 0, 'online_time_eligibility': False,
                'formal_acceptance': False, 'backbone_parameters': sum(p.numel() for p in model.parameters())}
    write_json(dest / 'report.json', {**metadata, 'state': 'partial', 'arms': results})
    for arm in ARMS:
        results[arm] = train_arm(arm, model, initial, initial_prediction_hash, orders, fit, dev,
                                 scales, floors, groups, dest)
        write_json(dest / 'report.json', {**metadata, 'state': 'partial', 'arms': results})
    apply_screen(results)
    write_json(dest / 'report.json', {**metadata, 'state': 'completed', 'arms': results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--supervise', action='store_true', help='check inputs and acquire the B GPU lease')
    mode.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    mode.add_argument('--configuration', action='store_true', help='print the fixed experiment settings without running')
    args = parser.parse_args()
    if args.supervise:
        supervise()
    elif args.configuration:
        print(json.dumps({**configuration(), 'configuration_sha256': configuration_sha()}, indent=2))
    else:
        train()


if __name__ == '__main__':
    main()
