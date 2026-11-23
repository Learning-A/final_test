"""Reconstruct the fixed R2 split from existing verified R1 inputs, without fitting."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

LAB = Path(__file__).resolve().parents[1] / 'idfs_forecast_lab'
sys.path.insert(0, str(LAB))
from idfs_lab.fine_inputs import SECOND, build_fine_input, index_fine
from idfs_lab.fine_paired_dataset import paired_capture, vector
from idfs_lab.fine_receipt import file_sha
from idfs_lab.joint import index_windows
from idfs_lab.masked_inputs import build_masked_input
from idfs_lab.r2_scaling import HOSTS, checked_input, contract_sha, feature_contract
from verify_r1_training_core_batch import PINS, REFERENCE


PROCESSING_MARGIN_NS = 50_000_000
SCHEMA_INDEX_SHA = 'f88ec8872529a97f001629a28fdcbcf738806eb96167ac201a3a2037256b73e2'
FILE_PINS = {name: PINS[name] for name in (
    'train', 'original_index', 'index', 'archive', 'reference_archive', 'sample_ids',
)} | {'schema_index': SCHEMA_INDEX_SHA}
PARTITION_SIZES = {'fit': 5328, 'internal_dev': 2661}
WORKLOADS = {'ai_training_ses', 'ai_inference_ses', 'bigdata_baleen', 'graph_zipf', 'hpc_wrf'}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def checked_member(archive, name, digest):
    contents = archive.read(name)
    check(hashlib.sha256(contents).hexdigest() == digest, f'checksum differs for {name}')
    return contents


def load_r2_inputs(paths):
    """Return fixed second × node arrays in the original sample-list order."""
    for name, digest in FILE_PINS.items():
        check(file_sha(paths[name]) == digest, f'pinned {name} checksum differs')
    source_schema = json.loads(Path(paths['schema_index']).read_bytes())
    for module, digest in source_schema['schema_modules_sha256'].items():
        check(Path(module).name == module and file_sha(LAB / 'idfs_lab' / module) == digest,
              f'frozen input construction differs for {module}')

    original = json.loads(Path(paths['original_index']).read_bytes())
    train_sets = {r['dataset_id']: r for r in original['datasets'] if r['role'] == 'train'}
    core = json.loads(Path(paths['index']).read_bytes())
    captures = core['captures']
    check(core['state'] == 'delivered' and len(captures) == len(train_sets) == 15
          and {c['dataset_id'] for c in captures} == set(train_sets), 'expected the original 15 training captures')
    check(Counter((c['workload'], c['index']) for c in captures)
          == Counter((workload, index) for workload in WORKLOADS for index in (1, 2, 3)),
          'expected three distinct captures per workload')

    contract = feature_contract()
    pinned = json.loads(Path(paths['sample_ids']).read_bytes())
    result = {}
    for part, seconds in PARTITION_SIZES.items():
        keys = [tuple(key) for key in pinned[part]]
        check(len(keys) == len(set(keys)) == seconds, f'fixed {part} sample count differs')
        result[part] = {
            'keys': keys,
            'features': np.empty((seconds, len(HOSTS), 258), dtype=np.float64),
            'actual': np.empty((seconds, len(HOSTS)), dtype=np.float64),
            'available_ns': np.empty(seconds, dtype=np.int64),
        }
    positions = {part: defaultdict(dict) for part in PARTITION_SIZES}
    for part in positions:
        for position, (capture, second) in enumerate(result[part]['keys']):
            positions[part][capture][second] = position
    check(set(positions['fit']).isdisjoint(positions['internal_dev']), 'capture leaked across partitions')
    check(set(positions['fit']) | set(positions['internal_dev']) == set(train_sets), 'sample list capture scope differs')

    coverage = {}
    with (zipfile.ZipFile(paths['train']) as training,
          zipfile.ZipFile(paths['archive']) as fine_core,
          zipfile.ZipFile(paths['reference_archive']) as fine_reference):
        for capture in captures:
            dataset_id = capture['dataset_id']
            metadata = train_sets[dataset_id]
            partition = 'internal_dev' if metadata['index'] == 3 else 'fit'
            check((capture['source_run_id'], capture['source_role'], capture['index'], capture['workload'])
                  == (metadata['run_id'], 'train', metadata['index'], metadata['workload']),
                  f'capture identity differs for {dataset_id}')
            check(capture['clock_uncertainty_ns'] is None and capture['online_receive_proof'] is False,
                  f'expected offline source timing for {dataset_id}')

            original_manifest = json.loads(checked_member(
                training, f'{dataset_id}/handoff_manifest.json', metadata['manifest_sha256']))
            check(original_manifest['role'] == 'train' and not original_manifest['fault_affected'],
                  f'original training capture needs review for {dataset_id}')
            original_members = {entry['path']: entry['sha256'] for entry in original_manifest['files']}
            original_windows = [json.loads(line) for line in checked_member(
                training, f'{dataset_id}/processed/windows.jsonl', original_members['processed/windows.jsonl']).splitlines()]
            recording_start = json.loads(training.read(f'{dataset_id}/recording_start.json'))['epoch_ns']

            delivery = capture['delivery']
            is_reference = dataset_id == REFERENCE
            check(delivery['kind'] == ('existing_release' if is_reference else 'archive_member'),
                  f'fine-data source differs for {dataset_id}')
            archive, prefix = (fine_reference, '') if is_reference else (fine_core, delivery['directory'] + '/')
            manifest = json.loads(checked_member(archive, prefix + 'export_manifest.json', capture['export_manifest_sha256']))
            member = next(entry for entry in manifest['files'] if entry['name'] == 'fine_context.jsonl.gz')
            fine_bytes = checked_member(archive, prefix + member['name'], member['sha256'])
            check(len(fine_bytes) == member['bytes'], f'fine-data length differs for {dataset_id}')
            fine_rows = [json.loads(line) for line in gzip.decompress(fine_bytes).splitlines()]

            expected = positions[partition].get(dataset_id)
            check(bool(expected), f'capture missing from fixed {partition} samples')
            paired = paired_capture(original_windows, fine_rows, capture_id=dataset_id,
                                    run_id=metadata['run_id'], source_role='train',
                                    run_index=metadata['index'], workload=metadata['workload'],
                                    hosts=HOSTS, recording_start_ns=recording_start,
                                    expected_starts=expected, processing_budget_ns=PROCESSING_MARGIN_NS)
            # Check semantic ordering from the source builders on actual accepted inputs.
            windows_index = index_windows(original_windows)
            fine_index = index_fine(fine_rows)
            start = min(expected)
            history_floor = ((recording_start + SECOND - 1) // SECOND) * SECOND
            for host in HOSTS:
                base = build_masked_input(windows_index, start, host, HOSTS, dataset_id, history_floor)
                near = build_fine_input(fine_index, start, host, dataset_id, history_floor,
                                        processing_budget_ns=PROCESSING_MARGIN_NS)
                check(base is not None and base['feature_names'] + near['feature_names'] == contract['names'],
                      f'feature names differ for {dataset_id}/{host}')

            output = result[partition]
            for sample in paired['samples']:
                position = expected[sample['window_start_ns']]
                check(sample['features_available_upper_ns'] <= sample['feature_deadline_ns']
                      and not sample['fine_evidence_online_ready'], f'input eligibility differs for {dataset_id}')
                output['features'][position] = [vector(sample, node, 'F0_F1') for node in range(len(HOSTS))]
                output['actual'][position] = sample['actual'][:len(HOSTS)]
                output['available_ns'][position] = sample['features_available_upper_ns']
            coverage[dataset_id] = {
                'workload': metadata['workload'], 'part': partition,
                'seconds': paired['accepted_seconds'], 'rejected': paired['rejected'],
                'observed_fine_slots': sum(sum(row['fine_observed']) for row in paired['samples']),
            }
    for split in result.values():
        checked_input(split['features'].reshape(-1, 258), contract['names'])
    return result, {
        'samples': PARTITION_SIZES, 'source_role': 'train', 'input_sha256': FILE_PINS,
        'contract_sha256': contract_sha(contract), 'coverage': coverage,
        'processing_margin_ns': PROCESSING_MARGIN_NS, 'online_time_eligibility': False,
        'fitting_performed': False, 'external_validation_read': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in FILE_PINS:
        parser.add_argument('--' + field.replace('_', '-'), type=Path, required=True)
    args = parser.parse_args()
    _, evidence = load_r2_inputs(vars(args))
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
