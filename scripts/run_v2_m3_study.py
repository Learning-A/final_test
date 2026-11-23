"""Run one short causal TCN with the fixed R2 L0 controls."""
import argparse
from pathlib import Path
import sys

from run_v2_m2_study import BUDGET, check_registration, r2, worker
from idfs_lab.v2_short_tcn import ShortCausalTCN

JOB = 'b19_v2_m3_short_causal_tcn_v1_a01'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    job = r2.require_training_host()
    config, paths, hashes = check_registration(job, JOB)
    if args.worker:
        worker(job, config, paths, hashes, arm='M3', model_factory=ShortCausalTCN,
               parameter_count=23457, use_workload_column=False)
    else:
        r2.check(not (job / 'results').exists(), 'Do not repeat M3')
        r2.run_leased([sys.executable, '-B', str(Path(__file__).resolve()), '--worker'], job_id=JOB,
                      owner_thread=r2.OWNER, code_sha=config['code_sha'], data_hashes=hashes,
                      budget_seconds=BUDGET, use_gpu=True)


if __name__ == '__main__':
    main()
