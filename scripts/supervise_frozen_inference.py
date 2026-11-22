"""Use B's assigned slot for one CPU-only, no-fit frozen inference check."""
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'idfs_forecast_lab'))
from idfs_lab.resource_slots import run_leased


if __name__=='__main__':
    root=Path(os.environ['IDFS_GPU_JOB_ROOT']).resolve()
    command=[sys.executable,str(Path(__file__).with_name('run_frozen_inference_check.py')),
             '--frozen',str(root/'input/frozen.zip'),'--train',str(root/'input/train.zip'),
             '--original-index',str(root/'input/original-index.json'),
             '--core-index',str(root/'input/core-index.json'),'--archive',str(root/'input/r1-train-core.zip'),
             '--reference-archive',str(root/'input/graph-r0.zip'),'--output',str(root/'results')]
    run_leased(command,job_id=root.name,owner_thread='01a203f5-3aa6-7b21-8a1f-8fece4787432',
               code_sha=os.environ['E2E_CODE_SHA'],budget_seconds=180,use_gpu=False,
               data_hashes={'frozen':'76c81d14092b84bd80aaeadd69c477739a4c15df17e17a2c3fc43f1738655eb0',
                            'train':'889d303141802a9c826497eb3cf5f31ddac7c7a5d4db63c48698c5894d1d4425',
                            'core_index':'b4efd63c33aa44dd124c3553233857a5d9d896e6f4c53e42b2ed8f59747381ce',
                            'fine':'d6fc9b0e6227d4ea67fbc53c914702c278cf1d4ec2b8d68ec189a538eb9bd0f9',
                            'graph':'9f25bff3f4870ccb9f2b7e96e82d79d1d544dc4b627101061d881564a78d6752'})
