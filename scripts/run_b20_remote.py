"""b507 CPU lease wrapper for the B20 cluster expert comparison."""
import hashlib, os, subprocess, sys
from pathlib import Path
from idfs_lab.cpu_training_lease import cpu_lease

ROOT = Path(__file__).resolve().parent.parent
JOB = "b20_cluster_workload_expert_v1_a01"
CODE_SHA = "075131e4604af4f3dd04e01f4257bd7ab3b6b053"
def sh(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
data = {
    "i1": sh(ROOT / "input/inputs.npz"),
    "horizon": sh(ROOT / "input/prepared.npz"),
    "metadata": sh(ROOT / "input/metadata.json"),
    "sample_ids": sh(ROOT / "input/sample_ids.json"),
}
cmd = [sys.executable, "-B", str(ROOT / "run_cluster_workload_expert_cpu.py"),
       "--i1", str(ROOT / "input/inputs.npz"), "--horizon", str(ROOT / "input/prepared.npz"),
       "--metadata", str(ROOT / "input/metadata.json"), "--sample-ids", str(ROOT / "input/sample_ids.json"),
       "--output", str(ROOT / "output/cluster_workload_expert_metrics.json")]
os.environ.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="")
with cpu_lease(JOB, CODE_SHA, data, budget_seconds=1800):
    subprocess.run(cmd, check=True)
