# Local A/B resource supervisor

Implements the user-authorized protocol in A commit c492fb6, `LOCAL_DUAL_HEARTBEAT.md`.
The original `coordination/gpu_training.lock` is an existing shared gate. New
supervisors take shared nonblocking flock, then an exclusive per-UUID slot and
an exclusive owner-thread slot. Old exclusive trainers remain compatible and
serialize against every new supervisor. CPU fits reserve their owner's assigned
UUID slot while keeping CUDA invisible; GPU work uses that UUID only after an
idle check. Shared gate contents are never modified.

`idfs_lab.resource_slots.run_leased` requires the authorized b507 job placement,
fixed source/data hashes, CPU4 and at most 1800 seconds. It exclusively creates
a job record, launches one child process group, passes all lock descriptors to
the worker, records its PID, and waits. Success, failure, signal and timeout all
end the owned process group before releasing descriptors. Parent death leaves
the worker's inherited descriptors holding the locks. It does not kill unrelated
processes, delete locks, change the environment installation or launch workloads.

Real Linux tests ran on b507 in the fresh directory
`model_b_20261122T074416Z/lease_validation_local_b_20261123T0007` using the existing
tsfm Python. Five passed, zero failed/skipped. Tests use a temporary isolated
coordination directory, subprocesses without models and no GPU allocation:

- different slots coexist; same slot or same owner is refused;
- old exclusive gate blocks new tasks and shared new tasks block old exclusive;
- worker inheritance retains all locks after the calling context closes;
- symlinks are rejected and the legacy gate's bytes stay unchanged;
- failing and timed-out workers exit before their slots are reusable.

Command: `python -B -m unittest discover -s tests -p test_resource_slots.py -v`.
These are actual Linux flock/process tests, not model accuracy or live cluster evidence.
