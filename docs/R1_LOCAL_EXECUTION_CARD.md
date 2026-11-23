# B19 R1 fixed offline input experiment

experiment_id=b19-r1-causal-inputs-offline-v1; job_id=b19_r1_inputs_20261123_a01.
Owner B, thread 01a203f5-3aa6-7b21-8a1f-8fece4787432. User transferred old B
submission authority to this local thread. No new collection or deployment.

Continue the existing card and verified receipt at 24c2b9c. Data are the pinned
15 train captures, train01/02 fit (5,328 seconds), train03 internal holdout (2,661
seconds), source role unchanged. Same exact P2 sample IDs, labels and ordering.
No validation ZIP is opened, no external validation model evaluation is added.

Primary extra source-clock margin is fixed at **50 ms**. Report **100 and 200 ms**
as sensitivity variants without selecting the best variant. All use the same
old sample keys; any changed sample set aborts, rather than silently dropping
difficult windows. Margins are hypothetical source-time availability budgets;
unknown clock uncertainty remains null. They do not establish online receipt,
measured processing budget or formal 30ms eligibility.

F0 reuses all nine frozen base144 P2 models (report SHA
16b13ccd4a0181edc2b266fe531c983c0eb4f28136991180683bf177ae63422b) and must reproduce
every old internal-dev prediction within relative 1e-10. F0+F1 adds each node's
114 fine features, missingness mask retained, no fine client/pressure/long60.
For each margin, fit nine ExtraTrees models, 64 trees/depth8/leaf3/seed2026,
raw-scale squared loss, CPU4/workers0, no calibration or parameter search.
There are exactly 27 new fits, all within one 1800-second total worker budget.
GPU is invisible. B reserves its assigned UUID slot using the shared legacy
gate plus exclusive UUID/owner locks; descriptors cover the worker lifetime.

Before submission register source SHA and command in #19. New authorized root:
`/home/lmj/idfs_forecast_lab/model_b_20261122T074416Z/b19_r1_inputs_20261123_a01`.
Existing Python: `/home/lmj/miniconda3/envs/tsfm/bin/python`.
Entry: `python -B source/scripts/supervise_r1.py`, with IDFS_GPU_JOB_ROOT set to
the job directory, E2E_CODE_SHA set to the registered commit, CUDA_VISIBLE_DEVICES
empty and OMP/MKL/OPENBLAS/NUMEXPR threads all 4.

Stop on hash/label/sample disagreement, unavailable slot, failed F0 parity,
time or resource conflict, or hard total timeout. Never repeat this job ID,
expand the budget, change existing environments or kill other processes.

Report each workload/node MAPE (actual-zero exclusion only), MAE/WAPE, zero and
sample counts, low-positive APE bands, worst group, paired changes and coverage;
cluster direct and node-sum are secondary. Raw/final predictions, models,
hashes, fit loss and serialization parity are retained. Latency numbers only
measure model calls and exclude feature construction/receipt/publication.

Primary research screen stays fixed: node macro MAPE relative improvement >=5%,
worst node-group MAPE ratio <=1.05, every workload node macro MAE ratio <=1.05.
Failure is recorded without changing losses/capacity/thresholds to rescue the
card. Passing this research screen would still not establish formal accuracy.
A independently recomputes delivered frozen predictions before adoption.

## Report-write recovery (same experiment, no repeated fits)

The first worker completed all nine 50ms fits, then failed at JSON report writing
because a NumPy scalar entered a reduction. Its failed record, logs and nine
weights stay intact. Elapsed supervised time was 81.748 seconds; charge 82 seconds
against the original total budget. Continuation job `b19_r1_inputs_20261123_a02`
gets at most **1718 seconds**, so the combined worker budget cannot exceed 1800.
This is a partial-result recovery, not an automatic rerun of completed training.

Fix only conversion to native scalar types. Pin the nine existing primary weights
and their original fit times, validate model specification and reuse their bytes
without fitting. Reconstruct primary predictions from these weights; fit only
the remaining eighteen 100/200ms models. Do not change margins, sample keys,
model/loss, selection rules or source-clock qualification. Register the repair
commit and recovery-index SHA before launch. A new real-NumPy persistence test
must pass. A second fault stops the card with partial evidence rather than
another automatic continuation.
