# Phase 11-A calibration preflight

Status: observer, offline simulation, aggregation and freezing implemented.
GPU launch still requires explicit authorization and a real formal-seed registry.

The selector registration is fixed in `scripts/calibrate_utility_replay.py`:
alpha `[0.25, 0.5]`, replay temperature `[0.5, 1, 2]`, W `[3, 10, 30]`,
M_min `[4, 32]`, in that Cartesian product order (36 candidates).
ESS fraction target is 0.75. Operational upper bounds are source mass 0.5,
repeat rate 0.9, added memory 128 MiB and sampling overhead 20 ms;
minimum uniform component is 0.25. These values were fixed without reading
calibration outcomes. TD3 target-update tau is not a candidate parameter.

Register before any collection or result loading:

```bash
python scripts/calibrate_utility_replay.py register --output /disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/calibration/return-blind-v1-preflight/candidates.json
```

Dedicated proposed seed: 104729. The implementation plan does not name the
three formal pilot seeds. `seed_preflight` refuses missing or conflicting
formal seeds; diagnostic-v1 seeds must not be assumed to be the future pilot
seed registry. Seed registration is therefore pending that explicit registry.

Proposed collection budget is 60 post-warmup generations, 10 warmup iterations,
population 32, one EA episode/member and one RL episode/generation on Swimmer.
At max episode length 1000 this bounds collection at 2.3 million environment
steps (2300 episodes), excluding separately specified evaluation. Diagnostic
interval is 10; residency must be captured every generation. Wall time is
unknown until a separately authorized measurement; no GPU run has started.

The read-only `utility_replay_calibration_capture.capture` interface takes a
real replay snapshot, frozen diagnostic records, independent measured timing
and memory. It copies metadata without reading training keys or payloads.
It records exact resident EA counts, all index-aligned sidecars, ring pointer,
capacity, raw three-kind scores, validity and measured generation. Invalid
scores may remain null. Separate simulation randomness is namespaced by
calibration seed and generation. `write_snapshot` refuses overwrites.

The observer attaches only to a validated uniform workflow after setup. It
uses the existing JIT rollout result and pre-insertion Replay to reconstruct
the source grouping for the frozen diagnostic. The baseline learner key is
never advanced by observation. No formal arm validation is relaxed. Each
step emits ring metadata; diagnostic checkpoints retain separate observer
score versions and original checkpoint ranks, never historical remeasurement.

The simulator reuses `compute_source_eligibility`,
`build_weighted_source_distribution` and `sample_weighted_ea_indices`.
It uses separate domain-tagged keys, preserves absent/expired scores as
neutral, rejects future or missing measured scores, and records 3 score-kind
measurements for each of 36 candidates at every analysis generation 20–70.
Coverage is conditional on measured resident mass; unscored mass is logged
separately. Aggregation is mean ESS/coverage, maximum mass/repeat/memory/time,
and all numerical checks. The aggregation protocol has its own SHA-256 in
the collection preregistration, without changing the original candidate file.

Timing distinguishes synchronized real uniform Replay sampling, frozen
diagnostic time, offline distribution construction (including compilation),
and synchronized conditional-EA sampling after compilation warmup. The
selection overhead_ms is the absolute time for 1024 conditional EA draws,
not a subtraction from the differently sized training batch or total step
time. Memory reports sidecar-copy bytes and deduplicated device-array bytes;
neither is claimed to be peak process/GPU allocator memory.
Measured offline metrics are written once under `offline_metrics/`, with
corpus/protocol identity and checksums. Repeating selection reuses those
frozen measurements rather than letting timing noise change a tie-break.

Synthetic unit-test outputs are temporary test-only artifacts.
No `calibration_metrics.csv`, `selection_trace.json` or `resolved_config.yaml`
may be published as a selected configuration before real data is complete.

Resource preflight (read-only): `nvidia-smi`, `free -h`, `df -h /disk1`, and
check existing tmux processes. Snapshot sidecars are 13 MB/generation before
compression at capacity one million; reserve at least 2 GiB for a 60-generation
corpus plus logs/metrics. Full payload copying is unnecessary.

Future commands (not executed by this implementation task), in tmux
`evorl-utility:review` and Conda `evorl-conflict`:

1. Resolve a dedicated uniform collection config with seed 104729,
   diagnostic_interval=10, warmup_iters=10, total_episodes=2300, JIT enabled,
   Swimmer and the registered 32-member population. Keep its scoring batch
   size and baseline learner settings fixed before collection.
2. Replace FORMAL_1/2/3 only with the actual approved formal seed registry:

```bash
python scripts/collect_utility_replay_calibration.py prepare --config COLLECTION_CONFIG.yaml --candidates /disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/calibration/return-blind-v1-preflight/candidates.json --formal-seeds FORMAL_1 FORMAL_2 FORMAL_3 --directory COLLECTION_DIRECTORY
python scripts/collect_utility_replay_calibration.py run --directory COLLECTION_DIRECTORY
python scripts/collect_utility_replay_calibration.py analyze --directory COLLECTION_DIRECTORY --formal-base-config FORMAL_BASE.yaml --output SELECTION_DIRECTORY
```

Prepare and run require clean committed code; run checks the preregistered
commit and checksums and refuses implicit resume. Analyze verifies all 60
generation files and requires a formal base config whose seed is in that
registry. It emits full arm YAMLs and a Phase 10 manifest alongside selection
artifacts. No evaluation is performed or logged by the collection runner.

Future selection command, after real metrics/provenance exist:

```bash
python scripts/calibrate_utility_replay.py select --registration CANDIDATES_JSON --metrics REAL_METRICS_CSV --provenance CORPUS_PROVENANCE_JSON --output /disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/calibration/return-blind-v1-selected
```

Resolved arm fields map to Phase 10 names. Analyze merges them into the given
shared formal base, checks/freezes all four via the Phase 10 manifest code,
and supplies `arm_A.yaml` through `arm_D.yaml` for the formal startup. No
calibration trajectory is claimed to predict candidate training returns.
