# Offline calibration implementation v2

Scope: offline analysis only. No rollout, learner, Utility, freshness, arm,
source-weight formula or formal Replay sampler is changed. The v1 simulator
remains available as the equivalence reference. Its input validation is
extracted unchanged into a shared pure helper.

## Reused work

For each snapshot, v2 validates and groups resident sources once, constructs
one device source map and EA mask, and builds three source tables. Eligibility
is cached for all registered `(kind, W, M_min)` combinations before candidate
measurement, so cache memory does not depend on candidate execution order or
resume position.

The original `build_weighted_source_distribution` computes source weights and
per-source transition probabilities using a compact identity mapping. V2
expands those exact values through the cached source map and applies the same
full-capacity prefix sum and final-CDF correction. It does not replace the
sampler with a different random algorithm. Original warmup and timed
`sample_weighted_ea_indices` calls, call count, key namespace and draw count
remain unchanged.

Finite, normalization, uniform-floor, neutral-source and sampled-origin checks
run on device using the original tolerances. Only small source summaries,
scalar results and eligibility are copied back. The full transition probability
vector is copied back only by equivalence tests.

Theoretical reductions, not a measured end-to-end speedup:

- Full source mapping and large device input uploads: once per generation
  rather than 108 times per generation.
- No per-measurement million-element probability download for validation.
- Large-array JIT expansion has no alpha/temperature static arguments; small
source computations still use the authoritative primitive and its cache.

The gather and prefix sum use separate executables to avoid a large fused
gather/scan compilation. Small source vectors are zero-padded to a power-of-two
block only at expansion/check boundaries; padded entries are never indexed and
do not enter source weights or ESS. All observed source counts 640–992 use the
same 1024-slot block, so the heavy expansion/check kernels do not specialize
on each source count. A cache regression test checks this reuse explicitly.

No complete v2 calibration has been run by the implementation task. Overall
speedup remains UNKNOWN until a separately authorized full offline run.

## Measurement semantics

`snapshot_preparation_ms` records device upload, source grouping and eligibility
preparation. `simulation_ms` records candidate distribution construction.
`overhead_ms` retains the original definition: one synchronized conditional-EA
sampler call after an identical-key warmup, with 1024 draws. Checkpoint writing
and progress logging are outside this timed region. These numbers are not
substitutes for real learner Replay sampling or whole-training overhead.

Memory counts deduplicated live buffers including resident offline caches and
warmup/timed samples. It is not allocator or process peak memory. Memory and
timing are implementation-specific measurements and are never relabelled as
v1 measurements. The original measured uniform Replay sampling and diagnostic
times remain attached to the corpus observations.

## Resume and provenance

Each generation/candidate writes a new, checksum-protected immutable record
after all three score kinds complete. Publication is atomic and exclusive.
Progress is flushed as JSONL after every candidate. A resume reuses a complete
record without rerunning its timed sampling. Partial unpublished records are
retained; they are not treated as completed measurements.

Resume requires the same analysis commit, code hashes, JAX/backend fingerprint,
corpus checksum, fixed candidate policy, protocol and formal base config.
Live-process locks prevent concurrent runs; stale locks are archived without
sending signals. An existing output is never overwritten.

The new entry point separates `collection_commit` from `analysis_commit`.
The old entry point's commit check is unchanged. `analysis_registration.json`
binds the immutable source corpus and its original preregistration to the new
implementation, code hashes, runtime, candidate policy and output directory.
The final artifact includes `analysis_version.json` and both commits in the
resolved configuration. `provenance.commit` identifies the new analyzer.

## Future commands — not executed by this implementation task

Use a new work/output directory under the external calibration root. Keep the
original corpus and original output location read-only. Prepare only after
committing this implementation and verifying a clean worktree:

```bash
python scripts/analyze_utility_replay_calibration_v2.py prepare \
  --directory /disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/calibration/phase11b-option-b-v1-seed104729-head8240863 \
  --formal-base-config /disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/calibration/phase11b-inputs-v1-head8240863/formal_base_seed1.yaml \
  --corpus-sha256 404814e829e3d395c3234bea5e3d34eda69da8d865946e1ee1184b8d0aad495c \
  --work-dir NEW_V2_WORK_DIRECTORY --output NEW_V2_SELECTION_DIRECTORY
python scripts/analyze_utility_replay_calibration_v2.py run --work-dir NEW_V2_WORK_DIRECTORY
```

Run in `evorl-utility:review` with the existing `evorl-conflict` Python and
`env -u LD_LIBRARY_PATH`. Preparation performs no simulation; `run` performs
offline simulation only. No Replay collection or environment interaction is
needed. Resume is an explicit rerun of the same registered `run` command.

The policy remains 36 candidates, generations 20–70, three score kinds,
unchanged warmup/timed sampling, ESS target 0.75 and original filters/tie-break.
Any actual mismatch, corrupted checkpoint or provenance change is an error;
there is no fallback that relaxes correctness or remeasures completed results.
