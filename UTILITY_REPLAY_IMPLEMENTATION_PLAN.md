# Learner-Utility-Aware Replay Pilot Implementation Plan

> **Execution mode:** Native + tests-first. Execute this plan in one continuous native context; do not use subagent-driven implementation because the replay, JAX, PRNG, and workflow phases share tightly coupled state and invariants. Execution is not authorized by this document alone.

**Status:** Implementation plan for review. No algorithm implementation, intervention run, or push is authorized.

**Goal:** Implement the approved `UTILITY_REPLAY_DESIGN.md` v1.1 as a pilot-only, source-aware replay overlay while preserving Original ERL behavior, budgets, EA/RL replay mixture, and baseline PRNG progression.

**Architecture:** Add a pilot-specific replay wrapper with provenance sidecars, a pure source-scoring and conditional-EA sampling module, and an `ERLWorkflow`-derived pilot workflow. The baseline sampler always makes the full uniform draw first. The overlay may replace only positions whose baseline draw selected EA-origin transitions. Frozen learner-utility diagnostics supply utility scores read-only; they are reused, not reimplemented.

**Tech stack:** Python 3.11, JAX/JAXLIB 0.9.0, Equinox, Optax, Hydra, Brax, pytest, NumPy/SciPy/Pandas for host-side audit analysis.

**Spec:** `UTILITY_REPLAY_DESIGN.md` v1.1 at commit `15462a025d44af56ca0791336d15a104ec2d8b06`.

## Global constraints

- Work only on `research/learner-utility-replay-pilot`, based on annotated tag `learner-utility-diagnostic-v1`.
- Use `/disk1/zmh_projects/evorl`, tmux `evorl-conflict-research:0.0`, conda `evorl-conflict`, and run Python as `env -u LD_LIBRARY_PATH ...`.
- Use Native/tests-first development. Each implementation phase begins with a focused failing test and ends with focused tests plus the applicable cumulative gates.
- Do not modify Original ERL baseline source. Do not modify frozen diagnostic-v1 behavior.
- Do not alter interaction count, learner-update count, replay batch size/capacity, population operations, evaluation checkpoints, or evaluation episode count.
- Do not implement eviction, fitness modification, EA selection changes, or utility-aware population selection.
- Do not tune using evaluation return, AUC, final return, or formal pilot outcomes.
- Do not push unless separately authorized. Do not start Phase 13 runs without separate approval after smoke review.

## Review focus

Reviewers should verify that the plan makes `alpha=0` an exact bypass, preserves realized EA/RL slots, keeps intervention randomness outside the baseline key schedule, implements the Design v1.1 probability law exactly, and prevents future phases from continuing after a failed gate.

---

## Planned files and responsibilities

### New production files

| File | Responsibility |
|---|---|
| `evorl/replay_buffers/source_aware_replay_buffer.py` | Pilot-specific replay state, provenance sidecars, aligned ring insertion, baseline index draw/gather adapter. |
| `evorl/algorithms/erl/erl_td3/source_replay.py` | Pure score-table, rank, freshness, eligibility, analytical probability, overlay-index, and sampler-audit functions. |
| `evorl/algorithms/erl/erl_td3/erl_utility_replay.py` | Pilot workflow orchestration; reuses Original ERL and frozen diagnostics while owning pilot state and intervention keys. |
| `configs/agent/erl/erl-utility-replay.yaml` | A/B/C/D arm, `alpha`, `tau`, `W`, `M_min`, diagnostics, audit, and resolved-calibration configuration. |
| `scripts/calibrate_utility_replay.py` | Return-blind deterministic calibration, candidate filtering, tie-breaking, and resolved-config freeze. |
| `scripts/analyze_utility_replay_pilot.py` | Post-run matched-budget AUC, final-return, gate, and audit summaries; not used for calibration. |

Direct imports should be used so `evorl/replay_buffers/__init__.py` need not change. If an export is later proven necessary, stop and add that file explicitly to the reviewed scope before editing it.

### New tests

| File | Responsibility |
|---|---|
| `tests/test_source_aware_replay_buffer.py` | Sidecar alignment, masking, wraparound, provenance, baseline draw, and gather. |
| `tests/test_source_replay.py` | Rank/freshness/eligibility math, probabilities, overlay invariants, deterministic and empirical sampling. |
| `tests/test_erl_utility_replay_workflow.py` | Lifecycle, exact workflow equivalence, PRNG/state isolation, budgets, and diagnostic reuse. |
| `tests/test_utility_replay_calibration.py` | Return-blind schema, filters, ESS target, tie-break, and freeze manifest. |

### Protected files: read-only throughout implementation

Original ERL baseline:

```text
evorl/algorithms/erl/erl_td3/erl_origin.py
evorl/algorithms/erl/erl_td3/erl_ga.py
evorl/algorithms/erl/erl_td3/erl_td3_workflow.py
evorl/replay_buffers/replay_buffer.py
```

Frozen diagnostic-v1 assets:

```text
evorl/algorithms/erl/erl_td3/erl_learner_utility.py
evorl/algorithms/erl/erl_td3/learner_utility.py
configs/agent/erl/erl-learner-utility.yaml
tests/test_erl_learner_utility_workflow.py
```

At every phase boundary run:

```bash
git diff -- \
  evorl/algorithms/erl/erl_td3/erl_origin.py \
  evorl/algorithms/erl/erl_td3/erl_ga.py \
  evorl/algorithms/erl/erl_td3/erl_td3_workflow.py \
  evorl/replay_buffers/replay_buffer.py \
  evorl/algorithms/erl/erl_td3/erl_learner_utility.py \
  evorl/algorithms/erl/erl_td3/learner_utility.py \
  configs/agent/erl/erl-learner-utility.yaml \
  tests/test_erl_learner_utility_workflow.py
```

Expected output: empty. Any output is a hard STOP.

## Core implementation boundaries

### Proposed public data and function interfaces

```python
class ReplayProvenance(NamedTuple):
    origin_type: jax.Array
    source_token: jax.Array
    source_generation: jax.Array
    score_version: jax.Array

class SourceAwareReplayBufferState(NamedTuple):
    data: PyTree
    current_index: jax.Array
    buffer_size: jax.Array
    origin_type: jax.Array
    source_token: jax.Array
    source_generation: jax.Array
    score_version: jax.Array

class SourceScoreTable(NamedTuple):
    source_token: jax.Array
    raw_score: jax.Array
    rank_normalized_score: jax.Array
    measured_generation: jax.Array
    score_kind: jax.Array
    validity: jax.Array
    resident_count: jax.Array
    weighted_eligible: jax.Array
    version: jax.Array

def add_with_provenance(state, transitions, provenance, mask=None): ...
def draw_baseline_indices(state, key, batch_size): ...
def gather_by_indices(state, indices): ...
def build_source_scores(records, score_kind, generation, version): ...
def compute_source_distribution(table, current_generation, alpha, tau, window, min_resident): ...
def overlay_ea_indices(baseline_indices, state, distribution, intervention_key, alpha): ...
```

Names may change only if tests retain the same semantic contracts. The score table is bounded and contains only sources that may remain resident. Ring overwrites update `resident_count`; expired or under-resident sources become neutral and are never deleted merely for being ineligible.

### JAX/JIT boundary

- JAX-pure, fixed-shape, JIT-compatible: ring-index computation, sidecar insertion, resident-count updates, rank/probability arrays, EA-slot masks, conditional replacement indices, gather, and sampler metrics required during training.
- Host/eager: converting diagnostic records into checkpoint score updates, artifact/CSV/JSON writing, calibration candidate enumeration, resolved-config hashing, and post-run statistics.
- No Python dictionaries, variable-length lists, Pandas objects, filesystem calls, or host callbacks enter a jitted learner step.
- The score table and replay sidecars are PyTrees of fixed-shape arrays. Shape/capacity is resolved before compilation.
- Avoid building a temporary `[batch_size, replay_capacity]` mask or weight matrix. A cached one-dimensional EA transition CDF/weight vector may be rebuilt only at score/freshness-version boundaries and sampled with fixed-shape primitives.
- Record compile time separately from steady-state sampling time. All correctness tests run once with JIT enabled and once with `JAX_DISABLE_JIT=1` where practical.

### Sidecar memory estimate and guard

Recommended dtypes per replay slot:

```text
origin_type       int8    1 byte
source_token      int32   4 bytes, -1 sentinel for RL
source_generation int32   4 bytes
score_version     int32   4 bytes
                         --------
raw sidecars              13 bytes/slot
```

For capacity 1,000,000 this is 13,000,000 bytes (12.40 MiB) before allocator alignment. A cached `float32` transition-weight/CDF array adds 3.81 MiB. Budget the steady additional device allocation at at most 24 MiB plus the bounded score table; prohibit batch-by-capacity temporaries. Phase 1 measures actual allocation and records it. `source_token = generation * population_size + individual_id` must be range-checked against signed int32; overflow is a configuration error, not silent wrapping.

### PRNG ownership

- `baseline_key`: owned by inherited Original ERL state and advanced exactly as baseline for rollouts, uniform replay draw, learner updates, and evaluation.
- `diagnostic_key`: derived/managed by the frozen read-only diagnostic path and never written back to baseline key progression.
- `intervention_key`: explicit pilot-only state, seeded deterministically from the run seed plus a versioned namespace. It is used only for B/C/D EA replacements.
- `alpha=0`: bypasses overlay sampling, returns baseline indices directly, and does not consume or advance `intervention_key`.
- `B_ref_key` and optional `B_eval_key`: separately namespaced and independently folded; tests require distinct keys and fingerprints.

## Executable gate matrix

| Gate | Required verification | Blocks |
|---|---|---|
| 1 metadata alignment | Phase 1 focused tests | all later phases |
| 2 RL-origin invariance | Phase 2 tests | Phase 3+ |
| 3 EA/RL mixture invariance | Phase 2 tests | Phase 3+ |
| 4 `alpha=0` sampler equivalence | bitwise index/batch/key tests | weighted sampling |
| 5 workflow equivalence | deterministic tree-wise comparison after every step | weighted sampling and experiments |
| 6 probability correctness | analytical, edge-case, deterministic, empirical tests | real overlay |
| 7 anti-circularity | independent-key/fingerprint/dataflow tests | smoke/pilot |
| 8 budget equivalence | four-arm deterministic counters and manifest validation | smoke/pilot |

The first failing command stops execution. Do not skip, weaken, xfail, or replace a gate with visual inspection.

---

## Phase 0 — Baseline audit

**Files read:** protected files above, `evorl/replay_buffers/replay_buffer.py`, diagnostic modules, existing diagnostic tests/config.

**Temporary output:** `.codex_validation/utility_replay_pilot/baseline_audit.md`.

1. Trace and document exact call sites for replay state initialization, add, uniform index generation, gather, EA-origin insertion, RL-origin insertion, learner batch construction, learner update, and diagnostic score production.
2. Record array shapes, masks, flatten order, key ownership, and whether each call is inside JIT/scan.
3. Prove the proposed subclass/composition seams can intercept insertion and sampling without changing protected files.
4. Add a pre-implementation checksum/diff manifest for protected files.

**RED test:** none; this is a read-only prerequisite. The audit is incomplete until each listed dataflow node has a file, symbol, input/output shape, and key owner.

**GREEN evidence:** reviewed audit plus empty protected-file diff.

**STOP:** if composition cannot preserve exact baseline index draw or provenance cannot be attached before flattening, report the concrete blocker. Do not edit baseline to work around it.

**Commit:** none; `.codex_validation` remains untracked.

## Phase 1 — Provenance sidecar (Design Gate 1)

**Modify:** create `source_aware_replay_buffer.py` and `test_source_aware_replay_buffer.py` only.

**Dependencies:** Phase 0 green.

### RED

Add these tests first:

```text
test_masked_add_keeps_payload_and_provenance_aligned
test_wraparound_overwrite_keeps_payload_and_provenance_aligned
test_ea_add_encodes_generation_and_individual_token
test_rl_add_uses_reserved_source_sentinel
test_source_token_int32_overflow_is_rejected
test_sidecar_memory_estimate_stays_within_registered_limit
```

Run:

```bash
python -m pytest -p no:cacheprovider -q tests/test_source_aware_replay_buffer.py
```

Expected RED: imports/behavior absent, not unrelated environment failure.

### GREEN

Implement the minimal pilot replay state and aligned `add_with_provenance`. Reuse baseline ring-index semantics exactly. Apply the same insertion indices and mask to payload and every sidecar. Count only written elements during overwrite bookkeeping.

Rerun the focused file, then existing replay-buffer tests.

**GREEN evidence:** all Phase 1 tests pass; device allocation delta and dtype table are written to the validation audit.

**STOP:** any payload/metadata mismatch, silent token overflow, or memory beyond the registered cap.

**Commit allowed:** yes, only after Gate 1 and protected-file diff pass. Suggested message: `feat: add replay provenance sidecars`.

## Phase 2 — Baseline-draw overlay skeleton (Design Gates 2–3)

**Modify:** create `source_replay.py`; extend the two Phase 1 test files only as appropriate.

**Dependencies:** Gate 1.

### RED tests

```text
test_overlay_skeleton_preserves_all_rl_indices
test_overlay_skeleton_preserves_ea_slot_positions
test_overlay_skeleton_preserves_realized_ea_rl_counts
test_overlay_skeleton_returns_baseline_indices_before_weighting
```

Construct mixed EA/RL sidecars with fixed baseline indices. Assertions compare indices and Boolean slot masks exactly.

### GREEN

Implement only:

```text
baseline indices -> origin lookup -> EA slot mask -> unchanged indices
```

Do not add weighted probabilities or extra random draws.

**STOP:** any RL index changes or EA slot position/count differs.

**Commit allowed:** yes after Gates 2–3 pass. Suggested message: `feat: add baseline replay overlay skeleton`.

## Phase 3 — `alpha=0` exact equivalence (Design Gates 4–5; hard STOP)

**Modify:** create `erl_utility_replay.py`, `erl-utility-replay.yaml`, and `test_erl_utility_replay_workflow.py`; extend source tests.

**Dependencies:** Gates 1–3.

### RED sampler tests

```text
test_alpha_zero_indices_are_bitwise_baseline_indices
test_alpha_zero_batches_are_treewise_bitwise_equal
test_alpha_zero_baseline_prng_progression_is_equal
test_alpha_zero_does_not_advance_intervention_key
```

### RED workflow test

```text
test_alpha_zero_tiny_workflow_is_exactly_baseline_equivalent
```

For a deterministic tiny run, compare after every step:

```text
actor, target actor, critic, target critic
critic/actor optimizer states
replay payload/current_index/buffer_size
EC state
baseline PRNG state
metrics and learner/population update counts
```

Pilot-only metadata/timing fields must be enumerated and excluded by exact path, never by broad subtree suppression.

### GREEN

Implement the smallest workflow wrapper and `alpha == 0` branch. It must return the original baseline index array object/value before intervention sampling. The baseline draw remains the one and only baseline draw.

Run focused tests, frozen diagnostic tests, and the repository regression command with the documented pre-existing MJX exclusion only if still necessary.

**HARD STOP:** any unequal baseline field, consumed intervention key, protected-file diff, or unexplained regression. Weighted sampling work is forbidden until resolved.

**Commit allowed:** yes only after Gates 4–5 and regression pass. Suggested message: `feat: add exact-equivalent utility replay workflow`.

## Phase 4 — Source score table

**Modify:** `source_replay.py`, `test_source_replay.py`, and minimal workflow wiring tests.

**Dependencies:** hard Gate 5.

### RED tests

```text
test_score_table_stores_required_fields
test_rank_normalization_is_within_checkpoint
test_rank_normalization_uses_deterministic_average_ties
test_rank_normalization_maps_higher_score_to_higher_priority
test_score_table_supports_fitness_td_and_utility_kinds
test_score_table_version_is_monotonic
test_utility_scores_reuse_frozen_diagnostic_records
test_ring_overwrite_updates_resident_counts
test_score_slot_reclaimed_only_after_resident_count_zero
test_score_slot_reuse_clears_all_stale_metadata
test_score_slot_allocation_is_deterministic
test_score_table_full_with_all_sources_resident_has_explicit_failure
test_score_table_capacity_is_fixed_before_jit
```

### GREEN

Implement a bounded fixed-shape `SourceScoreTable`. Accept frozen diagnostic records at the host boundary; do not recalculate learner utility. Fitness, high-TD, and utility all pass through the same normalization function. Update resident counts from the exact add/overwrite indices.

`score_table_capacity` is a static configuration value resolved before JIT. For the V1 pilot it defaults to `replay_buffer_capacity`: every resident source owns at least one resident transition, so the number of resident sources cannot exceed replay capacity. A smaller explicitly configured capacity is allowed only for calibration/smoke when its bound is documented; reaching it is an explicit safe failure, never a reason to resize under JIT.

Slot lifecycle is fixed and deterministic:

- a slot is reclaimable only when `resident_count == 0`, meaning no replay index still references that source;
- a new source takes the lowest-index reclaimable slot;
- slot reuse overwrites or clears every field: `source_token`, `raw_score`, `rank_normalized_score`, `measured_generation`, `score_kind`, `validity`, `resident_count`, `weighted_eligible`, and `version`;
- if the table is full and every slot still has resident transitions, insertion raises an explicit capacity error and the run stops safely;
- there is no dynamic growth, Python unordered allocation, or silent overwrite of a resident source.

**STOP:** ranking across checkpoints, reimplementation of utility, inconsistent tie handling, negative resident counts, or unbounded table growth.

**Commit allowed:** yes. Suggested message: `feat: add replay source score table`.

## Phase 5 — Unified freshness and resident eligibility

**Modify:** `source_replay.py`, `test_source_replay.py`.

**Dependencies:** Phase 4.

### RED tests

```text
test_fresh_valid_source_with_minimum_residency_is_eligible
test_expired_source_is_neutral
test_insufficient_resident_source_is_neutral
test_missing_or_nan_score_is_neutral
test_empty_eligible_set_reduces_to_baseline_conditional_distribution
test_single_eligible_source_preserves_ineligible_baseline_mass
test_mixed_eligible_ineligible_sources_preserve_neutral_mass
test_all_score_kinds_share_identical_window_and_min_resident_rules
```

### GREEN

Implement `age <= W`, finite validity, and `resident_count >= M_min` in one common function used by B/C/D. Ineligible transitions remain resident and baseline-uniform. No arm-specific exceptions.

**STOP:** expired/invalid/under-resident sources receive amplification, or B/C/D resolve different `W`/`M_min`.

**Commit allowed:** yes. Suggested message: `feat: add source freshness and eligibility policy`.

## Phase 6 — Weighted source sampler (Design Gate 6)

**Modify:** `source_replay.py`, `test_source_replay.py`.

**Dependencies:** Phase 5.

### RED analytical tests

```text
test_source_probabilities_sum_to_one_and_are_finite
test_transition_probabilities_sum_to_one_and_are_finite
test_sampler_is_deterministic_for_fixed_key
test_uniform_floor_is_preserved_per_transition
test_ineligible_sources_are_not_amplified
test_tiny_sources_are_not_amplified_below_m_min
test_sampling_is_uniform_within_source
test_empty_eligible_set_is_uniform_over_ea_transitions
test_empirical_sampling_matches_analytical_transition_probability
test_sampler_reports_required_audit_metrics
```

Use deterministic synthetic populations including unequal `n_i`, tied ranks, one eligible source, all ineligible, and extreme finite logits. Empirical tolerance and draw count are fixed in the test before implementation.

### GREEN math

Implement exactly:

```text
q_i = n_i / N_EA
Q_E = sum(i in E) q_i
s_i = q_i                              for i not in E
s_i = Q_E * softmax_E(r_i / tau)       for i in E
P_i = (1-alpha) * q_i + alpha * s_i
P(x_ij|EA) = (1-alpha)/N_EA + alpha*s_i/n_i
```

Sample uniformly within the selected source. Emit ESS/effective-source count, maximum source probability, resident count, unique-transition ratio, repeat-sampling rate, and uniform-floor mass with documented denominators.

**STOP:** non-finite or unnormalized probability, missing uniform floor, transition amplification of an ineligible source, analytical/empirical mismatch, or capacity-proportional temporary exceeding memory budget.

**Commit allowed:** yes only after Gate 6. Suggested message: `feat: implement bounded source replay sampler`.

## Phase 7 — Real overlay replacement

**Modify:** `source_replay.py`, `erl_utility_replay.py`, related tests.

**Dependencies:** Gate 6.

### RED tests

```text
test_real_overlay_changes_only_ea_slots
test_real_overlay_preserves_rl_indices_bitwise
test_real_overlay_preserves_ea_slot_positions_and_count
test_real_overlay_preserves_batch_size
test_real_overlay_preserves_learner_update_count
test_real_overlay_gathers_payload_from_final_indices
```

### GREEN

Wire weighted conditional EA replacements after the baseline full-batch draw and before unchanged learner gather/update. B/C/D differ only by selected score column.

Rerun all Gates 1–6 and the Phase 3 exact-equivalence workflow test.

**STOP:** replacement outside EA slots, any second baseline draw, changed batch/update count, or `alpha=0` regression.

**Commit allowed:** yes. Suggested message: `feat: connect weighted ea replay overlay`.

## Phase 8 — Score lifecycle integration

**Modify:** `erl_utility_replay.py`, config, workflow tests.

**Dependencies:** Phase 7 and frozen diagnostic tests green.

### RED tests

```text
test_scores_use_pre_mixing_population_trajectories
test_reference_batch_is_sampled_pre_exposure
test_checkpoint_scores_share_one_reference_fingerprint
test_score_table_version_precedes_matching_replay_insertion
test_new_score_affects_only_later_learner_sampling
test_diagnostic_hook_is_training_state_invariant
test_utility_path_calls_frozen_diagnostic_primitive
```

### GREEN lifecycle

Wire:

```text
raw pre-mixing trajectory
-> pre-exposure B_ref and read-only fitness/TD/utility records
-> within-checkpoint rank normalization and new version
-> provenance-aligned replay insertion
-> availability to subsequent learner draws only
```

Reuse the diagnostic-v1 provenance, real TD3 loss, optimizer-faithful virtual update, and diagnostic-local key derivation.

**STOP:** same-operation feedback, score generated from mixed replay, diagnostic mutation, or recomputed utility semantics.

**Commit allowed:** yes after lifecycle and state-invariance tests. Suggested message: `feat: integrate pre-exposure replay scores`.

## Phase 9 — Anti-circularity (Design Gate 7)

**Modify:** workflow interface/tests; no runtime B_eval cost unless the held-out loss endpoint is enabled.

**Dependencies:** Phase 8.

### RED tests

```text
test_reference_and_eval_batches_use_independent_keys
test_reference_and_eval_fingerprints_are_distinct
test_eval_batch_never_enters_score_computation
test_eval_batch_never_changes_sampling_probabilities
test_disabled_eval_endpoint_has_zero_sampling_overhead
```

### GREEN

Define an optional `B_eval` interface with a separately namespaced key and audit fingerprint. When disabled, no batch is sampled and no device work is added. Primary outcome remains deterministic evaluation-return AUC; held-out loss is secondary only.

**STOP:** shared keys/fingerprints, B_eval dataflow into scoring, or loss on B_ref presented as outcome.

**Commit allowed:** yes after Gate 7. Suggested message: `test: enforce replay anti-circularity`.

## Phase 10 — Budget equivalence (Design Gate 8)

**Modify:** workflow/config tests and common audit counters only.

**Dependencies:** Gate 7.

### RED tests

```text
test_abcd_arms_have_equal_environment_interactions
test_abcd_arms_have_equal_learner_update_counts
test_abcd_arms_have_equal_batch_size
test_abcd_arms_have_equal_population_operations
test_abcd_arms_have_equal_evaluation_checkpoints
test_abcd_arms_have_equal_evaluation_episode_counts
test_abcd_arms_have_equal_total_training_budget
test_all_arms_emit_common_audit_schema
```

### GREEN

Create one shared arm configuration schema. A resolves to `alpha=0`; B/C/D share exactly the same `alpha`, `tau`, `W`, and `M_min`, differing only in `score_kind`. Validate counters programmatically at run finalization.

**STOP:** any budget/counter mismatch or arm-specific freshness/eligibility setting.

**Commit allowed:** yes after Gate 8 and cumulative regression. Suggested message: `test: enforce matched utility replay budgets`.

## Phase 11 — Return-blind calibration tooling

**Modify:** create calibration script/test; extend config schema.

**Dependencies:** Gates 1–8.

### RED tests

```text
test_calibration_rejects_return_auc_and_final_return_columns
test_correctness_filter_rejects_invalid_candidates
test_operational_band_filter_is_deterministic
test_ess_target_selects_minimum_registered_distance
test_tie_break_uses_max_mass_repeat_rate_overhead_then_lexicographic_order
test_resolved_config_contains_candidate_hashes_thresholds_trace_and_checksums
test_resolved_config_freezes_common_parameters_for_bcd
test_resolved_config_is_reproducible_from_same_inputs
```

### GREEN

Implement finite ordered candidate products `A x T x W_set x M_set`. The input schema permits only sampler/system fields: ESS, maximum source mass, uniform-floor contribution, fresh/expired coverage, repeat rate, numerical status, memory, and compute overhead. Reject forbidden outcome fields rather than silently ignoring them.

The calibration corpus must be isolated from the formal matched pilot and come from exactly one pre-registered source:

- **Option A:** frozen diagnostic-v1 artifacts, used only for score stability, rank persistence, freshness statistics, and sampler simulation; future return and intervention performance are forbidden; or
- **Option B:** a dedicated calibration-only run/seed that is not one of the formal matched three seeds. Its seed, commit, resolved config, artifact path, and SHA-256 manifest must be recorded before selection.

Formal pilot return, AUC, and final return can never flow back into selection of `alpha`, `tau`, `W`, `M_min`, ESS target, or operational bands. Once any formal outcome is visible, changing any of those values creates a new pilot version and requires rerunning every A/B/C/D arm.

Freeze output at:

```text
/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/calibration/<version>/
  candidates.json
  calibration_metrics.csv
  selection_trace.json
  resolved_config.yaml
  checksums.sha256
```

**STOP:** no candidate survives, a forbidden field is read, non-deterministic selection occurs, or resolved config lacks provenance.

**Commit allowed:** yes for code/tests only; calibration artifacts remain outside Git. Suggested message: `feat: add return-blind replay calibration`.

## Phase 12 — A/B/C/D one-environment, one-seed smoke

**Code changes:** none expected after launch. Fixes require returning to the failing phase and rerunning all dependent gates.

**Dependencies:** all Gates 1–8, regression, protected-file diff, and frozen resolved config.

### Preflight tests

```text
test_smoke_manifest_uses_one_environment_one_seed_four_arms
test_smoke_manifest_references_frozen_resolved_config
test_smoke_output_validator_checks_all_gate_counters
```

Verify in tmux and conda:

```bash
nvidia-smi
env -u LD_LIBRARY_PATH /home/admin001/anaconda3/envs/evorl-conflict/bin/python \
  -c "import jax; print(jax.devices()); print(jax.default_backend())"
```

Run A, B, C, D sequentially, never concurrently, on Swimmer and one registered seed. This smoke validates completion, budget equality, finite probabilities, non-degenerate weighting, freshness fallback, RL-slot invariance, and audit output only. It cannot determine algorithm superiority.

Artifacts:

```text
/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/smoke/
  swimmer/seed_<seed>/<A|B|C|D>/
    manifest.json
    resolved_config.yaml
    progress.json
    logs/run.log
    audit/source_sampling.csv
    audit/budget_counters.json
    audit/reference_fingerprints.json
    metrics/evaluation.csv
    checksums.sha256
  aggregate/smoke_validation.json
```

**GREEN:** four runs complete and validator confirms all operational gates. Do not inspect smoke return to tune parameters.

**STOP:** crash, NaN, degenerate weighting, key/fingerprint violation, RL-slot change, budget mismatch, checksum failure, or protected-file change.

**Commit:** no experiment outputs. A code fix may be committed only after returning to RED→GREEN tests and rerunning all affected gates. No push.

## Phase 13 — Matched three-seed pilot (description only; not authorized to run)

**Dependencies before future authorization:** Phase 12 reviewed and green, frozen config unchanged, clean working tree, checksums verified, and explicit user approval.

Planned matrix:

```text
environment: Swimmer
arms: A Original ERL, B Fitness, C high-TD, D Learner Utility
seeds: three pre-registered matched seeds
execution: sequential on the fixed GPU environment
budget/evaluation schedule: identical and frozen before launch
```

Primary endpoint:

```text
AUC[m,s] = sum_k 0.5 * (R[m,s,k] + R[m,s,k+1]) * (b[k+1] - b[k])
```

using deterministic evaluation return at identical interaction checkpoints. Final return is secondary. Missing scheduled evaluations invalidate the matched pair unless one pre-registered policy is applied uniformly.

Artifact layout:

```text
/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/matched_3seed/
  swimmer/seed_<seed>/<A|B|C|D>/...
  aggregate/per_seed_auc.csv
  aggregate/matched_differences.csv
  aggregate/go_no_go.json
  aggregate/report.md
  aggregate/checksums.sha256
```

GO requires all correctness gates plus:

```text
median(AUC_D - AUC_A) > 0
at least 2/3 seeds: AUC_D > AUC_A
and the same median-positive plus 2/3 rule versus B OR versus C
```

If all 3 seeds have `FinalReturn_D < FinalReturn_A`, result is automatic NO-GO. No thresholds, score directions, candidate sets, endpoints, or missing-run policy may change after formal outcomes are visible.

**Commit/run authorization:** none in this plan. Stop after Phase 12 and request review before Phase 13.

---

## Cumulative verification commands

Focused tests grow by phase; before each allowed commit run all currently existing pilot tests:

```bash
env -u LD_LIBRARY_PATH /home/admin001/anaconda3/envs/evorl-conflict/bin/python \
  -m pytest -p no:cacheprovider -q \
  tests/test_source_aware_replay_buffer.py \
  tests/test_source_replay.py \
  tests/test_erl_utility_replay_workflow.py \
  tests/test_utility_replay_calibration.py
```

Also rerun frozen learner-utility tests and repository regression. If the already documented optional MJX import/environment issue remains, exclude only `tests/test_mjx_env.py`, state the exclusion exactly, and do not count it as a pilot failure. Any new failure is a STOP.

Before every commit:

```bash
git diff --check
git status --short
git diff --stat
git diff -- <protected file list>
```

Review the full diff; never use `git reset --hard` or `git clean -fd`.

## Commit permissions and branch safety

- Phase 0: no commit.
- Phases 1–11: one local commit per green, reviewable phase is allowed; do not bundle a failed or partially green phase.
- Phase 12: artifacts are not committed. Only separately tested fixes may be committed.
- Phase 13: no execution or commit without new approval.
- No push is authorized.
- Tag `learner-utility-diagnostic-v1` and branch `research/learner-utility-diagnostic` remain untouched.
- Before work, record `git rev-parse HEAD`, branch, status, and protected-file checksums. After each phase, repeat them.
- Rollback means revert only the pilot phase's own reviewed commit or restore a new pilot file deliberately; never reset the repository or overwrite user changes.

## Failures that immediately block later phases

1. Wrong branch, wrong base commit, dirty unexplained working tree, or changed protected file.
2. Metadata/payload misalignment, incorrect mask/overwrite behavior, token collision/overflow, or excess sidecar memory.
3. Any RL index or realized EA/RL slot change.
4. Any `alpha=0` difference in indices, batch, baseline key, training state, metrics, or counters.
5. Invalid/non-finite probabilities, lost uniform floor, tiny-source amplification, or analytical/empirical mismatch.
6. Score computed after exposure, same-operation feedback, or mutation by the read-only diagnostic.
7. B_ref/B_eval leakage or outcome data entering calibration.
8. Any budget mismatch across A/B/C/D.
9. Calibration with no valid survivor or a non-reproducible resolved config.
10. Smoke crash, NaN, degeneracy, checksum failure, or audit-schema incompleteness.
11. Any request to tune after viewing formal outcomes; that requires a new design version and complete rerun.

## Completion boundary for this plan

Implementation planning is complete when this document is reviewed and approved. No source/config/test modification, intervention experiment, or remote push is part of the current task. After approval, execution starts at Phase 0 and may advance only through the gates above.
