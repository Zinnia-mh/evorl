# Learner-Utility Source-Control Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an observer-only diagnostic that compares EA, baseline RL-own, frozen replay-random, and reference-self source utility without changing baseline ERL/TD3 training or adding environment steps.

**Architecture:** A new `ERLUtilitySourceControlsWorkflow` subclasses the sealed `ERLLearnerUtilityWorkflow`. Its pre-exposure context freezes immutable learner references and pre-samples all replay/reference controls before EA exposure; its EA and RL rollout hooks retain the baseline-generated transitions and run an eager, deferred diagnostic before the real learner update. Pure core helpers perform independent scaled virtual updates and aggregation, while a separate artifact component enforces the exclusive output root and atomically writes host-side records and summaries.

**Tech Stack:** Python 3.10+, JAX/JAXLIB, Flax, Optax, Chex, Hydra/OmegaConf, NumPy/Pandas/SciPy, pytest, Brax, Git worktrees, tmux.

**Spec:** `docs/superpowers/specs/2026-10-04-learner-utility-source-controls-design.md`

## Global Constraints

- Work only in `/disk1/zmh_projects/evorl_utility_source_controls` on `research/utility-source-controls`; never modify `/disk1/zmh_projects/evorl` or replay-pilot.
- Base lineage is `research/learner-utility-diagnostic` at `df954935334e201f3e2759b848d5be2c177833f7`; design commit is `a812a7031026f8e9a80d1ac4e492c38d25549145`.
- Use the existing `evorl` Conda environment; do not change Python, JAX, JAXLIB, Flax, Optax, CUDA, or dependencies.
- All runtime writes, logs, checkpoints, temporary files, and caches must resolve under `/disk1/zmh_projects/evorl_utility_source_controls_outputs`.
- `ERLUtilitySourceControlsWorkflow` must subclass `ERLLearnerUtilityWorkflow`; do not alter the sealed workflow's behavior.
- RL-own must reuse the baseline RL rollout. Additional diagnostic rollout and environment steps are forbidden.
- Four sources are fixed: EA, RL-own, Replay-random, Reference-self. References=4, replay controls=8, step scales=`1.0,0.5,0.1`; zero-step is a hard gate.
- Utility remains `L_before - L_after` using the existing TD3 critic-loss adapter and optimizer-faithful virtual update semantics.
- No utility-aware replay, prioritization, filtering, actor/critic training changes, replay-ratio changes, or Case A-D feedback into training.
- Follow RED -> verify failure -> minimal GREEN -> full relevant regression for every production change.

## Review Focus

1. **Replay not ready at a due checkpoint:** emit explicit ineligible evidence without sampling or corrupting later checkpoints; pinned by Task 5 replay-readiness tests.
2. **Nested/symlink path escape:** reject before any collector call; pinned by Task 2 canonicalization and rollout-order tests.
3. **Multi-agent/multi-env RL trajectory axes:** flatten valid transitions without crossing source semantics or padding; pinned by Task 6 parameterized shape tests.
4. **Non-finite utility/loss values:** retain raw evidence, mark checkpoint invalid, and exclude from formal summaries; pinned by Tasks 4 and 8.
5. **Interrupted artifact finalization:** keep append-only raw evidence and never replace a valid summary with a partial file; pinned by Task 8 atomic-write failure tests.

---

## 1. Current code map

| Path / symbol | Current role | Modify? | Planned use/change |
|---|---|---:|---|
| `evorl/algorithms/erl/erl_td3/erl_learner_utility.py:129` `ERLLearnerUtilityWorkflow` | Sealed diagnostic subclass and context scoping | No | Parent class only; inherit cadence and baseline step delegation. |
| `erl_learner_utility.py:157` `step()` | Creates diagnostic context before calling baseline `ERLWorkflow.step()` | No | Dynamic dispatch into the new `_make_pre_exposure_context()` gives the exact pre-EA hook. |
| `erl_learner_utility.py:185` `_make_pre_exposure_context()` | Selects learner 0 and critic optimizer state | No | Call via `super()`, enrich in new subclass with snapshots, keys, controls, counters. |
| `erl_learner_utility.py:203` `_sample_current_reference()` | Samples one read-only replay reference | No | Reuse fingerprint logic; new module samples 4 references and 8 controls at context creation. |
| `erl_learner_utility.py:339` `_ec_rollout()` | Captures raw EA population before replay insertion | No | New subclass overrides this hook to retain EA sources without running the old immediate diagnostic. |
| `evorl/algorithms/erl/erl_td3/erl_origin.py:115` `ERLWorkflow.step()` | True baseline iteration | No | Verified order: key split -> EA rollout/add -> RL rollout/add -> learner update -> EA update/injection. |
| `erl_origin.py:128-130` | Baseline PRNG split | No | Diagnostic derives folded keys from `state.key`; never consumes returned baseline keys. |
| `erl_origin.py:137` `_ec_rollout()` call | EA rollout and replay insertion | No | New hook preserves return values and insertion order exactly. |
| `erl_origin.py:152` `_rl_rollout()` call | RL rollout and replay insertion | No | New hook retains the real RL transitions and triggers deferred diagnostic before line 174 learner update. |
| `erl_origin.py:174` `_rl_update()` | Real critic/actor update | No | Deferred diagnostic must finish before this call and use only frozen state. |
| `erl_origin.py:200-205` | EA update/injection | No | Baseline output equivalence tests cover population and fitness. |
| `evorl/algorithms/erl/erl_td3/erl_td3_workflow.py:257` `rollout_episode()` | Collector call, trajectory flatten, replay add | No | Reproduce only inside new hooks where raw trajectories are required; preserve lines 268-288 exactly around capture. |
| `evorl/algorithms/td3.py:47` `TD3NetworkParams` | Actor, critic, target actor, target critic PyTree | No | Frozen learner state already includes critic and target critic. |
| `td3.py:137` `TD3Agent.critic_loss()` | Authoritative TD3 critic loss | No | Continue via `td3_critic_loss`; same frozen target params and noise configuration. |
| `evorl/replay_buffers/replay_buffer.py:75` `ReplayBuffer.sample()` | Functional random replay sample | No | Sample frozen reference/control batches at pre-exposure context creation with derived keys. |
| `evorl/algorithms/erl/erl_td3/learner_utility.py:37` `build_source_batches()` | EA provenance and matched batches | No | Reuse for EA; add separate RL-own flattening helper in new module. |
| `learner_utility.py:97` `derive_diagnostic_key()` | Folded diagnostic key without baseline advancement | No | Reuse with a new source-control tag and stable sub-tags. |
| `learner_utility.py:105` `virtual_critic_utility()` | Existing scale-1 functional Optax update | No | Preserve sealed behavior; new scaled helper uses the same optimizer path and scales final temporary updates. |
| `configs/agent/erl/erl-learner-utility.yaml` | Sealed diagnostic Hydra config | No | Copy relevant baseline fields into a new source-control config, leaving defaults unchanged. |
| `scripts/train.py:59` `train()` | Hydra entry, workflow build/init/learn | No | Smoke invokes this entry; output guard runs during new workflow build before collector execution. |
| `tests/test_learner_utility.py` | Pure sealed diagnostic tests | No | Run unchanged as regression. |
| `tests/test_erl_learner_utility_workflow.py` | Sealed workflow lifecycle/equivalence tests | No | Run unchanged as regression. |
| Existing artifact writer | None in sealed Git branch; prior formal artifacts were external orchestration | N/A | Add a dedicated writer rather than scattering I/O in workflow code. |
| `configs/env/brax/swimmer.yaml` | Verified Swimmer environment | No | Use in smoke with new agent config and explicit overrides. |

## 2. File structure

**Create**

- `evorl/algorithms/erl/erl_td3/utility_source_controls.py` — frozen context/record dataclasses, RL-own batching, scaled virtual update, tolerance/sign/reference aggregation, fingerprints.
- `evorl/algorithms/erl/erl_td3/erl_utility_source_controls.py` — new workflow subclass and exact EA/RL/pre-exposure hooks.
- `evorl/algorithms/erl/erl_td3/utility_source_control_artifacts.py` — output guard, schema validation, atomic writer, summaries and Case A-D report.
- `configs/agent/erl/erl-utility-source-controls.yaml` — disabled-safe source-control configuration.
- `tests/test_utility_source_controls_core.py` — pure source/scale/reference/snapshot tests.
- `tests/test_utility_source_controls_workflow.py` — hook ordering, RL-own reuse, invariance, no-extra-rollout, JIT boundaries.
- `tests/test_utility_source_controls_artifacts.py` — schemas, summaries, report language, atomic writes.
- `tests/test_utility_source_controls_output_guard.py` — path/symlink/early-failure tests.
- `docs/learner_utility_source_controls/README.md` — method, schema, smoke/reproduction instructions; no smoke results until execution.

**Modify only if required by a failing integration test**

- `evorl/algorithms/erl/erl_td3/__init__.py` — export the new class only if project import conventions require it; Hydra can otherwise load the full module path without modification.

**Explicitly do not modify**

- `erl_learner_utility.py`, `learner_utility.py`, `erl_origin.py`, `erl_td3_workflow.py`, `td3.py`, replay buffer implementations, replay-pilot files, and existing learner-utility tests.

## 3. Frozen snapshot strategy

### JAX state semantics

JAX arrays and EvoRL `PyTreeData` values are immutable; baseline updates return new trees rather than mutating buffers in place. Therefore the runtime snapshot stores references to the current immutable leaves, wrapped with `jax.lax.stop_gradient`/`tree_stop_gradient` where appropriate. Do not `device_get` or deep-copy full parameter/optimizer trees merely to freeze them. Unit tests keep the original trees and assert exact tree equality after all diagnostic paths.

### `FrozenDiagnosticSnapshot`

Create a frozen host dataclass with:

```python
@dataclass(frozen=True)
class FrozenDiagnosticSnapshot:
    generation: int
    learner_agent_state: AgentState
    critic_opt_state: chex.ArrayTree
    baseline_key: chex.PRNGKey
    baseline_key_fingerprint: str
    critic_fingerprint: str
    target_critic_fingerprint: str
    optimizer_fingerprint: str
    replay_metadata_fingerprint: str
    counter_fingerprint: str
    reference_batches: tuple[SampleBatch, ...]
    reference_fingerprints: tuple[str, ...]
    replay_control_batches: tuple[SampleBatch, ...]
```

- **Critic/target critic:** select learner 0 as the sealed diagnostic does; keep immutable leaves in `learner_agent_state`, plus stable fingerprints for temporal alignment and audit.
- **Optimizer:** select learner 0 from `state.opt_state.critic`; keep immutable leaves and discard every `optimizer.update()` returned state.
- **Replay:** never copy the capacity-sized buffer. At `_make_pre_exposure_context(state)`, sample and retain four reference batches and eight replay-control batches before EA rollout; retain only replay metadata/fingerprint for audit.
- **PRNG:** retain `state.key` and hash its bytes; derive all diagnostic keys with `derive_diagnostic_key()` and `fold_in` tags. Do not return or assign a diagnostic key to baseline state.
- **Counters:** capture generation, sampled episodes/timesteps, RL timesteps and iteration values as host scalars/fingerprint.

`tree_fingerprint(tree) -> str` hashes leaf shape, dtype, and bytes at diagnostic checkpoints only. Runtime fingerprints cover frozen learner/counters and small sampled batches; no full replay-data hash is performed. Full tree equality is reserved for focused tests.

## 4. Exact workflow hooks and data flow

```text
ERLLearnerUtilityWorkflow.step(state)       # existing line 157
  -> new _make_pre_exposure_context(state) # BEFORE parent baseline step
       freeze learner/counters/key
       sample reference x4 + replay controls x8 from state.replay_buffer_state
  -> ERLWorkflow.step(state)                # existing line 115
       -> new _ec_rollout(...)
            baseline collector exactly once
            retain raw EA per-source batches and fitness
            baseline replay insertion exactly once
       -> new _rl_rollout(...)
            baseline RL collector exactly once
            retain matched RL-own batch
            baseline replay insertion exactly once
            compare current critic fingerprint to frozen fingerprint
            run deferred diagnostic from frozen snapshot
            persist host records
       -> existing _rl_update(...)          # only after diagnostic
       -> existing EA update/injection
```

“Deferred” changes only computation time. Every loss and temporary update uses `FrozenDiagnosticSnapshot.learner_agent_state` and `.critic_opt_state`, never the current mutable local variables from later training.

### Temporal alignment

Before deferred computation, fingerprint the learner critic passed into the real RL rollout path and compare with the frozen critic fingerprint:

- equal and no learner-update marker observed -> `exact`;
- different or explicit test hook reports an intervening update -> `delayed`.

Never hard-code `exact`; tests inject a changed current learner to verify `delayed`.

## 5. Source preparation

### EA

The new `_ec_rollout()` uses the sealed raw-population pattern and `build_source_batches()` before flatten/replay add. It stores `SourceBatch` records and EA fitness in the context; it does not invoke the sealed immediate diagnostic.

### RL-own

The new `_rl_rollout()` performs the same single `jax.vmap(self.rl_collector.rollout)` as baseline. Before `dones` are removed, `build_rl_own_source_batch(raw_trajectory, generation, source_batch_size, key)`:

- accepts `[num_rl_agents, time, env, ...]` raw leaves;
- applies the same shifted-done validity mask per agent;
- flattens agent/time/env axes without padding or cross-source substitution;
- deterministically samples exactly `source_batch_size` with a derived diagnostic key;
- returns one `SourceControlBatch(source_type='RL-own', source_id='gen_<g>_rl_own', ...)`;
- returns ineligible reason `insufficient_rl_own_samples` rather than rolling out again.

Then the hook reproduces baseline flattening and replay insertion once. A collector counter test proves diagnostic on/off use identical call counts and environment-step metrics.

### Replay-random

At context creation, split one derived replay-control key into eight configurable keys and call `replay_buffer.sample(state.replay_buffer_state, key)` before EA collector execution. Retain these fixed batches. Tests seed old replay with one marker and current EA trajectory with another, proving controls contain only the old marker.

### Reference-self

Use a diagonal definition: for frozen reference `R_k`, source ID `reference_self_<k>` uses `B_source=R_k` and evaluates on the same `R_k`. Do not cross-evaluate `R_k` on unrelated `R_j`, because that would cease to be the specified self control. The four diagonal records still feed source-level summaries.

## 6. Virtual update and zero-step

Add:

```python
def scaled_virtual_critic_utility(
    *, loss_fn: LossFn, optimizer: optax.GradientTransformation,
    params: chex.ArrayTree, opt_state: chex.ArrayTree,
    source_batch: Any, reference_batch: Any,
    key: chex.PRNGKey, step_scale: float,
) -> UtilityResult
```

The function computes gradients and calls the real `optimizer.update(grads, opt_state, params)` exactly once. It discards the returned optimizer state, multiplies each final update leaf by `step_scale`, and applies it to temporary params. This defines scale as **virtual optimizer update magnitude**, not gradient scale; Adam moments, clipping and transform behavior are therefore computed exactly as at scale 1.0, while the real optimizer/LR/state remain untouched.

Zero-step calls this same function with `step_scale=0.0`. It uses the same source preparation, loss function, reference key, optimizer transform and `apply_updates`; only the final temporary update is zero. It must not bypass the path by returning params directly.

Use the same reference noise key for `L_before` and `L_after`. Every `(source, scale, reference)` gets a deterministic folded key so evaluation order cannot change values.

`resolve_numerical_tolerance(dtype, before, after, configured)` returns the configured positive tolerance when supplied; otherwise it uses `32 * finfo(dtype).eps * max(1, abs(before), abs(after))`. Record the rule and resolved per-record tolerance. Near-zero uses its separately configured override or the same dtype-aware rule.

## 7. Host-side record representation and aggregation

Use frozen Python dataclasses outside JIT:

- `SourceControlBatch` — source type/ID, optional EA fitness, batch, eligibility/reason, valid count.
- `SourceUtilityRecord` — exact schema from the spec plus `numerical_tolerance` and optional `evolutionary_fitness`.
- `ReferenceStabilityRecord` — checkpoint/source/scale aggregate.
- `CheckpointDiagnostic` — status, reasons/warnings, temporal alignment, raw records, zero-step results.

JAX functions return array-valued loss/utility/TD statistics. Convert with `jax.device_get` only at the eager deferred-diagnostic boundary before constructing records. No strings, dynamic Python lists, CSV/JSON operations, or host filesystem calls appear inside jitted kernels.

`aggregate_reference_utilities(values, tolerance)` excludes near-zero values from sign voting, reports `near_zero_fraction`, and returns `sign_consistency=None, all_near_zero=True` when every value is near zero.

Non-finite before/after/utility values mark the checkpoint invalid, remain in raw records, and are excluded from formal summaries.

## 8. Output guard and artifact writer

### Guard

`validate_artifact_root(candidate: str | Path, allowed_root: Path, forbidden_roots: Sequence[Path]) -> Path`:

- requires an absolute candidate;
- calls `resolve(strict=False)` on candidate and allowed/forbidden roots;
- uses `Path.is_relative_to()` (Python >=3.10) after symlink resolution;
- accepts only allowed root or descendants;
- explicitly rejects canonical repo, old diagnostic artifacts, replay-pilot paths, `..` escapes and symlinks outside;
- runs in `_build_from_config()` before `super()._build_from_config(config)` can create collectors/workflow state.

### Writer

`UtilitySourceControlArtifactWriter` owns all host I/O:

- constructor validates the root, creates only the per-run directory, and writes an atomic `run_config.json`;
- `append_checkpoint(checkpoint)` appends schema-validated raw rows to `per_source_utility.csv` and retains checkpoint status evidence;
- `finalize_checkpoint(checkpoint)` atomically rewrites checkpoint/reference summaries from complete raw records;
- `finalize_run()` atomically writes cross-source and step-scale summaries plus `diagnostic_report.md`;
- each atomic write uses a temporary file inside the same run directory followed by `Path.replace()`;
- Case A-D logic exists only in report-generation functions and has no workflow/replay dependencies.

EA fitness/rank metrics use each EA source's cross-reference `utility_median` at `step_scale=1.0`. This yields one auditable utility value per EA source/checkpoint and avoids treating four correlated reference evaluations as independent sources.

Required columns are constants tested against exact CSV headers. An interrupted write test forces an exception before replace and verifies the prior valid summary remains intact.

## 9. Runtime and test invariance strategy

### Runtime checks at due checkpoints

- fingerprint frozen critic/target critic/optimizer/key/counters before diagnostic;
- fingerprint the same frozen trees after diagnostic (they must match exactly);
- fingerprint replay metadata immediately before and after the deferred diagnostic (baseline EA/RL insertion has already occurred; no insertion is allowed during this interval);
- record collector call counts/metrics indirectly through unchanged baseline counters, not instrumentation that changes collectors;
- mark invalid on mismatch and retain evidence.

### Focused tests

Use `chex.assert_trees_all_equal` on actor, critic, target critic, optimizer, keys, replay, environment fixtures, EC state/population, fitness, and metrics. Run paired diagnostic-disabled/enabled steps from identical fixtures and assert identical baseline outputs except diagnostic-owned host artifacts.

## 10. Configuration

Create `configs/agent/erl/erl-utility-source-controls.yaml` from the frozen diagnostic configuration with:

```yaml
workflow_cls: evorl.algorithms.erl.erl_td3.erl_utility_source_controls.ERLUtilitySourceControlsWorkflow
diagnostic_enabled: false
diagnostic_interval: 10
artifact_root: null
source_batch_size: 256
num_reference_batches: 4
num_random_control_batches: 8
step_scales: [1.0, 0.5, 0.1]
near_zero_tolerance: null
zero_step_tolerance: null
reference_self_systemic_negative_threshold: 0.5
ea_mismatch_negative_rate_margin: 0.10
step_sensitivity_negative_rate_drop: 0.20
case_b_negative_rate_min: 0.80
case_b_negative_rate_max: 0.90
```

Disabled remains baseline-equivalent. Enabled requires an explicit absolute `artifact_root`. Do not add defaults that point into the repository. Case A requires valid gates and both RL-own and Replay-random negative rates to trail EA by at least `0.10`; Case B uses the frozen `0.80..0.90` control range; Case D requires a monotonic scale response and at least `0.20` reduction from scale `1.0` to `0.1`. Record these thresholds in `run_config.json` so report classification is reproducible rather than subjective.

---

## 11. Implementation tasks (strict RED -> GREEN)

### Task 1: Core records, tolerances, aggregation, and scaled virtual update

**Files:**
- Create: `evorl/algorithms/erl/erl_td3/utility_source_controls.py`
- Create: `tests/test_utility_source_controls_core.py`

**Interfaces:**
- Produces `SourceControlBatch`, `FrozenDiagnosticSnapshot`, `SourceUtilityRecord`, `CheckpointDiagnostic`, `tree_fingerprint`, `resolve_numerical_tolerance`, `scaled_virtual_critic_utility`, `aggregate_reference_utilities`.

- [ ] Write failing tests: `test_step_scale_multiplies_final_optax_update_not_gradient`, `test_zero_step_reuses_optimizer_path_and_returns_zero_utility`, `test_all_sources_start_from_identical_params_and_optimizer_state`, `test_sign_consistency_ignores_near_zero_votes`, `test_all_near_zero_returns_na_consistency`, `test_non_finite_utility_invalidates_checkpoint`.
- [ ] Run `conda run -n evorl python -m pytest -p no:cacheprovider tests/test_utility_source_controls_core.py -q`; expect import/behavior failures for missing module/functions.
- [ ] Implement the declared dataclasses/helpers and minimal scaled virtual path.
- [ ] Re-run the focused file; expect all tests PASS.
- [ ] Commit: `git commit -m "diag: add source-control utility primitives"`.

### Task 2: Output-root guard before rollout

**Files:**
- Create: `evorl/algorithms/erl/erl_td3/utility_source_control_artifacts.py`
- Create: `tests/test_utility_source_controls_output_guard.py`

**Interfaces:**
- Produces `validate_artifact_root()` and `UtilitySourceControlArtifactWriter` constructor contract for later tasks.

- [ ] Write failing tests for allowed root/child and rejection of relative path, canonical repo, legacy artifact root, replay-pilot output, `..` escape, and a symlink outside.
- [ ] Add `test_invalid_artifact_root_fails_before_collector_or_environment_call` using a counter fixture.
- [ ] Run the test file; expect missing guard failures.
- [ ] Implement only canonicalization, forbidden-root validation, and constructor preflight.
- [ ] Re-run; expect PASS and zero collector calls on every invalid path.
- [ ] Commit: `git commit -m "diag: guard source-control output root"`.

### Task 3: Frozen checkpoint snapshot and diagnostic key partitioning

**Files:**
- Modify: `evorl/algorithms/erl/erl_td3/utility_source_controls.py`
- Create: `evorl/algorithms/erl/erl_td3/erl_utility_source_controls.py`
- Create: `tests/test_utility_source_controls_workflow.py`

**Interfaces:**
- Produces `ERLUtilitySourceControlsWorkflow._make_pre_exposure_context(state)` and stable source/reference/evaluation key tags.

- [ ] Write failing tests `test_snapshot_selects_learner_zero_critic_target_and_optimizer`, `test_snapshot_keeps_immutable_state_after_baseline_replacement`, `test_diagnostic_keys_do_not_advance_baseline_key`, `test_snapshot_records_generation_counters_and_fingerprints`.
- [ ] Run core+workflow files; expect missing class/context failures.
- [ ] Implement subclass `_build_from_config()` preflight and `_make_pre_exposure_context()` around `super()` without full replay copying.
- [ ] Re-run; expect PASS.
- [ ] Commit: `git commit -m "diag: freeze source-control checkpoint state"`.

### Task 4: Pre-exposure reference and replay-control capture

**Files:**
- Modify: `erl_utility_source_controls.py`
- Modify: `tests/test_utility_source_controls_workflow.py`

**Interfaces:**
- Extends context with four reference batches and eight configurable replay controls sampled from the incoming replay state.

- [ ] Write failing tests `test_controls_sample_before_ea_collector_and_replay_add`, `test_controls_exclude_current_ea_marker`, `test_reference_and_replay_counts_are_configurable`, `test_replay_not_ready_emits_ineligible_checkpoint_without_sampling`, `test_sampling_leaves_replay_state_treewise_equal`.
- [ ] Run targeted tests; expect ordering/count failures.
- [ ] Implement deterministic key partition and pre-sampling in `_make_pre_exposure_context()`; never retain/copy full replay data.
- [ ] Re-run; expect PASS.
- [ ] Commit: `git commit -m "diag: capture frozen replay controls pre exposure"`.

### Task 5: EA source staging without immediate diagnostic

**Files:**
- Modify: `erl_utility_source_controls.py`
- Modify: `tests/test_utility_source_controls_workflow.py`

**Interfaces:**
- Produces context fields `ea_sources`, `fitnesses`, and exact baseline replay output while deferring measurement.

- [ ] Write failing tests `test_ea_hook_retains_pre_mixing_provenance`, `test_ea_hook_adds_replay_once`, `test_ea_hook_does_not_emit_records_before_rl_rollout`, `test_disabled_or_non_due_path_delegates_exactly_to_parent`.
- [ ] Run targeted tests; expect missing hook failures.
- [ ] Implement `_ec_rollout()` by preserving the sealed collector/flatten/add sequence and staging sources only.
- [ ] Re-run; expect PASS and exact baseline trajectory/replay outputs.
- [ ] Commit: `git commit -m "diag: stage EA sources for deferred controls"`.

### Task 6: Reuse real RL-own transitions and temporal alignment

**Files:**
- Modify: `utility_source_controls.py`
- Modify: `erl_utility_source_controls.py`
- Modify: `tests/test_utility_source_controls_core.py`
- Modify: `tests/test_utility_source_controls_workflow.py`

**Interfaces:**
- Produces `build_rl_own_source_batch()` and eager `_rl_rollout()` that stages the real rollout, inserts it once, and invokes deferred evaluation before learner update.

- [ ] Write failing parameterized shape tests for one/multiple RL agents and one/multiple env axes; assert exact matched size and `insufficient_rl_own_samples`.
- [ ] Write failing lifecycle tests `test_rl_own_uses_baseline_transition_identity`, `test_diagnostic_adds_no_collector_call_or_environment_steps`, `test_rl_replay_insertion_occurs_once`, `test_deferred_diagnostic_precedes_real_learner_update`, `test_temporal_alignment_exact_when_critic_unchanged`, `test_temporal_alignment_delayed_when_intervening_critic_differs`.
- [ ] Run targeted tests; expect missing RL helper/hook failures.
- [ ] Implement raw RL capture with baseline flatten/add semantics, `stop_gradient` read-only view, and fingerprint-based temporal alignment.
- [ ] Re-run; expect PASS and identical baseline counters.
- [ ] Commit: `git commit -m "diag: reuse baseline RL transitions for source controls"`.

### Task 7: Four-source multi-reference deferred evaluation and hard gates

**Files:**
- Modify: `utility_source_controls.py`
- Modify: `erl_utility_source_controls.py`
- Modify: `tests/test_utility_source_controls_core.py`
- Modify: `tests/test_utility_source_controls_workflow.py`

**Interfaces:**
- Produces `_run_deferred_source_control_diagnostic(context, rl_source, replay_after_rl)` returning `CheckpointDiagnostic`.

- [ ] Write failing tests for EA/RL/replay source × 3 scales × 4 references, diagonal reference-self × 3 scales, zero-step through the same path, source-state independence, exact schema fields, batch mismatch rejection, zero-step invalidation, reference-self warning-only behavior, and deterministic evaluation under source reordering.
- [ ] Run targeted tests; expect missing evaluator failures.
- [ ] Implement eager orchestration and jitted numerical kernels; use only frozen learner/optimizer/controls.
- [ ] Add runtime frozen-tree and replay-before/after-diagnostic fingerprint checks.
- [ ] Re-run; expect PASS.
- [ ] Commit: `git commit -m "diag: evaluate four frozen utility source controls"`.

### Task 8: Artifact writer, summaries, rank metrics, and report

**Files:**
- Modify: `utility_source_control_artifacts.py`
- Create: `tests/test_utility_source_controls_artifacts.py`

**Interfaces:**
- Completes writer `append_checkpoint()`, `finalize_checkpoint()`, `finalize_run()` and pure summary/report functions.

- [ ] Write failing exact-header tests for all CSV/JSON artifacts.
- [ ] Write fixture tests for checkpoint/cross-source/step-scale/reference summaries, EA Spearman, Kendall tau-b, inversion rate, top-F rank, and top-F==top-U.
- [ ] Write Case A-D report tests asserting permitted wording and absence of `proves`.
- [ ] Write invalid-checkpoint exclusion/raw-retention and interrupted-atomic-write tests.
- [ ] Run artifact tests; expect missing writer/summary failures.
- [ ] Implement append-only raw CSV, atomic summary/config/report writes, and run-end finalization.
- [ ] Re-run; expect PASS.
- [ ] Commit: `git commit -m "diag: write source-control evidence and summaries"`.

### Task 9: Hydra config, writer lifecycle, and JIT boundaries

**Files:**
- Create: `configs/agent/erl/erl-utility-source-controls.yaml`
- Modify: `erl_utility_source_controls.py`
- Modify: `tests/test_utility_source_controls_workflow.py`
- Modify: `tests/test_utility_source_controls_artifacts.py`

**Interfaces:**
- Binds config fields, creates/finalizes writer, and keeps host diagnostic hooks eager while baseline numerical kernels remain jitted.

- [ ] Write failing tests for disabled defaults, enabled explicit root, config list/counts, writer close/finalize, eager `_ec_rollout`/`_rl_rollout`, and no JAX tracer/string/shape regression.
- [ ] Run targeted tests; expect missing config/lifecycle failures.
- [ ] Implement config and lifecycle integration without modifying `scripts/train.py` or parent workflows.
- [ ] Re-run; expect PASS.
- [ ] Commit: `git commit -m "diag: integrate source-control workflow configuration"`.

### Task 10: Complete baseline invariance and focused GREEN gate

**Files:**
- Modify: `tests/test_utility_source_controls_workflow.py`
- Modify production files only in response to a witnessed RED failure.

**Interfaces:**
- Final focused proof that observer-only behavior preserves baseline.

- [ ] Add paired on/off step tests asserting actor, critic, target critic, optimizer, baseline PRNG progression, replay result, environment-call count, EC population/fitness, training counters and returned metrics are identical.
- [ ] Add no-extra-rollout and no-extra-replay-add counters for due/non-due checkpoints.
- [ ] Run all four focused files; expect any missing invariance to fail before fixes.
- [ ] Apply minimal fixes and rerun until all focused tests pass with no warnings/errors.
- [ ] Commit: `git commit -m "test: prove source-control baseline invariance"`.

### Task 11: Existing learner-utility and repository regression

**Files:** No planned production changes unless a new test exposes a real regression.

- [ ] Run `conda run -n evorl python -m pytest -p no:cacheprovider tests/test_learner_utility.py tests/test_erl_learner_utility_workflow.py -q`; require PASS and record counts/duration.
- [ ] Run `conda run -n evorl python -m pytest -p no:cacheprovider -q`; record every pass/fail/skip by name. Do not add skips or weaken assertions.
- [ ] Run `conda run -n evorl ruff check evorl/algorithms/erl/erl_td3/utility_source_controls.py evorl/algorithms/erl/erl_td3/erl_utility_source_controls.py evorl/algorithms/erl/erl_td3/utility_source_control_artifacts.py tests/test_utility_source_controls_*.py` and `ruff format --check` on the same paths.
- [ ] If regressions are caused by this branch, reproduce with a focused failing test and fix via RED/GREEN; otherwise document pre-existing failures exactly.
- [ ] Commit only necessary regression fixes: `git commit -m "fix: preserve source-control regression compatibility"`.

### Task 12: Documentation, environment provenance, and smoke preparation

**Files:**
- Create: `docs/learner_utility_source_controls/README.md`
- Modify: `tests/test_utility_source_controls_artifacts.py` for metadata completeness.

- [ ] Write a failing metadata test requiring all spec fields including real absolute paths, tmux names, package versions, tolerances and device.
- [ ] Implement metadata capture and documentation for schemas, safety, and reproduction.
- [ ] Verify branch/worktree/env/package/device values with read-only commands and include them in smoke `run_config.json`.
- [ ] Run focused artifact tests and documentation formatting checks.
- [ ] Commit: `git commit -m "docs: document learner utility source controls"`.

### Task 13: Brax/Swimmer smoke in dedicated tmux window

**Files:** Runtime outputs only under exclusive output root; no source edits unless smoke exposes a testable defect.

- [ ] Read `tmux ls`; reuse the existing EvoRL research session and create/reuse only a `source-controls` window fixed to the new worktree.
- [ ] Record `pwd`, branch, commit, `which python`, Python/JAX/JAXLIB/Flax/Optax versions and `jax.devices()`.
- [ ] Run `nvidia-smi`; proceed only if the new smoke fits without stopping/modifying replay-pilot.
- [ ] Create `/disk1/zmh_projects/evorl_utility_source_controls_outputs/{smoke,logs,cache,tests,summaries}` without touching old roots.
- [ ] Launch this command template from the dedicated window (timestamped run ID may replace `20261004T000000` but every path must stay under the root):

```bash
cd /disk1/zmh_projects/evorl_utility_source_controls
conda activate evorl
export JAX_COMPILATION_CACHE_DIR=/disk1/zmh_projects/evorl_utility_source_controls_outputs/cache
export WANDB_MODE=disabled
python scripts/train.py \
  env=brax/swimmer \
  agent=erl/erl-utility-source-controls \
  seed=42 \
  diagnostic_enabled=true \
  diagnostic_interval=1 \
  artifact_root=/disk1/zmh_projects/evorl_utility_source_controls_outputs/smoke/swimmer_seed42_20261004T000000 \
  pop_size=4 \
  source_batch_size=64 \
  num_reference_batches=4 \
  num_random_control_batches=8 \
  'step_scales=[1.0,0.5,0.1]' \
  warmup_iters=2 \
  total_episodes=80 \
  recorders='[log]' \
  checkpoint.enable=false \
  hydra.run.dir=/disk1/zmh_projects/evorl_utility_source_controls_outputs/smoke/swimmer_seed42_20261004T000000/hydra \
  2>&1 | tee /disk1/zmh_projects/evorl_utility_source_controls_outputs/logs/swimmer_seed42_20261004T000000.log
```

- [ ] Stop naturally after several valid checkpoints; do not begin a long run.
- [ ] Validate zero-step PASS, Reference-self status, all four sources, three scales, four references, output isolation, schema, no extra environment steps, and run metadata.
- [ ] If OOM/failure occurs, stop only the source-controls process; write a failing test before any code fix.

### Task 14: Final verification, implementation commit, and optional push

**Files:** All branch changes and generated compact documentation only; raw runtime artifacts remain outside Git.

- [ ] Run `git status --short`, `git diff --check`, focused tests, sealed learner-utility tests and full regression again from the final commit candidate.
- [ ] Confirm original `/disk1/zmh_projects/evorl` branch/status/processes are unchanged relative to recorded evidence.
- [ ] Confirm every runtime path is under the exclusive output root and enumerate all artifacts.
- [ ] Commit implementation with `git commit -m "diag: add learner utility source controls"` (earlier task commits may be retained; this commit covers final integrated changes).
- [ ] Push only `research/utility-source-controls` with `git push -u origin research/utility-source-controls` if remote permissions are healthy; never push another branch.
- [ ] Report the 36 required identity, environment, file, test, smoke, metric, artifact, commit and push fields; then stop without implementing utility-aware replay.

## 12. Risks, mitigations, and proving tests

| Risk | Mitigation | Test/evidence |
|---|---|---|
| JAX immutable references misunderstood | Retain immutable leaves; never write returned virtual state; exact PyTree tests | snapshot and paired baseline tests |
| JIT traces host strings/I/O | Keep orchestration/hooks eager; jit only numerical kernels | eager-hook and tracer-regression tests |
| Optimizer state accidentally reused/chained | Discard returned state and restart from frozen state per record | independent-start test |
| Full replay copy exhausts device memory | Pre-sample 4+8 batches before EA; retain metadata only | no-full-copy/context tests and GPU memory observation |
| Host/device synchronization overhead | `device_get` once at record boundary and only at due checkpoints | smoke timing/logs |
| Dynamic artifact writes corrupt summaries | Separate writer, append raw, atomic replace summaries | interrupted-write test |
| RL batch axes flattened incorrectly | Dedicated helper with agent/time/env parameterized cases | multi-axis core tests |
| EA source shape/provenance regression | Reuse sealed helper unchanged | existing learner-utility tests |
| Multi-env done masking leaks invalid transitions | Use shifted-done mask before flattening | RL-own validity tests |
| Diagnostic PRNG affects baseline | Fold from baseline key without assigning it | PRNG progression equivalence test |
| Controls sampled after EA exposure | Capture in `_make_pre_exposure_context()` before parent step | unique-marker ordering test |
| Deferred context outlives correct learner | fingerprint at start and deferred boundary; record exact/delayed | temporal-alignment tests |
| Reference storage increases device memory | Fixed 12 matched-size batches only; configurable counts | smoke `nvidia-smi` and metadata |
| Final-update scaling loses optimizer fidelity | Run full Optax transform, then scale only temporary update leaves | Adam scale semantics test |
| Reference-self definition drifts | Diagonal `R_k -> R_k` fixed in plan/schema | diagonal-control test |
| Output symlink escapes exclusive root | Resolve and test relative relationship before workflow build | symlink/escape guard tests |

## 13. Commit sequence

1. `diag: add source-control utility primitives`
2. `diag: guard source-control output root`
3. `diag: freeze source-control checkpoint state`
4. `diag: capture frozen replay controls pre exposure`
5. `diag: stage EA sources for deferred controls`
6. `diag: reuse baseline RL transitions for source controls`
7. `diag: evaluate four frozen utility source controls`
8. `diag: write source-control evidence and summaries`
9. `diag: integrate source-control workflow configuration`
10. `test: prove source-control baseline invariance`
11. optional targeted regression fix commit
12. `docs: document learner utility source controls`
13. final `diag: add learner utility source controls` integration/provenance commit if changes remain

## 14. Definition of done

- New class exists only on the isolated branch/worktree and subclasses the sealed diagnostic workflow.
- Pre-exposure controls are sampled before current EA replay insertion.
- RL-own comes from the one baseline RL rollout; environment steps and baseline key progression match diagnostic-off behavior.
- All four sources, three scales and four references use the same frozen learner/optimizer/target state.
- Zero-step is the same pipeline at scale 0 and passes; Reference-self warnings follow the spec.
- Runtime and focused tests prove baseline learner/replay/env/EA/counter invariance.
- Output guard rejects every escape before rollout; all smoke writes stay under the exclusive root.
- Focused, sealed learner-utility and full repository test results are recorded without new skips.
- Swimmer smoke produces several valid checkpoints and every required artifact/metric.
- Branch is committed, optionally pushed only to `research/utility-source-controls`, and the final 36-field report is delivered.
- No utility-aware replay implementation is started.
