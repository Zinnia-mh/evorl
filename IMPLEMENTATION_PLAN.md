# Learner Utility Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add a diagnostic-only ERL workflow that measures whether each EA individual's experience lowers the current TD3 learner's frozen reference critic loss, without changing Original ERL training behavior.

**Architecture:** Preserve `erl_origin.py` as the baseline authority. Add a separate learner-utility workflow and pure JAX/Optax helpers. The workflow captures per-individual EC trajectories before replay mixing, derives diagnostic-local keys without writing baseline state, applies virtual critic updates to functional copies, and writes host-side records outside the training state.

**Tech Stack:** Python 3.11, JAX, Optax, Chex, EvoRL TD3, pytest, CSV/JSON, SVG.

**Spec:** `DESIGN.md`

## Global Constraints

- Branch starts at `main@387d589`; do not merge or copy conflict-aware code.
- Do not modify `erl_origin.py`, baseline fitness, EA selection, replay sampling/eviction, or real TD3 updates.
- Do not implement utility-aware sampling, filtering, replay, selection, or multi-objective EA.
- Primary utility is optimizer-faithful: use the current critic Optax transform and a cloned critic optimizer state.
- Virtual output must never be written to critic, optimizer, target network, actor, replay, EA state, metrics, or baseline PRNG state.
- Use the learner's exact TD3 critic-loss/target construction; do not build an approximate diagnostic loss.
- Establish `source_id -> B_i` before replay mixing. Every compared source has the same fixed `source_batch_size`; record and exclude undersized sources.
- TD error and utility use the same `B_i`. Setting A is required; Setting B anchor reference is deferred beyond the first smoke.
- All Python runs use `env -u LD_LIBRARY_PATH /home/admin001/anaconda3/envs/evorl-conflict/bin/python ...`.
- Temporary output remains under `.codex_validation/learner_utility_diagnostic/` and is not committed.

## Review Focus

- A virtual update could write a real optimizer/target leaf; Task 1 snapshots all protected trees.
- Source batches could have unequal cardinality; Task 2 records ineligible short sources.
- Diagnostic keys could advance baseline randomness; Task 3 asserts baseline-key equality.
- Utility could use a different target construction; Task 1 compares the loss adapter to the real loss.
- Disabled diagnostics could change training; Task 4 exact-compares paired trajectories.

---

## File Structure

- Create `evorl/algorithms/erl/erl_td3/learner_utility.py`: pure source batching, TD error, key derivation, shared loss adapter, and virtual utility helpers.
- Create `evorl/algorithms/erl/erl_td3/erl_learner_utility.py`: isolated workflow subclass and host-side diagnostic record hook.
- Create `configs/agent/erl/erl-learner-utility.yaml`: disabled-by-default cadence, Setting A, and batch-size settings.
- Create `tests/test_learner_utility.py`: pure primitives and non-mutation tests.
- Create `tests/test_erl_learner_utility_workflow.py`: provenance, reference, disabled-equivalence, and CSV tests.
- Create `tests/test_learner_utility_analysis.py`: fixture-driven analysis tests.
- Create `scripts/analyze_learner_utility.py`: CSV-only statistics and first scatter SVG.
- Create `docs/research/learner_utility_diagnostic_report.md`: methods and smoke report.

## Task 1: Functional optimizer-faithful utility primitives

**Files:**
- Create: `evorl/algorithms/erl/erl_td3/learner_utility.py`
- Create: `tests/test_learner_utility.py`

**Interfaces:**
- Produces `derive_diagnostic_key(base_key, generation, tag)`, `critic_loss_adapter(agent, agent_state, batch, key)`, and `virtual_critic_utility(agent, optimizer, agent_state, critic_opt_state, source_batch, reference_batch, loss_key) -> UtilityResult`.
- `UtilityResult` contains scalar `utility`, `loss_ref_before`, and `loss_ref_after`; it contains no training state.

- [ ] **Step 1: Write failing loss-fidelity and non-mutation tests**

```python
def test_virtual_update_keeps_real_training_trees_identical(td3_fixture):
    before = td3_fixture.snapshot_training_trees()
    result = virtual_critic_utility(**td3_fixture.utility_inputs())
    assert result.utility.shape == ()
    chex.assert_trees_all_equal(before, td3_fixture.snapshot_training_trees())

def test_loss_adapter_matches_real_td3_critic_loss(td3_fixture):
    assert jnp.allclose(
        critic_loss_adapter(**td3_fixture.loss_inputs()),
        td3_fixture.real_critic_loss(),
        rtol=1e-6, atol=1e-6,
    )
```

- [ ] **Step 2: Run the tests and observe RED**

Run: `python -m pytest tests/test_learner_utility.py -q`

Expected: import failure for absent utility helpers.

- [ ] **Step 3: Implement the pure helpers**

Call the real TD3 critic-loss definition. Compute gradients on functional critic parameters; call `optimizer.update(grads, cloned_opt_state, params)` and `optax.apply_updates`. Compute both reference losses with the same frozen batch. Discard virtual parameters and virtual optimizer state.

- [ ] **Step 4: Run the tests and observe GREEN**

Run: `python -m pytest tests/test_learner_utility.py -q`

Expected: protected critic/optimizer/target/actor/replay/EA/key trees are bitwise equal.

- [ ] **Step 5: Commit**

```bash
git add evorl/algorithms/erl/erl_td3/learner_utility.py tests/test_learner_utility.py
git commit -m "feat: add functional learner utility primitives"
```

## Task 2: Pre-mixing provenance and equal source batches

**Files:**
- Modify: `evorl/algorithms/erl/erl_td3/learner_utility.py`
- Modify: `tests/test_learner_utility.py`
- Create: `tests/test_erl_learner_utility_workflow.py`

**Interfaces:**
- Consumes raw EC trajectories with population axis intact.
- Produces one `SourceBatch` per `source_id`, with `eligible`, `reason`, and a batch of exactly `M` transitions.

- [ ] **Step 1: Write failing provenance tests**

```python
def test_source_batches_are_created_before_flattening(raw_pop_trajectory):
    batches = build_source_batches(raw_pop_trajectory, generation=12, size=4, key=KEY)
    assert [x.source_id for x in batches] == ["gen_12_ind_0", "gen_12_ind_1"]
    assert all(x.batch.rewards.shape[0] == 4 for x in batches)

def test_short_source_is_ineligible(short_source_trajectory):
    result = source_batch_or_ineligible(short_source_trajectory, size=256, key=KEY)
    assert result.eligible is False
    assert result.reason == "insufficient_valid_transitions"
```

- [ ] **Step 2: Run RED tests**

Run: `python -m pytest tests/test_learner_utility.py tests/test_erl_learner_utility_workflow.py -q`

Expected: missing provenance APIs.

- [ ] **Step 3: Implement fixed-size provenance helpers**

Select valid transitions from each raw individual trajectory with a diagnostic-local deterministic key. Preserve the existing flatten/add input unchanged. Store optional lineage fields only if directly available; otherwise emit empty fields.

- [ ] **Step 4: Run GREEN tests**

Run: `python -m pytest tests/test_learner_utility.py tests/test_erl_learner_utility_workflow.py -q`

Expected: pre-mixing IDs, equal eligible sizes, and explicit short-source exclusion.

- [ ] **Step 5: Commit**

```bash
git add evorl/algorithms/erl/erl_td3/learner_utility.py tests/test_learner_utility.py tests/test_erl_learner_utility_workflow.py
git commit -m "feat: add learner utility source provenance"
```

## Task 3: Setting A workflow and deterministic diagnostic keys

**Files:**
- Create: `evorl/algorithms/erl/erl_td3/erl_learner_utility.py`
- Create: `configs/agent/erl/erl-learner-utility.yaml`
- Modify: `tests/test_erl_learner_utility_workflow.py`

**Interfaces:**
- Consumes ERL state, raw source batches, immutable base seed, and generation.
- Produces host-side diagnostic records only; disabled mode produces the baseline training state.

- [ ] **Step 1: Write failing shared-reference and key-isolation tests**

```python
def test_diagnostic_key_does_not_advance_baseline_key(workflow_state):
    before = workflow_state.key
    derived = derive_diagnostic_key(before, generation=7, tag=UTILITY_TAG)
    assert derived.shape == before.shape
    chex.assert_trees_all_equal(before, workflow_state.key)

def test_all_sources_share_one_current_reference_batch(workflow_fixture):
    records = workflow_fixture.run_one_enabled_diagnostic_checkpoint()
    assert len({r.reference_batch_fingerprint for r in records if r.eligible}) == 1
```

- [ ] **Step 2: Run RED tests**

Run: `python -m pytest tests/test_erl_learner_utility_workflow.py -q`

Expected: import failure for `ERLLearnerUtilityWorkflow`.

- [ ] **Step 3: Implement the diagnostic subclass**

Subclass baseline workflow; do not edit `erl_origin.py`. At every enabled K-generation checkpoint, derive a diagnostic-local key, sample one `B_ref_current` once from replay, evaluate every eligible source against it, and emit records after the training transition. Disabled is the config default and must not allocate records or alter state/key transitions.

- [ ] **Step 4: Run GREEN tests**

Run: `python -m pytest tests/test_erl_learner_utility_workflow.py -q`

Expected: shared reference fingerprint and unchanged baseline key/protected trees.

- [ ] **Step 5: Commit**

```bash
git add evorl/algorithms/erl/erl_td3/erl_learner_utility.py configs/agent/erl/erl-learner-utility.yaml tests/test_erl_learner_utility_workflow.py
git commit -m "feat: add learner utility diagnostic workflow"
```

## Task 4: Disabled exact equivalence and CSV contract

**Files:**
- Modify: `evorl/algorithms/erl/erl_td3/erl_learner_utility.py`
- Modify: `tests/test_erl_learner_utility_workflow.py`

- [ ] **Step 1: Write failing equivalence and record-contract tests**

```python
def test_disabled_workflow_is_exactly_equivalent_to_original(tiny_erl_config):
    original, diagnostic = run_paired_workflows(tiny_erl_config, seed=0, steps=3)
    chex.assert_trees_all_equal(original.final_state, diagnostic.final_state)
    chex.assert_trees_all_equal(original.metrics, diagnostic.metrics)

def test_enabled_record_has_required_fields(enabled_checkpoint):
    row = enabled_checkpoint.records[0]
    assert REQUIRED_COLUMNS <= row.keys()
    assert row["source_batch_size"] == enabled_checkpoint.config.source_batch_size
```

- [ ] **Step 2: Run RED tests**

Run: `python -m pytest tests/test_erl_learner_utility_workflow.py -q`

Expected: disabled comparison or explicit CSV schema is not yet wired.

- [ ] **Step 3: Implement CSV-only record emission**

Use a fixed column list covering identity, fitness/return, TD statistics, utility, reference losses, source/reference sizes, critic LR, reference setting, checkpoint, and timestamp. Write only below the configured validation directory. Disabled mode creates no record output.

- [ ] **Step 4: Run full correctness suite**

Run: `python -m pytest -p no:cacheprovider tests/test_learner_utility.py tests/test_erl_learner_utility_workflow.py -q`

Expected: all seven correctness gates pass.

- [ ] **Step 5: Commit**

```bash
git add evorl/algorithms/erl/erl_td3/erl_learner_utility.py tests/test_erl_learner_utility_workflow.py
git commit -m "test: verify learner utility diagnostic isolation"
```

## Task 5: CSV analysis and 1 environment x 1 seed smoke

**Files:**
- Create: `scripts/analyze_learner_utility.py`
- Create: `tests/test_learner_utility_analysis.py`
- Create: `docs/research/learner_utility_diagnostic_report.md`

- [ ] **Step 1: Write failing analysis tests**

```python
def test_analysis_reports_correlations_overlap_and_quadrants(tmp_path):
    output = analyze_csv(fixture_csv, tmp_path)
    assert {"pearson_fitness_utility", "spearman_fitness_utility", "top20_overlap", "quadrants"} <= output.keys()

def test_analysis_excludes_ineligible_sources(tmp_path):
    output = analyze_csv(csv_with_short_source, tmp_path)
    assert output["eligible_records"] == 3
    assert output["excluded_records"] == 1
```

- [ ] **Step 2: Run RED tests**

Run: `python -m pytest tests/test_learner_utility_analysis.py -q`

Expected: import failure for `analyze_csv`.

- [ ] **Step 3: Implement CSV-only statistics and core figure**

Compute Pearson/Spearman for fitness/utility and TD/utility, population-relative quadrants, top-20-percent overlap, and stage grouping when present. Emit only one required smoke visual: SVG fitness-versus-utility scatter. State in the report that results are observational and do not authorize replay intervention.

- [ ] **Step 4: Run GREEN analysis tests and smoke**

Run: `python -m pytest tests/test_learner_utility_analysis.py -q`

Run: one low-cost continuous-control environment by one seed in the sole `evorl-conflict-research` tmux session; save all outputs in `.codex_validation/learner_utility_diagnostic/`.

Expected: schema-valid CSV, protected-state evidence, and the core scatter. This validates the pipeline only; it does not claim the hypothesis.

- [ ] **Step 5: Commit only reusable code/tests/report**

```bash
git add scripts/analyze_learner_utility.py tests/test_learner_utility_analysis.py docs/research/learner_utility_diagnostic_report.md
git commit -m "docs: add learner utility diagnostic smoke report"
```

## Plan Self Review

- Every design constraint maps to Tasks 1-5.
- The first implementation milestone stops at Setting A and one environment x one seed.
- The plan never authorizes utility-aware replay behavior.
- The seven correctness gates are tested before smoke.
- Interface names introduced in Tasks 1-3 are consistently consumed by later tasks.
