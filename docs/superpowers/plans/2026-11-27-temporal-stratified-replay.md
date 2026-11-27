# Temporal Stratified Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement and run the fixed A/B/C Temporal Stratified Replay CP100 pilot.

**Architecture:** A pure fixed-shape temporal distribution feeds the existing EA-only replay overlay. A workflow subclass uses minimal base hooks and a dedicated audited runner forks all arms from one A CP10 state.

**Tech Stack:** Python, JAX, Brax, Orbax, pytest, tmux.

**Spec:** `docs/superpowers/specs/2026-11-27-temporal-stratified-replay-design.md`

## Global Constraints

- Start from `9b46e32c808e6756816555ab204f1caf6b208231` on `codex/temporal-stratified-replay`.
- Windows are fixed at recent 5 and history 25 generations; C mass is fixed 50/50.
- No Fitness admission, Utility scoring, new EA mechanism, extra rollout, or extra TD3 update.
- All artifacts stay under a new remote directory and never overwrite prior experiments.

## Review Focus

- Partial/unset ring slots must never receive probability.
- Pre-30-generation operation must be exact baseline identity.
- Empty recent/history strata must fail safe to baseline identity.
- RL positions and indices must remain byte-for-byte unchanged.
- Resume must preserve intervention RNG and the next sampled indices.

### Task 1: Temporal probability primitive

**Files:**
- Create: `evorl/algorithms/erl/erl_td3/temporal_replay.py`
- Test: `tests/test_temporal_stratified_replay.py`

- [ ] Write failing tests for B uniformity, C 50/50 mass, pre-window fallback, empty strata, and ring invalid slots.
- [ ] Run the tests and verify failure because the API is absent.
- [ ] Implement `build_temporal_ea_distribution(...)` with fixed-shape probability/CDF output.
- [ ] Run the focused tests green.

### Task 2: Real learner integration

**Files:**
- Modify: `evorl/algorithms/erl/erl_td3/erl_utility_replay.py`
- Modify: `evorl/algorithms/erl/erl_td3/utility_replay_training.py`
- Modify: `evorl/algorithms/erl/erl_td3/utility_replay_budget.py`
- Modify: `evorl/algorithms/erl/erl_td3/temporal_replay.py`
- Test: `tests/test_temporal_stratified_replay.py`

- [ ] Write failing tests for actual EA replacement, exact RL identity, full 32k/1k write path, and no Utility diagnostic invocation.
- [ ] Add behavior-preserving workflow/runner validation hooks and `TemporalStratifiedReplayWorkflow`.
- [ ] Run focused and existing replay tests green.

### Task 3: Checkpointed runner and GPU smoke

**Files:**
- Create: `configs/agent/erl/erl-temporal-stratified-replay.yaml`
- Create: `scripts/run_temporal_stratified_replay.py`
- Test: `tests/test_temporal_stratified_replay.py`

- [ ] Write failing tests for protocol freezing and intervention-key resume identity.
- [ ] Implement seed-5119 preregistration, shared CP10 fork, append-only audits, and non-overwriting outputs.
- [ ] Run focused tests and a real short GPU smoke; record probability fidelity, writes, budgets, compilation, and throughput.

### Task 4: CP100 pilot and report

**Files:**
- Create: `docs/temporal_stratified_replay_seed5119.md`

- [ ] Run A/B/C from the same CP10 state to CP100 in tmux.
- [ ] Verify complete checkpoints, budgets, distributions, RL identity, and finite results.
- [ ] Compute AUC20–100, Return100, first Return≥350, sampling age mass, and cost.
- [ ] Commit and push code and the complete report; leave the worktree clean.
