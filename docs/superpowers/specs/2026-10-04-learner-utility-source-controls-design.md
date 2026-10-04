# Learner-Utility Source-Control Diagnostic Design

## Objective

Determine whether the approximately 88.98% negative EA-source utility rate is specific to EA-generated data relative to the current RL learner, or is instead common to ordinary RL/replay data under the existing reference-loss and optimizer-faithful virtual-update diagnostic.

This stage is diagnostic only. It must not implement utility-aware replay or change baseline ERL/TD3 training semantics.

## Repository and operational isolation

- Canonical repository: `/disk1/zmh_projects/evorl`.
- Immutable baseline: `research/learner-utility-diagnostic` at `df954935334e201f3e2759b848d5be2c177833f7`.
- New branch: `research/utility-source-controls`.
- New worktree: `/disk1/zmh_projects/evorl_utility_source_controls`.
- Exclusive output root: `/disk1/zmh_projects/evorl_utility_source_controls_outputs`.
- Reuse the existing `evorl` Conda environment without package, CUDA, Python, JAX, Flax, or Optax upgrades.
- Reuse one existing EvoRL tmux session where possible and dedicate a `source-controls` window to this worktree. Never alter or stop the replay-pilot window or processes.
- Never switch, reset, clean, stash, or edit `/disk1/zmh_projects/evorl` while replay-pilot is running.

All logs, CSV/JSON/Markdown reports, plots, checkpoints, temporary artifacts, run metadata, and experiment-specific caches must be written under the exclusive output root. No new experimental output may be written under the canonical repository, replay-pilot outputs, or the frozen diagnostic outputs.

## Architecture

Add `ERLUtilitySourceControlsWorkflow` as a subclass of the frozen `ERLLearnerUtilityWorkflow`. The new workflow exists only on the new branch and does not change the behavior of `ERLLearnerUtilityWorkflow` or the replay-pilot implementation.

The source-control workflow uses delayed diagnostics on the EA and RL rollouts that baseline training already performs. It must add zero environment steps and must not perform a diagnostic-only rollout.

At the beginning of each due checkpoint, the workflow captures a diagnostic-only, read-only snapshot of:

- critic parameters and target critic parameters;
- critic optimizer state;
- baseline PRNG fingerprint;
- pre-exposure replay state/fingerprint;
- generation and training-counter metadata.

The snapshot does not pause, replace, or write back to the real learner. Baseline training continues in its existing order. All virtual computations start independently from the same snapshot and are discarded after measurement.

## Baseline timing and deferred measurement

The verified baseline order is:

1. split the baseline key for the iteration;
2. perform EA rollout and add its transitions to replay;
3. perform RL rollout and add its transitions to replay;
4. run critic/actor learner updates from replay;
5. perform the EA update/injection and advance metrics/state.

There is no learner update between EA and RL rollout, so the expected `source_temporal_alignment` is `exact`. The implementation must compute and record this rather than assume it. If the upstream order changes, it must record `delayed` and preserve baseline order rather than reshuffle training.

During EA rollout, the workflow retains the pre-mixing EA trajectories, freezes four reference batches and eight configurable replay-random control batches from the pre-exposure replay snapshot, and records the frozen learner state. During the baseline RL rollout it retains a read-only view of the transitions that rollout already produces. After the RL rollout, it evaluates all sources against the frozen checkpoint state before any real learner update.

If the baseline RL transitions contain fewer valid samples than the matched source batch size, RL-own is ineligible with an explicit reason. The workflow must not generate extra transitions.

## Utility computation and source controls

The existing TD3 critic-loss adapter and optimizer-faithful virtual update remain authoritative:

`U = L_ref(theta; B_ref) - L_ref(theta_virtual; B_ref)`

Each virtual update begins from the same frozen critic parameters, optimizer state, target critic, source batch, and reference controls. Virtual updates must never be chained across sources or scales.

The four source types are:

1. `EA`: matched batches from each pre-replay-mixing EA individual, retaining fitness and provenance.
2. `RL-own`: a matched batch from the real RL rollout already performed by baseline training.
3. `Replay-random`: eight configurable matched batches sampled read-only from the frozen pre-exposure replay state with diagnostic-only keys.
4. `Reference-self`: the reference batch itself used as the source batch.

Each source is evaluated at step scales `1.0`, `0.5`, and `0.1`. Scaling applies only to temporary Optax updates; it must not change the real optimizer learning rate or state.

A zero-step/no-op update is a hard correctness gate. Its absolute utility must be within a dtype-aware numerical tolerance. Failure marks the checkpoint `INVALID` and excludes it from formal summaries while preserving raw evidence. Reference-self negativity is a warning, not automatically invalid; systematic abnormality blocks an EA-mismatch conclusion.

Each source/scale is evaluated against four configurable frozen reference batches. Near-zero values do not vote on sign consistency. Sign consistency is the majority non-zero sign count divided by the total non-zero sign count. If all values are near zero, `sign_consistency` is absent/NA and `all_near_zero=true`.

## Baseline invariance

Diagnostic execution must not change the real actor, critic, target critic, optimizer state, baseline PRNG, replay buffer, environment state, EA population, evolutionary fitness, checkpoints, logs, or training counters. Diagnostic sampling uses derived keys and read-only state. Invariance checks compare the relevant trees/state before and after the diagnostic hook and reject the checkpoint on mutation.

## Output safety

Before any environment rollout, canonicalize `artifact_root`. Accept only the exclusive output root or descendants. Reject the canonical repository, original artifacts, replay-pilot outputs, frozen diagnostic outputs, symlink escapes, and relative/default paths that resolve outside the exclusive root.

Each run has an isolated run ID and directories for logs, artifacts, checkpoints, temporary files, and cache. Artifacts are staged within the run directory and finalized atomically. Failures retain evidence and explicit reasons without contaminating other runs.

## Artifact schema

`per_source_utility.csv` contains checkpoint, generation, source type/ID, step scale, reference ID/fingerprint, source batch size, losses before/after, utility, TD statistics, eligibility/reason, checkpoint status, temporal alignment, and EA fitness where applicable.

`checkpoint_summary.csv` groups valid records by generation, step scale, and source type and reports count, mean, median, standard deviation, min/max, and positive/negative/near-zero fractions.

`cross_source_summary.csv` compares all four sources and includes negative/positive rates, mean/median utility, EA-minus-RL and EA-minus-Replay negative-rate differences, and EA-minus-RL and EA-minus-Replay median-utility differences.

`step_scale_summary.csv` compares utility distributions and negative rates across `1.0`, `0.5`, and `0.1`.

`reference_stability.csv` stores per checkpoint/source/scale utility mean, median, standard deviation, negative and near-zero fractions, sign consistency, and `all_near_zero`.

`run_config.json` records repository/worktree/branch/commit identity, output root, tmux session/window, Conda and package versions, device, environment, seed, population, batch sizes, reference/random-control counts, step scales, tolerances, timestamp, baseline PRNG/replay fingerprints, and temporal alignment.

`diagnostic_report.md` applies only the predefined Case A-D interpretation rules and never uses causal language such as “proves.”

## Error handling

Reject a checkpoint for zero-step failure, batch-size mismatch, missing frozen state, chained source state, missing schema fields, replay mutation, or learner mutation. Preserve the raw evidence and the rejection reason. Reference-self negativity emits a warning unless systematic abnormality makes interpretation unsafe.

## Tests

Development follows strict red-green-refactor. Focused tests cover zero-step behavior, independent virtual starts, real-state invariance, target critic and PRNG invariance, replay immutability, real-transition RL-own reuse, absence of diagnostic rollouts, batch matching, replay-random/reference-self controls, multi-reference aggregation, near-zero-aware sign consistency, all-near-zero handling, step scaling without real-LR changes, output-root guards, artifact schemas, frozen-snapshot deferred computation, temporal alignment, JAX/JIT shape/tracer behavior, counters, and EA population/fitness invariance.

Verification order is focused source-control tests, existing learner-utility tests, then the full repository suite. Existing tests may not be deleted, skipped, or weakened.

## Smoke execution

After all tests pass, run only a small Brax/Swimmer smoke with population 4, source batch size 64, four reference batches, eight replay-random batches, and step scales 1.0/0.5/0.1 until several valid diagnostic checkpoints exist.

Before launch, verify the fixed worktree/branch/environment and inspect `nvidia-smi`. Run in the dedicated `source-controls` tmux window with explicit output/log/cache paths under the exclusive output root. If the new run OOMs, stop only the new run. Do not stop or modify replay-pilot.

## Interpretation and stopping rule

Apply only the predefined Case A-D interpretation rules. Report checkpoint-level EA fitness/utility Spearman, Kendall tau-b, inversion rate, top-fitness utility rank, and exact top-fitness/top-utility agreement alongside all source-control statistics.

After implementation, tests, smoke, artifacts, commit, and optional push are reported, stop. Do not implement or begin utility-aware replay.
