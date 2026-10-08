# Learner-Utility Source-Control Diagnostic

This observer-only diagnostic tests whether learner-utility differences are explained by gradient-source mismatch rather than by changes to baseline training. It compares four source families—EA, RL-own, Replay-random, and Reference-self—from one frozen critic, target critic, optimizer state, replay snapshot, and baseline PRNG state.

For each source update, Utility is `L_ref(theta) - L_ref(theta')`, where `theta'` is produced by one virtual critic update and then discarded. All sources at a checkpoint use the same reference batches and checkpoint-scoped random keys; reference evaluation keys are independent of source and step scale. Zero-step results verify numerical invariance, while Reference-self is the calibration control for whether the frozen critic can improve on its own reference distribution.

## Status and artifacts

Each checkpoint retains the Task 7.1 status model unchanged: `VALID + COMPLETE`, `VALID + INCOMPLETE`, or `INVALID`. Case A–D interpretation is produced only by the artifact/report layer.

All output must remain under `/disk1/zmh_projects/evorl_utility_source_controls_outputs`; no repository, replay-pilot, or baseline artifact path is a valid output root. A run directory contains:

- `run_config.json`: thresholds, common-randomness policy, static runtime provenance, and checkpoint-keyed dynamic provenance.
- `per_source_utility.csv`: formal and zero-step source/reference measurements.
- `source_status.csv`: source eligibility and checkpoint validity.
- `checkpoint_summary.csv`, `cross_source_summary.csv`, `step_scale_summary.csv`, `reference_stability.csv`: derived summaries.
- `diagnostic_report.md`: Case A–D report and warnings.

Static provenance records repository/worktree paths, branch and commit, artifact root, tmux session/window, conda environment, Python/JAX/JAXLIB/Flax/Optax versions, actual JAX device, environment, seed, population, and UTC timestamp. Dynamic provenance is stored in `checkpoint_provenance[checkpoint]` with the baseline PRNG fingerprint, replay-metadata fingerprint, and temporal alignment. The diagnostic is observer-only: it must not add environment steps, advance the baseline PRNG, mutate learner/replay/EC state, or alter returned training metrics.

## Task 13 fixed parameters

Use `env=brax/swimmer`, `seed=42`, `pop_size=4`, `source_batch_size=64`, `num_reference_batches=4`, `num_random_control_batches=8`, `source_control_step_scales=[1.0,0.5,0.1]`, and `diagnostic_interval=1`. Keep the committed numerical tolerances and Case A–D thresholds unless a later approved plan explicitly changes them. Use the fixed worktree, branch, `evorl-conflict` environment, and inject the actual tmux session/window at launch; choose an absolute, new run directory below the allowed output root.

## Reproduction

Configuration-only validation (safe; does not build the workflow or start rollout):

```bash
cd /disk1/zmh_projects/evorl_utility_source_controls
conda run -n evorl-conflict python scripts/train.py \
  env=brax/swimmer agent=erl/erl-utility-source-controls seed=42 \
  diagnostic_enabled=true diagnostic_interval=1 \
  artifact_root=/disk1/zmh_projects/evorl_utility_source_controls_outputs/smoke/PREFLIGHT_ONLY \
  pop_size=4 source_batch_size=64 num_reference_batches=4 \
  num_random_control_batches=8 'source_control_step_scales=[1.0,0.5,0.1]' \
  --cfg job --resolve
```

Task 13 execution template (documented only; do not run during Task 12):

```bash
tmux send-keys -t evorl-conflict-research:source-controls -l \
  'cd /disk1/zmh_projects/evorl_utility_source_controls && env -u LD_LIBRARY_PATH /home/admin001/anaconda3/envs/evorl-conflict/bin/python scripts/train.py env=brax/swimmer agent=erl/erl-utility-source-controls seed=42 diagnostic_enabled=true diagnostic_interval=1 artifact_root=/disk1/zmh_projects/evorl_utility_source_controls_outputs/smoke/task13 pop_size=4 source_batch_size=64 num_reference_batches=4 num_random_control_batches=8 "source_control_step_scales=[1.0,0.5,0.1]" +tmux_session=evorl-conflict-research +tmux_window=source-controls' Enter
```

The verified Task 12 runtime uses Python 3.11.16, JAX/JAXLIB 0.9.0, Flax 0.11.2, and Optax 0.2.8 in `evorl-conflict`. Pre-smoke GPU verification reports JAX backend `gpu`, device NVIDIA GeForce RTX 4090, and driver 550.163.01. A prior transient CUDA/cuSPARSE initialization failure was observed once during provenance inspection but was not reproducible in subsequent normal or `LD_LIBRARY_PATH`-clean runs; no dependency changes were made. The repository-wide historical `tests/test_mjx_env.py` collection issue is pre-existing and is not addressed by this diagnostic.

## Verified Smoke

RUN_ID `swimmer_seed42_20261008T170134` ran Brax/Swimmer with seed 42 on an RTX 4090 using JAX GPU. All 15 due checkpoints were valid and comparison eligible (0 invalid); zero-step, four sources, three scales, four references, eight replay controls, and runtime provenance passed validation. This is smoke validation only, not formal multi-seed evidence.

## Interpretation limits

Utility is one-step frozen critic learner utility. It is not actor return. It is not long-horizon policy improvement. It does not establish causality.
