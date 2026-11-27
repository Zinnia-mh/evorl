# Temporal Stratified Replay design

Date: 2026-11-27

## Intent

Test whether increasing the replay probability of recent EA transitions improves ERL-TD3 learning without changing experience generation, admission, GA, RL injection, buffer writes, TD3 updates, or the realized RL replay slots.

## Frozen arms

- A: original all-resident Uniform Replay.
- B: uniform EA probability over resident transitions with ages 0–29.
- C: 50% EA probability mass on ages 0–4 and 50% on ages 5–29, uniform within each stratum.
- Before the complete 30-generation window is available, B and C preserve the original baseline draw exactly. No parameter search is allowed.

## Architecture

Add a pure temporal-distribution builder and a workflow subclass. The subclass reuses the existing baseline draw, float64 CDF, 52-bit intervention draws, and EA-only overlay. Three small hooks in `ERLUtilityReplayWorkflow` allow the subclass to validate its own config, enable the weighted learner path without Utility `alpha`, and replace the per-generation Utility distribution refresh. Existing Utility Replay behavior is unchanged.

The existing `UtilityReplayState` checkpoints the independent intervention key and fixed-shape distribution. B/C never score a source, construct a reference batch, or execute a virtual TD3 update. All 32 EA trajectories and the RL trajectory use the original write path.

## Approaches considered

1. **Chosen: small hooks plus a separate temporal module.** Reuses the audited sampler and actual learner path while keeping the temporal mechanism isolated.
2. Duplicate `_weighted_step` in a new workflow. Rejected because a large copy could silently diverge from GA/TD3 budgets.
3. Sample outside the workflow in the runner. Rejected because it would not prove that real TD3 inputs used the temporal distribution.

## Safety and fallback

- Resident EA means `index < buffer_size` and `origin_type == EA_ORIGIN`.
- Age is `active_generation - source_generation`; future, unset, RL, overwritten, and age ≥30 slots have zero temporal probability.
- If the 30-generation window is not complete, or a required stratum is empty, preserve the original draw.
- The original replay key produces the baseline draw; a checkpointed intervention key replaces only EA positions. RL positions and indices are identical.
- Build one probability vector/CDF after both EA and RL insertion and reuse it for all learner updates in the generation.

## Experiment contract

Pre-register unused seed 5119. Generate A through CP10, then fork A/B/C from that exact full state and run to CP100. Primary metric is Return AUC over CP20–100. C vs B is primary; A vs B is auxiliary. No result-dependent stopping or parameter changes.
