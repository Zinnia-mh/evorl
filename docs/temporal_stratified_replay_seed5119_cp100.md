# Temporal Stratified Replay: seed 5119 CP100 pilot

## Frozen protocol

- Implementation commit: `19f0f0eadc931bec76654f5f4f3bfd4243b712d4`
- Seed: `5119`, selected and recorded before training
- Shared full-state CP10 SHA-256: `7580dab348b4221c6b10194b9233620d08b815040e74ca6825ea944d2367be5e`
- A: original full-resident Uniform Replay
- B: Uniform Replay over valid EA transitions from the most recent 30 generations
- C: 50% EA probability mass on ages 0–4 and 50% on ages 5–29, uniform within each stratum
- Frozen primary metric: Return AUC from CP20 through CP100; primary contrast: C versus B
- All evaluations used 128 episodes at generations 20, 30, ..., 100.

The GPU smoke passed before this pilot. No window, probability mass, seed, stopping rule, or training hyperparameter was changed after preregistration.

## Fairness and isolation checks

All arms used the same CP10 state. B and C both report the shared CP10 state as their restored and parent state hash. All arms finished with identical training and GA PRNG keys.

Each arm completed:

- 2,970,000 post-CP10 training environment steps (3,290,000 total at CP100)
- 1,440,000 Actor optimizer updates and 2,880,000 Critic optimizer updates
- 90 GA `ask`/`tell` generations
- 1,152 evaluation episodes on the same nine-checkpoint schedule
- 2,880,000 EA transitions and 90,000 RL transitions written after CP10

B/C changed only EA slots in the audited real TD3 input. Across 282,077 audited active EA slots per arm, B/C changed the EA index and changed zero RL indices. Neither arm sampled a negative-age or age-30-plus transition while the overlay was active (0 invalid temporal-age samples).

## Preregistered performance result

| Arm | AUC20–100 | Mean-return equivalent | Return100 | First Return >= 350 | Wall time |
|---|---:|---:|---:|---:|---:|
| A | 639,532,547.61 | 242.25 | 366.51 | CP70 / 2.30M steps | 1,704.53 s (28.41 min) |
| B | 465,998,787.06 | 176.51 | 355.47 | CP90 / 2.96M steps | 1,815.88 s (30.26 min) |
| C | 385,465,205.96 | 146.01 | 368.20 | CP90 / 2.96M steps | 1,851.95 s (30.87 min) |

Primary AUC contrasts:

| Contrast | Absolute difference | Relative difference |
|---|---:|---:|
| C versus B | -80,533,581.10 | -17.28% |
| C versus A | -254,067,341.65 | -39.73% |
| B versus A | -173,533,760.55 | -27.13% |

C's final return was 12.73 points above B and 1.70 points above A, but this late recovery did not offset its lower learning-curve area. It reached Return 350 at the same CP90 checkpoint as B and 660,000 environment steps later than A.

## Mechanism fidelity and cost

The observed active EA age distribution was:

| Arm | Ages 0–4 | Ages 5–29 | Target |
|---|---:|---:|---|
| B | 16.6671% | 83.3329% | Uniform over 30 generations (16.6667/83.3333%) |
| C | 49.9782% | 50.0218% | 50/50% |

The per-generation CDF refresh was small after compilation:

| Arm | Total refresh time | Active median | Active p95 | Median training step |
|---|---:|---:|---:|---:|
| A | 0 s | 0 ms | 0 ms | 15.16 s |
| B | 4.07 s | 1.081 ms | 1.293 ms | 18.37 s |
| C | 4.87 s | 0.987 ms | 1.205 ms | 18.78 s |

Relative to A, total wall time increased by 6.53% for B and 8.65% for C. C cost 36.07 seconds (1.99%) more than B. The steady probability refresh itself was approximately 1 ms; most observed step overhead belongs to executing and auditing the replacement sampling path rather than rebuilding the distribution.

## Exploratory CP30–100 view

The frozen-window exploratory AUC from CP30 through CP100 was:

- A: 627,112,037.66
- B: 453,693,726.88
- C: 373,897,838.86
- C versus B: -79,795,888.02 (-17.59%)

C and B were close at CP30–40, after which C lagged most strongly at CP60–70. Both recovered near CP90; C finished slightly higher at CP100. This is a late catch-up pattern, not evidence of improved early training efficiency.

## Decision

**STOP the current 5+25 Temporal Stratified Replay route.** In this single preregistered seed, the mechanism executed as designed but C was materially worse than B on the primary metric, and both temporal variants were worse than original Uniform A. The late CP100 return does not justify an independent replication of this fixed mechanism. This single-seed result is a bounded mechanism result, not a claim of statistical inferiority or a general statement about all temporal replay methods.

## Frozen remote artifacts

`/disk1/zmh_projects/evorl_artifacts/temporal_stratified_replay/seed5119-cp100-v1/`

- `preregistration.json`: frozen seed, arms, primary metric, and source hashes
- `shared-parent.json`: CP10 path and state hash
- `summary-{A,B,C}.json`: budgets, hashes, returns, timings, and mechanism audit
- `comparison.json`: preregistered performance summary
- `learning_curves.csv` and `learning_curves.svg`: complete nine-checkpoint curves
- `analysis.json`: contrasts, CP30–100 analysis, sampling fidelity, and timing statistics
- `runs/{A,B,C}/completion.json`: sealed final checkpoint references and report hashes
