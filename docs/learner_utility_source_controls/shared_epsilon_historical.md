# Shared-epsilon historical RL-derived utility diagnostic

Completed on 2026-11-24 using the existing uniform A, seed 2027 Swimmer checkpoints at generations 50, 150 and 300. These describe stages within a 300-generation pilot, not mid/late stages of a 3031-generation training run.

## Controls and restoration

- Small/large use the same Gaussian epsilon with scales 0.005 and 0.05. All three EC sources share rollout and transition-selection keys.
- Fixed observation distance uses reference batch 0 from the frozen historical pre-exposure Replay, shared across actors at that checkpoint. Its fingerprint is persisted; trajectory distance is retained separately.
- Historical file seals, full native state hashes, every projected leaf shape/dtype, historical configuration checksum/contract and runtime package versions passed validation.
- Saved state at g is used for hypothetical next-rollout diagnostics at g+1 (51, 151, 301). No future Replay, EA tell or TD3 update is used. Full baseline state fingerprints match before/after.
- Native population is 32; diagnostic Original EA is the fixed first four population members, not a new independent population or a population-wide estimate. Source batch size is 256; 4 frozen reference batches; original exploration settings retained.
- Only 3 checkpoint comparison units. Per scale: EA 12 source events/48 reference records; each RL source 3/12; Replay-random 24/96; Reference-self 12/12. Repeated reference records and temporally related source events are not independent replicates.

## Step-scale 1.0 checkpoint means

| Saved generation | EA | Unmutated EC | Small | Large | Small minus EA | Large minus EA |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.000688 | -0.001551 | -0.000396 | -0.000249 | -0.001085 | -0.000937 |
| 150 | -0.001343 | -0.001054 | -0.001116 | -0.001074 | 0.000226 | 0.000269 |
| 300 | -0.000242 | -0.000423 | -0.000585 | -0.001101 | -0.000344 | -0.000859 |

## Aggregate descriptive distributions

Each cell reports mean / median of reference measurements / positive fraction / near-zero fraction (threshold 1e-6). Checkpoints receive equal weight because per-checkpoint source counts match. Full per-checkpoint Fitness, TD Error, trajectory and fixed distances are in analysis.csv.

| Source | Scale | Mean | Median | Positive | Near zero |
|---|---:|---:|---:|---:|---:|
| EA | 1.0 | -0.000299 | -0.000312 | 27.1% | 0.0% |
| EA | 0.5 | 0.000325 | 0.000340 | 68.8% | 0.0% |
| EA | 0.1 | 0.000136 | 0.000156 | 70.8% | 2.1% |
| RL-own | 1.0 | -0.013241 | -0.001143 | 0.0% | 0.0% |
| RL-own | 0.5 | -0.003484 | -0.000329 | 33.3% | 0.0% |
| RL-own | 0.1 | -0.000176 | -0.000016 | 33.3% | 0.0% |
| RL-unmutated-EC | 1.0 | -0.001009 | -0.000696 | 0.0% | 0.0% |
| RL-unmutated-EC | 0.5 | -0.000044 | -0.000072 | 41.7% | 0.0% |
| RL-unmutated-EC | 0.1 | 0.000065 | 0.000036 | 58.3% | 0.0% |
| RL-derived-small | 1.0 | -0.000699 | -0.000700 | 8.3% | 0.0% |
| RL-derived-small | 0.5 | 0.000019 | -0.000087 | 41.7% | 0.0% |
| RL-derived-small | 0.1 | 0.000058 | 0.000003 | 50.0% | 0.0% |
| RL-derived-large | 1.0 | -0.000808 | -0.000778 | 8.3% | 0.0% |
| RL-derived-large | 0.5 | -0.000120 | -0.000101 | 16.7% | 0.0% |
| RL-derived-large | 0.1 | 0.000011 | 0.000020 | 66.7% | 0.0% |
| Replay-random | 1.0 | 0.000385 | 0.000281 | 68.8% | 0.0% |
| Replay-random | 0.5 | 0.000498 | 0.000492 | 83.3% | 0.0% |
| Replay-random | 0.1 | 0.000166 | 0.000160 | 100.0% | 0.0% |
| Reference-self | 1.0 | 0.002154 | 0.001034 | 100.0% | 0.0% |
| Reference-self | 0.5 | 0.001418 | 0.000878 | 100.0% | 0.0% |
| Reference-self | 0.1 | 0.000365 | 0.000273 | 100.0% | 0.0% |

## Interpretation and budget

No consistent derived advantage over EA or unmutated EC is present. At scale 1.0, both derived actors beat unmutated EC at g50, but not at g150/g300. Relative ordering changes with update scale; smaller steps reduce many negative utilities.
Fixed observation distances (small/large) are 0.0454/0.2945 at g50, 0.0403/0.3777 at g150, and 0.0523/0.4938 at g300. Larger behavioral distance does not imply greater utility: g300 large mutation has fitness 83.20 versus unmutated 365.86 and small 365.95; its TD-error statistic is 0.00876 versus 0.00387 and 0.00424. These three points do not establish correlations.
Reference-self is positive for all 12 scale-1 measurements. All three checkpoints are VALID_COMPLETE/exact; no nonfinite utility or TD-error values. All saved generations follow an injection boundary; this sampling cannot isolate injection timing effects.

Measured runtime: 248.945 + 248.383 + 244.852 = 742.181 seconds (~12.4 minutes). Total extra interactions: 24,000 steps (8,000/checkpoint, including 4 EA + 1 RL + 3 EC-source offline rollouts). Counterfactual EC controls alone: 9,000 steps. No new 300-generation training was needed.

Targeted validation: 88 workflow/artifact tests passed; real integration verified sealed restoration and complete GPU matrices at all three historical states. The runner scripts/diagnose_rl_derived_checkpoint.py performs fail-closed historical configuration/environment checks. Output root: /disk1/zmh_projects/evorl_utility_source_controls_outputs/mechanism/shared_epsilon_historical_20261124_v2

Further work, if requested: preselect additional saved checkpoints or actor directions and noise-matched controls before running; estimate 8,000 offline steps per additional checkpoint with the same first-four EA design. Do not infer a formal multi-seed result from this run.
