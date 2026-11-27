# Fitness Admission Independent Seed 4099 Replication

## Frozen contract

- Frozen algorithm: `codex/fitness-experience-admission@fb534b3b76f36a252025d90f54013482aa974da1`
- Replication runner only: `05e06b31b4a1d82aea6e72a7095c0fbb0d034c2b`
- Seed: 4099, fixed before training. A content-level search found no prior seed-4099 manifest, configuration, or training log.
- Primary metric: Return AUC over CP20-100, 650,000-3,290,000 training environment steps.
- Evaluation: 128 episodes every 10 generations.
- Arms: original Uniform A, F-Top16, and F-Random16.
- All arms use the same new original-A CP10 state, SHA-256 `49c79e4201aa38bf8b6a7d0beddec4840b5110a7bc72798a34f8fffa140e11b8`.

No algorithm, Top-K, Buffer capacity, Replay sampling, network, GA, TD3 update, or evaluation setting was changed. The new runner only prepares the seed-4099 protocol, creates the shared CP10 state, and records the existing audits.

## Results

| Arm | AUC20-100 (M) | vs A | vs Random16 | Return100 | First Return >=350 | Process wall time |
|---|---:|---:|---:|---:|---|---:|
| A | 777.590 | - | +25.56% | 366.61 | gen50 / 1.64M steps | 1667.5 s |
| F-Random16 | 619.282 | -20.36% | - | 363.34 | gen60 / 1.97M steps | 1546.7 s |
| F-Top16 | 579.997 | -25.41% | -6.34% | 366.73 | gen90 / 2.96M steps | 1545.3 s |

The primary result does not replicate a Fitness-ranking benefit: F-Top16 is 6.34% below the matched F-Random16 control and 25.41% below A. F-Random16 is also 20.36% below A. Final Returns converge near 363-367, but both filtered arms learn more slowly over the fixed common environment-step interval.

The previous development seeds and the independent replication now show:

| Seed | Top16 vs Random16 AUC | Top16 vs A AUC |
|---:|---:|---:|
| 3083 | +132.34% | +103.63% |
| 3079 | +27.21% | -5.60% |
| 4099 | -6.34% | -25.41% |

This pattern is not a stable performance advantage and is not evidence of statistical significance.

## Admission signal and Buffer composition

Fitness ranking itself was implemented correctly:

| Arm | Admitted Fitness mean | Rejected Fitness mean | Mean per-generation difference | Generations admitted mean higher |
|---|---:|---:|---:|---:|
| F-Top16 | 101.90 | -11.08 | +112.98 | 100.0% |
| F-Random16 | 53.67 | 52.24 | +1.44 | 48.9% |

Every F-Top16 generation exactly matched the highest 16 Fitness values. F-Random16 used the frozen independent seed/generation PRNG and did not read Fitness. Thus the negative result is not caused by a failure to apply the intended ranking.

Final Replay composition:

| Arm | EA transitions | RL transitions | EA/RL share | Mean/maximum EA age |
|---|---:|---:|---:|---:|
| A | 969,000 | 31,000 | 96.9% / 3.1% | 14.64 / 30 generations |
| F-Top16 | 941,000 | 59,000 | 94.1% / 5.9% | 28.91 / 58 generations |
| F-Random16 | 941,000 | 59,000 | 94.1% / 5.9% | 28.91 / 58 generations |

Hard half-admission therefore doubles the experience residence horizon and changes the EA/RL composition. Top16 and Random16 control these effects against each other, yet Top16 still does not win in seed4099. A plausible next research question is whether hard removal discards behavioral or transition-level coverage that episode Fitness does not capture, rather than whether a different Top-K can rescue the fixed rule.

## Correctness and budget verification

- Verification: PASS.
- A, Top16 and Random16 have identical initial/final counters and AUC domain.
- Final training and GA PRNG keys match across all three arms.
- Top16 and Random16 both restore the same A CP10 state hash.
- Every filtered arm contains 90 audited generations, all 32 individuals evaluated per generation, exactly 16,000/32,000 EA transitions admitted, and 1,000 RL transitions written.
- Actor/Critic final counts are 1,440,000 / 2,880,000 in all arms.
- CP10/25/50/75/100 are sealed for all arms.
- Original Uniform Replay remains active; no Utility, weighted Replay, or extra rollout is used.

Budget after the shared CP10 state:

- Per arm training: 2,970,000 environment steps.
- Three-arm training total: 8,910,000 environment steps.
- Per arm evaluation: 1,152 episodes.
- Three-arm evaluation total: 3,456 episodes / 3,456,000 environment steps.

## Decision

The predeclared continuation criterion is not met. Fixed F-Top16 should not be expanded to more seeds or a Top-K search. The next useful work is a mechanism analysis of coverage lost by hard episode-Fitness admission, including behavior/state-action diversity, retained transition diversity, and residence-age effects, before proposing a softer or diversity-aware admission rule.

## Remote artifacts

`/disk1/zmh_projects/evorl_artifacts/fitness_experience_admission/seed4099-cp100-v1`

- `preregistration.json`: frozen seed, metric and configuration hashes.
- `shared-parent.json`: exact A CP10 path and state hash.
- `comparison.json`: exact results and contrasts.
- `verification.json`: budget, provenance, checkpoint and PRNG checks.
- `learning_curves.csv`: all CP20-100 evaluations.
- `fitness_admission.csv`: per-generation admitted/rejected Fitness summaries.
- `summary-*.json`, `training-*.log`, and `runs/*`: full summaries, logs and sealed checkpoints.
