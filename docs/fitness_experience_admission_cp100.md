# Fitness-Based Experience Admission CP100 Pilot

## Scope and frozen contract

- Base: `codex/d0-budgeted-scoring@6d5b8578fb22f144951327ece9f2c025b052cb2c`
- Branch: `codex/fitness-experience-admission`
- Implementation: `bf85415967a3ef68c3798d672d810447b10476a0`
- Manifest-routing fix: `12d9d2a6cd6ee2956c4d2e458401253266eac5d9`
- Environment: Brax Swimmer; population 32; original Uniform Replay; original GA, TD3 updates, RL experience and periodic RL injection.
- Frozen parent: the complete original-A CP10 state for each seed. Parent state hashes were preserved and verified before forking.
- F-Top16 and F-Random16 both evaluate all 32 EA individuals. Exactly 16 individuals' valid transitions are admitted; all 32 Fitness values continue into the unchanged GA evolution.
- Primary metric was fixed before training: Return AUC over CP20-100, corresponding to 650,000-3,290,000 training environment steps. Evaluations use 128 episodes every 10 generations.

No Utility, TD-error priority, weighted Replay, extra rollout, or R/U/M mechanism is active.

## Performance

| Seed | Arm | AUC20-100 (M) | vs A | vs Random16 | Return100 | First Return >=350 | Training step time |
|---:|---|---:|---:|---:|---:|---|---:|
| 3083 | Frozen A | 294.868 | - | - | 339.53 | gen90 / 2.96M steps / 1282.5 s | 1433.4 s |
| 3083 | F-Random16 | 258.425 | -12.36% | - | 334.03 | not reached | 1440.7 s |
| 3083 | F-Top16 | 600.428 | +103.63% | +132.34% | 367.47 | gen70 / 2.30M steps / 992.8 s | 1443.2 s |
| 3079 | Frozen A | 271.393 | - | - | 121.19 | not reached | 1524.8 s |
| 3079 | F-Random16 | 201.398 | -25.79% | - | 153.05 | not reached | 1438.1 s |
| 3079 | F-Top16 | 256.198 | -5.60% | +27.21% | 176.76 | not reached | 1438.7 s |

F-Top16 beats F-Random16 in both development seeds. This is positive evidence that Fitness conveys useful admission information beyond merely halving EA writes. It does not establish a stable improvement over original Uniform A: seed 3083 is strongly positive, while seed 3079 is slightly negative in AUC despite a higher Return at CP100.

The two seeds are development pilots, not independent evidence of statistical significance. R and U results remain contextual references only and were not used to select or alter this experiment.

## Admission and Replay mechanism

Across each 90-generation arm:

- EA transitions physically evaluated: 2,880,000.
- EA transitions admitted: 1,440,000 (exactly 50%).
- RL transitions admitted normally: 90,000.
- Training interactions after CP10: 2,970,000, identical to A.
- Actor updates: 1,440,000; Critic updates: 2,880,000, identical across arms.
- Final Buffer: 1,000,000 resident transitions, current index 850,000, one wrap after CP10.
- Final composition: 941,000 EA (94.1%) and 59,000 RL (5.9%).
- Resident EA age at CP100: mean 28.91 generations, median 29, maximum 58.

The changed EA/RL composition and longer residence are intrinsic consequences of half-admission and remain a confound when comparing either filtered arm with A. F-Top16 versus F-Random16 controls those effects because their physical write counts and ring schedule are identical.

Fitness-selection checks:

| Seed | Arm | Admitted Fitness mean | Rejected Fitness mean | Mean admitted-rejected | Generations admitted mean higher |
|---:|---|---:|---:|---:|---:|
| 3083 | F-Top16 | 162.22 | -0.67 | +162.89 | 100.0% |
| 3083 | F-Random16 | 102.26 | 94.49 | +7.77 | 60.0% |
| 3079 | F-Top16 | 97.64 | -1.96 | +99.60 | 100.0% |
| 3079 | F-Random16 | 59.64 | 52.41 | +7.23 | 55.6% |

For every Top16 generation, the admitted set exactly matched the stable descending Fitness ranking. Random16 selection is generated from an independent seed/generation PRNG namespace and never reads Fitness.

## Correctness and budget verification

- Targeted unit tests: 3 passed.
- Real Brax GPU smoke: PASS; both arms evaluated all 32 individuals, wrote only 16 EA trajectories, wrote RL experience normally, invoked Utility zero times, and left the GA state and training PRNG identical after the controlled generation.
- Formal journal verification: PASS for all four runs, 90 rows each.
- Sealed checkpoints exist at CP10/25/50/75/100 for every arm.
- Reconstructed CP20 and CP100 learner draws match the original Uniform Replay indices exactly; no weighted overlay is active.
- Final training and GA PRNG keys match between F-Top16 and F-Random16 within each seed, as expected because admission selection uses an independent stream.
- The initial seed-3083 Random16 launch was rejected by the frozen Top16 manifest before output creation or environment interaction. The separate frozen Random16 manifest routing was committed in `12d9d2a`; the failed log is preserved.

Physical experiment budget, excluding the reused CP10 parents:

- Formal training: 11,880,000 environment steps over four arms.
- Formal evaluation: 4,608 episodes / 4,608,000 environment steps.
- GPU smoke: 3,088 environment steps.
- Failed pre-training manifest attempt: zero environment steps.

## Decision

Fitness-based admission has a real source-specific signal relative to random half-admission, but the performance advantage over original A is seed-sensitive. The result does not justify a Top-K sweep or a claim of stable algorithmic improvement. If this route continues, the next defensible step is one pre-registered independent seed at the unchanged Top16 setting, with A and Random16 retained, rather than tuning K on these two seeds.

## Remote artifacts

Root: `/disk1/zmh_projects/evorl_artifacts/fitness_experience_admission/cp100-v1`

- `comparison.json`: exact metrics, contrasts, parent hashes and mechanism summaries.
- `verification.json`: formal correctness checks.
- `learning_curves.csv`: all CP20-100 evaluation points for A, F-Top16 and F-Random16.
- `fitness_admission.csv`: per-generation admitted/rejected Fitness summaries.
- `seed_*/summary-F-*.json`: raw arm summaries.
- `seed_*/runs/*/segments/*`: journals and sealed checkpoints.
- `smoke.json`, `smoke.log`, and training logs: original execution evidence.
