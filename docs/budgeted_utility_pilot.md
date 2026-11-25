# Minimal distributed Utility scoring Pilot

Base404a01b; only schedule/source selection changes. Utility, Adam/TD3,
EA/RL slots, alpha.5,temperature.5,freshness30,min_resident4 remain unchanged.
User spec: distribute32 real source scores over10 generations, prioritize new
unscored sources, minimum tests/Smoke then a preregistered performance Pilot.

Implementation: default identity source-selection hook in the existing
pre-exposure diagnostic. New mixin enables diagnostic on quota-positive
generations and selects3/4 current new sources via independent diagnostic PRNG.
Quota(g)=floor((g-warmup)*32/10)-floor((g-warmup-1)*32/10).
No replay rescoring or new source registry. Existing cohort rank formula stays
unchanged, but3/4-source cohorts necessarily coarsen ranks; this is recorded as
a schedule consequence and limits claims of a pure timing-only effect.
Uniform and original burst D0 remain available on the same code version.

Historical comparisons match every source token. CP50 age0/10/20/30 rho
.370/-.084/.469/.203; CP300 .613/.180/.069/.110 (mean of8 current A evaluations
ranked against original stored scores). Different learner timing, historical
minibatches/references, and partial age30 residency prevent pure age causality.
No further GPU ranking measurements. Nonmonotonic results and age0 mismatch
do not uniquely justify introducing a second stale-refresh mechanism now.

Execution plan:
- Write failing targeted schedule/selection/sampler-boundary tests; implement.
- Commit fixed code; real Brax/Swimmer tiny GPU Smoke, real TD3 input trace.
- Pilot seed3079 fixed before runs, no Return-based seed screening. Three arms:
  A Uniform, D0 corrected burst baseline, S distributed. Global generation50,
  warmup10; pop32, [256,256],source/batch256,maxepisode1000, same initialization.
  Eval every10 generations128episodes; primary Return AUC on g20..50 common
  environment-budget domain. Secondary early AUC g20..30, final Return, mean
  final3 evaluations, actual update/interaction counters, wall and score cost.
  Total perarm1.64M training steps,640k Actor/1.28M Critic updates,512k evaluation
  steps. Expected~1 GPU-hour total from previous run timings; no continuation
  beyond50 because of early ranking. Same scoring budget128 real scores each
  D0/S; source coverage/freshness are measured, improvement not assumed.
- End with descriptive paired comparisons. An AUC improvement over both A/D0
  with no material final-return regression merits further controlled testing;
  otherwise HOLD expansion, revisit hypothesis. No statistical claim from1seed.

Shared-start correction: tiny Smoke found63 warmup Replay floats differed by
at most2.63e-6 across independent JIT warmups; params/optimizer/population/keys
were identical. OriginalfailedSmoke stays frozen in budgeted-scoring-v1.
All comparison arms now fork the same complete native A warmup checkpoint10,
validated read-only with full SHA and config equality except declared arm
controls. This removes numeric warmup differences and saves repeated warmup.
Retry and Pilot outputs use budgeted-scoring-v2. No algorithm change.

## Completed results (seed3079, no further training)

All three arms finishedg50. A/D0/S Return AUC:69736093.37/54524262.08/86848150.69; final Return:60.293/81.768/310.543. S passes the frozen10% AUC signal gate (+24.54%vsA,+59.28%vsD0), but trails A atg20/30/40 and wins through theg50 jump. Early AUC20..30 is27.43%belowA; no stable-return claim.

D0/S each128 real scores. Mean coverage5.395%/6.848%; comparableg20..50 score age9.677/11.232generations, so improved freshness is not established. Post-warmup step time973.94/1707.03seconds(S1.753xD0). Smaller scoring cohorts also change rank granularity despite the same formula. All counters/PRNG/input audits pass;9targeted tests and realSmoke pass. Native completed reports/checkpoints verified. HOLD full formal training; positive preliminary signal merits limited independent-seed performance replication and a same-schedule rank-placebo control. No more experiments launched.

Full remote report/curves: /disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/algorithm/budgeted-scoring-v2/pilot/REPORT.md
