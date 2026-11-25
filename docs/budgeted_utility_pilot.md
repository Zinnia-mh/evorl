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
