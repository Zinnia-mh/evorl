# R: persistent random source priorities without Learner Utility

Basef1cdb57: verified S/P and corrected float64CDF/52-bit sampler.
No M/M-Late/GA changes. Independent branch and remote output.
All artifacts under/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/algorithm/random-source-priority-v1.

R uses exactly P's3/4 quota, independent source-selection key and stable
cohort permutation. Choose among current32EA sources; preserve original IDs.
Validate enough real transitions (minimum256), finite rollout payload/Fitness,
and pre-existing Replay minimum. Use equal-spaced normalized rank grid over
valid selected IDs, then existing P permutation for effective random ranks.
Single valid source=.5. Same residency, freshness30,minresident4,alpha.5,temp.5,
version and EA/RL slot behavior. Retain corrected sampler and TD3 unchanged.
Ranks persist over source lifetime; same partial-cohort remapping as P.

No true Utility, discarded Critic update, diagnostic context, source minibatch
or reference minibatch. These entrypoints raise if reached. The normal actual
TD3 optimizer remains active as required. Baseline jitted EC collector inserts
data once; source-priority/provenance metadata updated without another rollout.

Legacy config weighted-arm flagdiagnostic_enabled=true and numeric tag3 are
only compatibility with the frozen source-table/checkpoint ABI; they do not
enable Utility in R. Every record/manifest labels synthetic_random_cohort_rank,
with zero realUtility/virtualUpdate/reference/sourceBatch counters. Never
interpret raw_score field in these R checkpoints as a measured Utility.

P and R are not claimed identical: P can have realUtility ties or nonfinite
score rejection; R uses full finite grid and finite payload gate. Same marginal
random-rank distribution assumes no ties; actual rank-to-source realizations
differ. P used raw diagnostic EC/batch path; R uses baseline jitted collector.
Environment/TD3/evaluation budgets remain equal; no assumption of equal Return.

Frozen before training:
- Seed3083 complete A CP10->150; no A/P retraining.
- Primary AUC20..150 over650k..4.94M steps; eval every10g x128episodes.
- First scheduled Return>=350 steps/time; no threshold imputation.
- Post-CP10 synchronized step wall is main compute cost (compilation included,
  evaluation/checkpoint I/O excluded). Shared warmup excluded for all arms;
  seed3079 historical warmup time is not separately measured and is not invented.
- Positive3083 signalR/A AUC>=5% triggers fixed3079 second development seed.
  Otherwise no3079 training and no alternative seed search.
- Descriptive near-A band+/-5%, near-P band+/-10%; meaningful P cost reduction25%.
  These are practical decision bands, not statistical significance/equivalence.
- Physical per-seed new4.62M training+1.792M eval; native full CP10 reused.
  Actual Actor/Critic and eval budgets equal A/P; synthetic priorities448.
- No interim tuning or additional algorithms.

Tests: exact P source selection/quota, rank persistence/residency/freshness,
real corrected sampler EA changes/RL preservation. Tiny realBrax GPU Smoke
with Utility/source minibatch functions tripwired. Actual TD3 traces at20/150
during full run; no repeat of historical189-test audit.

Implementation files:
scripts/random_source_priority.py: direct synthetic updates and original sampler.
scripts/run_random_source_priority.py: native forks, logging, fixed seed gate.
scripts/analyze_random_source_priority.py: CPU equal-horizon cost/Return comparison.
tests/test_random_source_priority.py.
