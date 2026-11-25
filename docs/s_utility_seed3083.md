# Independent fixed seed3083 A/S/P replication

Reference commit ba5c570. Existing seed3079 runs remain frozen.
Seed3083 fixed before outcomes;170 prior registry/config files had no exact use.
A/S/P all train from the same complete new A warmupCP10 toCP150.
No changes to evorl/, Utility, TD3, source choice, quotas, probabilities,
alpha.5,temperature.5,freshness30 or existing P cohort permutation rule.
P uses the same independent tag+training-seed+cohort construction with3083.

Fixed primary:Return AUC g20..150, training steps650000..4940000.
Additional fixed metrics:first scheduled128-episode evaluation with mean
Return>=350 (no consecutive-hit rule), its training-step counter and actual
cumulative training time; S/P AUC g70..150 (2300000..4940000); final Return;
total time. Missing threshold hits are censored/not reached, never imputed.

Comparable training time=measured common warmup/init seconds+sum of synchronous
workflow.step wall times. Includes compilation, rollout and scoring; excludes
evaluation, checkpoint I/O and idle monitoring. Also retain process wall time
and per-evaluation elapsed wall time. S/P are charged the shared measured
warmup cost, not a free initialization. No step-count-to-time conversion.

Swimmer1000,pop32,[256,256],batch/source256;32 real scores per10 generations,
3/4 per generation; S/P448 scores,4.94M logical training steps,2.24M Actor/
4.48M Critic updates,eval every10g x128episodes (14 evaluations).
Actual physical cost:new sharedwarmup320k+3x4.62M training=14.18M;
3x1.792M evaluation=5.376M. Expected about4 GPU-hours on4090.

No tuning or early stopping at intermediate Returns.
Descriptive route rule frozen before training: both S/P above A and S/P
primary AUC gain>=10% is a clear additional-speed signal(A); both above A
and absolute S/P gap<10% is a near-result(B); both fail to beat A is C.
Mixed outcomes are reported as mixed rather than forced into these cases.
These cutoffs are practical labels, not significance or equivalence tests.
CaseA may justify further ranking work; B prioritizes questioning Utility
necessity/lower-cost sampling; C stops scaling. Do not launch those next
experiments automatically.
