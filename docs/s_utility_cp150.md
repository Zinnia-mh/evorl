# S duration and ranking placebo: fixed continuation

Parent Pilot is frozen:seed3079, A/S atCP50, shared A warmupCP10.
New independent outputs:algorithm/budgeted-scoring-cp150-v1.

Only new method:P permutes rank/source labels within each eligible scoring
cohort. True Utility is still measured at the same3/4-per-generation quota,
32 per10 generations. Cohort rank multiset, eligibility, residency and all
alpha/temperature/freshness/TD3/EA-RL settings remain unchanged. Permutations
use independent NumPy PCG64 seed3079 plus a fixed tag/cohort generation, so
they remain stable while that eligible cohort membership stays unchanged.
Canonical raw scores/ranks remain true; only distribution construction sees
the permuted table. Uniform permutations may include identity; moved ranks
are logged. No baseline PRNG is consumed.

Original native checkpoint reader validates the original configuration,
mechanism, environment, file seals and full-state SHA. New run contracts permit
only the explicit total-episode extension, output paths and arm controls.
A/S retain CP50 state, Optax/Replay/PRNG and the exact original four evaluation
records; P restores the same CP10 state. No old directory is modified.

Seed3079, final generation150, totalepisodes4940,4.94M logical training steps,
2.24M Actor/4.48M Critic updates and14 evaluations20..150 x128 episodes per arm.
Primary:Return AUC over the common650000..4940000 training-step domain.
Secondary:AUC50..150 (1640000..4940000),mean last5 evaluations,final Return,
post-warmup/continuation cost. Old20..50 numbers remain separately frozen.
Fresh work:11.22M training interactions+4.352M evaluation interactions.
S/P total448 true source scores (S128 old+320 new;P448 new).
No decisions or hyperparameter changes at60/80/100. A positive full and
post50 AUC contrast against A and P plus compatible last5 Return supports
independent-seed replication; this single seed cannot establish significance.
No D0 retraining, D1, RL-derived or sampler changes.

Implementation:one new experiment runner using existing native fork/resume
and existing rank_transform hook, new targeted tests only; no changes under
evorl/. This intentionally preserves the parent mechanism digest.

Recovery: save a sealed child start checkpoint before the first new step;
canonicalize repeated-generation journals after rollback and record discarded
score calls separately. These safeguards do not change training or P.
