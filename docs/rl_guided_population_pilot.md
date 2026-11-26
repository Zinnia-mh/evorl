# Online RL-guided GA population M: fixed first Pilot

User goal: actual multi-generation evolutionary intervention, not throwaway
diagnostic policies. All work and artifacts remain remote under/disk1/zmh_projects.

## Frozen algorithm and experiment
- Base f1cdb57, stable Uniform A code; no evorl/ changes or old workflow merge.
- Seed3083 chosen to reuse complete frozen Uniform A CP50, not selected by M results.
- Restore full learner/targets/Adam/Replay/GA/PRNG/runtime, including original eval prefix.
- Before rolloutg60,70,...150 replace tail8 of32 with4 Gaussian directions
  actor+/-0.005*epsilon from the current pre-step RL Actor. First24, including
  elites, unchanged at replacement. Independent folded mutation keys.
- Real32-policy Fitness evaluation, ordinary tournament/elitism/mutation,
  next-generation breeding, original1-actor RL injection every10g retained.
- Uniform Replay, same TD3,3.30M new training and1.28M new evaluation steps.
  Logical final counters4.94M steps,2.24M Actor/4.48M Critic updates.
- Every10g evaluate128episodes. Primary AUC60..150(1.97M..4.94M steps).
  Native20..150 AUC auxiliary, sharing frozen20..50 prefix with A.
- First scheduled meanReturn>=350 training steps/time, final Return and cost
  are secondary. No interim tuning, seed replacement, or early stopping.
- Comparable total time charges measured A prefix throughCP50 plus real M steps;
  also retain M physical process wall and core versus observation overhead.

## Minimal implementation
scripts/rl_guided_population.py: paired parameter replacement, originalGA
delegation plus exact parent-map metadata, host pedigree, actual rollout receipt.
scripts/run_rl_guided_population_pilot.py: full-state fork/recovery, immutable
preregistration, real GPU Smoke, Pilot, raw per-generation genealogy/timing.
scripts/diagnose_rl_guided_population.py: readonly actual TD3/Optax Utility.
scripts/analyze_rl_guided_population.py: fixed CPU metrics and curves.
tests/test_rl_guided_population.py:9 focused tests.

Pedigree means any M ancestor, not a parameter-fraction estimate. Original-A
means baseline EA, which retains its normal single-actor RL injection. The
standard_rl pedigree flag tracks only known ancestry since theCP50 fork.

## Utility and diversity contract
Offline CP50/100/150,256 transitions per current-generation resident source;
score32 A-EA,32 M-EA,M RL-own and Replay-random in the same frozen M learner,
target networks, real Critic Adam state and3 common current-M-Replay references.
Real virtual TD3 Critic update at step1, discarded; no Replay/PRNG write.
Explicit post-update checkpoint phase, not D0 pre-exposure Utility. A data
are same-generation cross-run counterfactual sources; never inserted in M Replay.
Source mean/median/positive fraction aggregate3references per distinct source;
references and shared-ancestry individuals are not independent training seeds.
RL-own retains exploration noise; EA policies are deterministic.
Trajectory action distance is source-conditioned; fixed common CP50 observations
measure whole-population behavioral geometry before/after injection and at CPs.
If no no-M ancestry remains, report absent rather than inventing that control.

## Verification and provenance
9 targeted tests passed; actual tiny Brax GPU Smoke pop32,episode8 passed.
Exact injected parameter SHA at collector input, actual collector obs/actions
equal Replay bitwise, correct8 source tokens and fixed interaction budget.
CPU optimizer tests preserve originalGA outputs/keys and prove high-Fitness M
parents can persist and breed. The tiny untrained-actor Smoke selected no M
offspring; preserve that negative observation, do not guarantee survival.
Off-mode GPU repeated baseline has float variation up to6.8e-5; exact-byte
neural-state test was inappropriate. Explicit1e-4 float tolerance, exactGA state,
keys/origin/token checks, unchanged TD3 implementation; failure logs retained.
Independent Actor forward recomputation also varies with GPU batching; use the
stronger exact collector-input and actual-trajectory-to-Replay receipts.

The original implementation-only plan usedCP10/start40. Before any M performance
run it was changed to reliable completeCP50/start60 to reduce rerunning and fork
closer to intervention. Primary changed prospectively to60..150, not after results.
Original plan and revision evidence remain in the new remote artifact directory.

LCERL motivation only: late evolution and opposite proximal perturbations;
this is not its adaptive operator, clustering archive or full architecture.
Primary author-group description:
https://www.translab.top/index.php/en/2026/01/20/%E3%80%90ep39-%C2%B7-research-highlights%E3%80%91-evolutionary-reinforcement-learning-with-late-start-evolution-and-clustering-archive-published-in-ieee-tevc/
