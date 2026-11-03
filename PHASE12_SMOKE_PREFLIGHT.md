# Phase 12-A Smoke preflight

No four-arm GPU Smoke or matched Pilot was run during tool development.
Calibration remains the immutable Phase 11-B selection (alpha=.5, replay
temperature=.5, W=30, M_min=4, diagnostic interval=10).

## Observer contract

Production Smoke adds no callback or auxiliary result to the Critic/Actor JIT.
Host overrides retain immutable sampling inputs from the actual learner call,
then invoke the original learner unchanged. After the step completes, a
separate pure executable replays its sampling key schedule against those exact
Replay and distribution arrays. Reconstructed keys are checked and discarded.
They are never fed back into training. Runtime checks bind the reconstruction
to actual actor/critic optimizer counters and the actual intervention terminal
key. The baseline step key must follow the original four-way split.

Schema 2 marks index-derived fields as `deterministic_reconstruction`, including
the legacy-named `observed_batches`, origin-slot counts, replacement counts and
index hashes. They are not direct callback observations. Environment counters,
optimizer deltas, scores, distributions and reference fingerprints are read
from the real run. Independent-key sampling-probe time is not the duration of
sampling inside the training kernel. Reconstruction time is separate from
training time. The W+1 neutral probe is counterfactual, not observed expiry.

Tests compare reconstructed indices with direct sampler calls and test-only
callbacks from an actual learner invocation. The diagnostic learner outputs
are discarded. Separate state-equality tests use production observation and
require bitwise equality after every real generation, including a valid,
non-neutral Utility checkpoint and a separate neutral case. No generation is
artificially advanced in these acceptance tests.

The legacy callback failure was isolated on CPU: identical learner inputs and
indices, but the ordered callback changed two optimizer first-moment elements
by at most 5.960464477539063e-08. Independent same-implementation and callback-off
controls were exact. GPU frozen-input controls were exact with and without the
callback. Thus the observed CPU failure is not evidence of GPU nondeterminism.
Isolation logs and HLO are retained in `.codex_validation/utility_replay_pilot/`.

## Residency performance evidence

The hot refresh uses `source_residency_batch.py`: grouped Host counts plus
deterministic registry allocation, O(C log C + S + U), replacing O(C*S)
transition-by-registry scans. The generic registry capacity remains C; M_min
does not bound how many one-transition sources may still be resident.
One-million-capacity checks, exact ring/metadata/sampling comparisons and
resource records are retained under
`.codex_validation/utility_replay_pilot/phase12_capacity_profile_v2/` and the
associated capacity regression logs. The callback fix does not change those
source/statistics files; their prior passing evidence is reused.

## Four-arm acceptance

The user-approved `matched-settings-and-budget-v1` policy requires common
resolved initialization settings, seed, networks, environment, EA/RL mechanisms,
training/evaluation budget and population-operation counters. Only the four
preregistered arm fields (arm, score_kind, alpha, diagnostic_enabled) are excluded
from the shared resolved-settings fingerprint. The runner fingerprints its
actual workflow config, and output acceptance binds the saved config and report
to the immutable manifest. Full post-warmup state fingerprints remain recorded
for provenance; they are not cross-process equality gates.

Phase 12-C established exact equality of 109 pre-warmup Pilot state leaves in
A1/A2/B1, and equality of all 105 shared initial leaves with Original ERL.
Original ERL and isolated Brax reset/step reproduced GPU non-repeatability on
identical inputs, including one identical compiled rollout executable. Evidence
is retained under the external diagnostics/phase12c-first-divergence-v1-head6b2d0ee
directory. The exact native primitive remains unresolved; no baseline, simulator
or GPU configuration was changed to suppress it. No common-checkpoint protocol
is introduced. These known simulator differences do not waive any PRNG, sampling,
data-validity or actual-budget gate.
Within each run, RL indices/origins, optimizer counts and original PRNG
transitions remain strict. Cross-arm post-learning Replay payloads, realized
EA/RL slot counts and draw hashes may legitimately differ through feedback;
they are not equality gates. A requires zero intervention; B/C/D require an
observable non-neutral intervention at generation 20, with valid resident scores,
a prepared distribution, eligible sources and changed EA indices. Reconstruction
is bound to the actual TD3 learner inputs; direct-index equivalence tests and
actual optimizer-counter/key checks remain required. Each arm is validated before
the next starts; single-arm acceptance never claims four-arm budget equality.

ESS acceptance uses Host float64 normalized variance:
mean(p)^2 / (mean(p)^2 + mean((p-mean(p))^2)), over resident sources.
This equals normalized ESS fraction without clipping and does not modify p or
CDF. The original float32 ESS, probability masses, squared mass and CDF interval
error remain in every row. `effective_eligible_mass` is explicitly alpha times
eligible_baseline_mass; it is not the fraction of changed indices. EA replacement
counts include resampling within the neutral component and must not be presented
as the magnitude of the Utility probability tilt.

CDF audit scope follows actual use: uniform A and unprepared distributions do
not feed a CDF into training. Their raw CDF anomalies are retained as diagnostics.
An active weighted CDF must be finite/nonnegative, cover the upper boundary,
and agree with the monotone float64 cumulative reference from its nonnegative
transition probabilities. The error allowance is the dtype/length gamma_n
forward-summation bound, capped at the existing 1e-5 probability tolerance.
Raw monotonicity, negative increments, reference error and the bound are recorded;
small dtype-consistent reversals are not silently converted into exact monotonicity.
Production CDF, probabilities, RNG and sampler code remain unchanged. Anomalies
save raw arrays plus runtime counters under audit/cdf_generation_<g>.npz/json;
initialization SHA/counters are saved before the first training step.
The legacy cdf_non_ea_interval_mass64 field is a signed raw-CDF difference
statistic, not an empirical sampling probability when rounding dips occur.
Its bounded signed roundoff is recorded; actual EA/RL validity is enforced by
the reconstructed learner indices and unchanged runtime origin checks.

The short budget keeps the formal one-million Replay capacity, population,
networks, batch and scoring parameters. Seed 1, Swimmer horizon 1000, warmup 10
and 12 training generations give 320 initial plus 396 training episodes (716
total), 396000 training environment steps, 192 actor and 384 critic optimizer
updates. All arms use rl_updates_frac=.001, eval interval 10, 2 evaluation
environments and 2 episodes; evaluation checkpoints are 20 and 22. These are
Smoke checks, not a performance comparison or calibration result.

## Commands (actual GPU Smoke requires separate authorization)

Run inside `evorl-utility:review`, after committing and checking a clean tree:

```bash
cd /disk1/zmh_projects/evorl
PYTHON_BIN=/home/admin001/anaconda3/envs/evorl-conflict/bin/python
FROZEN=/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/calibration/phase11b-selected-v2-head-dbcb4f9-seed104729
SMOKE=/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/smoke/swimmer/seed_1/phase12b-v3-matched-settings-head$(git rev-parse --short=8 HEAD)
env -u LD_LIBRARY_PATH "$PYTHON_BIN" scripts/run_utility_replay_smoke.py prepare \
  --frozen "$FROZEN" --directory "$SMOKE" --seed 1
```

Preparation creates an immutable manifest, full arm configs and checksums. It
does not call workflow setup/step or launch training. Existing directories are rejected.
When separately authorized, run arms sequentially, stop on any failure:

```bash
set -e
mkdir -p "$SMOKE/logs"
for arm in A B C D; do
  env -u LD_LIBRARY_PATH "$PYTHON_BIN" scripts/run_utility_replay_smoke.py run-arm \
    --directory "$SMOKE" --arm "$arm" > "$SMOKE/logs/$arm.log" 2>&1
  env -u LD_LIBRARY_PATH "$PYTHON_BIN" scripts/run_utility_replay_smoke.py validate \
    --directory "$SMOKE" --arm "$arm"
done
env -u LD_LIBRARY_PATH "$PYTHON_BIN" scripts/run_utility_replay_smoke.py validate \
  --directory "$SMOKE"
```

The formal launch validates both the shared Smoke preflight and existing arm
manifest, checks committed code identity, and rejects an existing arm output.
Results live under `<SMOKE>/<A|B|C|D>/`; aggregate acceptance is
`<SMOKE>/aggregate/smoke_validation.json`. No automatic rerun, threshold
relaxation or return-based tuning is allowed.
