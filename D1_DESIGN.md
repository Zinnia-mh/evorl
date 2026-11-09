# D1-v1 mechanism (development only)

Base: D0 tag learner-utility-replay-d0-v1-seed1, commit228a301. No frozen
source or artifact is edited. Independent D1 modules/config/probe/tests only.
Candidate parameters are fixed before results: rho=.3,H=15,W=30,M_min=4,
relative Utility scale s=.01, epsilon1e-8, per-source unaged allocation cap c=.1.
These are mechanism candidates, not a performance-calibrated formal protocol.

Every post-warmup generation measures the current EA population pre-exposure,
using the existing shared reference and true discarded TD3 virtual update.
Warmup remains native EA-only; it is not retrospectively scored. An insufficient
source/reference is invalid, not fabricated. Scores of old sources are never
virtually refreshed. Raw Utility U=L_before-L_after is unchanged.

Dimensionless relative reduction r=U/max(abs(L_before),epsilon) mitigates Critic
loss-scale drift across generations. This is a comparability heuristic across
different contemporaneous references, not proof of stationary historical value.
v=.5+.45*r/(abs(r)+s) lies in [.05,.95]. Negative Utility is less-harmful ranking,
never claimed positive benefit. All-negative scores remain ordered and eligible,
but v<.5 limits their total priority budget; they are not blindly discarded.
Missing/nonfinite diagnostic fields are invalid. No rank within an individual
checkpoint is used for cross-generation priority.

For live EA source i: n_i actual residency, b_i=n_i/N_EA, age=g-measured_g.
I contains valid finite current identities, n_i>=4, 0<=age<=30. Let
N_I=sum_I n_i, h_i=min(c,n_i*v_i/N_I) for i in I, else0.
a_i=h_i*2**(-age/H), S=sum_i a_i. Define
p_i=(1-rho*S)*b_i+rho*a_i.

Equivalently the fixed rho channel is Q_i=(1-S)b_i+a_i, and
p=(1-rho)b+rho*Q. The unused priority allocation returns to Uniform; rho=.3
is the channel budget, while rho*S is the actual non-backoff priority mass.
The stored weighted_source_probability is this normalized Q, not a/S.

S<=1, so probabilities are nonnegative and sum1. The entire EA pool supplies
priority mass rho*S, not just eligible baseline mass. All residents retain at
least(1-rho)*their uniform transition probability. Per-source direct extra
allocation <=rho*c=.03. The cap is applied BEFORE age decay; there is no age
renormalization and no clipping plateau after decay. With fixed membership,
scores and counts, jointly aging by H halves p-b exactly. Aging one source
with all others fixed monotonically lowers its own direct and net increment.
Membership/FIFO changes can legitimately change comparisons; no blanket
monotonic claim is made across changing Replay populations. At age31 the
priority is removed, but Replay payload is not deleted. When all priorities
vanish or rho=0, prepared=false and the actual baseline sampler/key path is
used exactly. RL slots and keys are the existing EA-only overlay semantics.

The bounded transformed value uses the legacy rank_normalized_score array as
D1-specific priority_value storage, while raw_score, token, measured_generation,
validity and version remain authoritative. Config utility_variant=d1-v1 is
mandatory. No state-schema migration is implied: native config/mechanism digest
checks reject D0 checkpoints. D1 checkpoints preserve the transformed value and
all existing learner/replay/optimizer/PRNG fields. Old scores are not recomputed
on resume. The explicit D1 checkpoint wrapper additionally rejects missing/wrong
variant config. D0 remains restorable only with its own fixed code/config.

Historical D1-v1 architecture (superseded by D1-2 below): subclass the D0 workflow, force diagnostic_interval1 for active
D1, reuse its pre-add diagnostic/insertion and JIT TD3 cores. Replace installed
new-source priority values immediately after pre-add scoring/insertion returns,
before any learner update. Override _weighted_rl_update to build the D1
distribution from already recounted actual residency, then call the unchanged
learner update. No monkeypatch/global modification or callback in JIT. The D0
candidate distribution computed by the inherited Host step is discarded before
TD3; this redundant construction is recorded as cost, not hidden. This avoids
editing shared D0 code or cloning its training loop. Frozen diagnostic kernels
and source/reference sampling are unchanged. rho0 normalizes an internal copy
to the valid uniform configuration and disables diagnostics; caller config is
not mutated. No history rescoring occurs in either path.

Validation: tests-first pure probabilities/negative scores/age and window/cap;
small Ring provenance and draws; exact baseline degeneration; real JIT Swimmer
with32 sources,8-step episodes,development seed2027,small network/replay and
three post-warmup generations. Capture actual sampler inputs outside JIT and
reconstruct deterministic final indices using the existing tested function;
check terminal key and optimizer counts. This is reconstruction, not direct
JIT observation. Checkpoint full-state roundtrip and resumed continuation.
Report cold and steady diagnostic/refresh/source/sampler timings separately;
small CPU JIT results do not estimate million-capacity GPU training throughput.
Any real invariant failure is HOLD. No Return threshold, formal seed or training
performance tuning is permitted. D1 is not admitted by D0's formal launcher or
baseline registry until a separate D1 launch protocol/compatibility review.

## Implemented verification

23 targeted tests PASS, plus7 existing D0 Ring regressions in an independent
process. Real CPU JIT probe: seed2027,32 sources per generation,8-step episodes,
512Replay,16TD3 batch,8x8 networks. One native warmup plus3 learning generations:
131 episodes/1048steps,128 Actor and256 Critic calls per learning generation.
Generation3 full-state native checkpoint roundtrip and restored generation4
both PASS. Every new source's raw Utility/token/version/measurement and every
resident count were asserted against real rollout/Replay data. Actual learner
input reconstruction changed4026/3972/3961 EA indices and zero RL indices.

Observed eligible32/64/64, coverage50.79%/100%/100%, priority-only mass
18.47%/17.93%/16.41%, L1 .18179/.01108/.01142. A priority channel's allocated
mass is not the same as distance from Uniform; fully scored sources with
similar relative values naturally have small L1. Synthetic matched-buffer
tests distinguish D0 eligible-mass conservation from D1 full-pool transfer.

CPU diagnostic cold37.82s, subsequent16.88/15.39s. D1 refresh cold.436s and
last warm.00106s; source recount last warm.00148s; discarded inherited D0
distribution last warm.00044s. Independent-key real-state sampler cold
.358-.396s and five-draw warm mean .59-1.08ms. Warm generation count is small;
these are not estimates of million-capacity/full-network GPU throughput.
Virtual diagnostic cost dominates and needs a separate bounded performance
preflight before any long D1 experiment. No performance superiority is claimed.

Artifacts: `/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/d1/mechanism-v1c-seed2027`.
The first config-only probe failed on a Hydra inheritance path before any
rollout; a test-helper import path was also corrected. No probability, index,
age, real residency or checkpoint invariant failed. Earlier logs are retained.

## D1-2 single-distribution refresh

The D1 Host step now mirrors the frozen parent step with exactly one AST
substitution: its refresh call uses the D1-only refresh method. The existing
EA rollout, pre-exposure diagnostic, RL rollout, key split, update counts,
injection and returned state order are unchanged. An AST regression enforces
this structural correspondence. The D0 parent and all ABC paths are untouched.

D1-only d1_residency.recount_and_map preserves the original Ring recount's
allocation order, error status and score invalidation. The existing np.unique
also returns its inverse; final transition-to-source slots are returned with
resident counts. This avoids a second full-Replay mapping and GPU source-table
sort. Recount grouping remains O(C log C), registry work O(S+U), and the final
device distribution/CDF is O(C+S). The standalone calculator has a compact Host
mapping fallback; the actual workflow passes the mapping from its single recount.

The previous discarded D0 refresh and the second D0 alpha-zero baseline build
inside D1 are both absent. Baseline probabilities/counts are computed directly
with the original float32 operations. D1 constructs exactly one final source
probability, transition probability and CDF before calling unchanged TD3.
The formula, precision, cap, age window and checkpoint schema remain unchanged.
The probe now labels actual D1 refresh separately from discarded D0 cost (zero).

Verification artifacts (outside Git):
- phase-d1-2-v1-seed2027: real formal-network GPU JIT, two post-warmup steps;
  32 true pre-exposure scores per step, unchanged TD3 update rate, seed2027.
  The first learner update is also run through frozen 9441b98 code on identical
  inputs; all resulting state/optimizer/metric leaves are bitwise equal.
  Full checkpoint roundtrip followed by the second real step passes.
- phase-d1-2-v2-scale-only-seed2027: corrected well-formed synthetic score_kind
  and eligibility metadata; 1m capacity/table, 1024 sources, mixed ages0..31,
  complete age sweep and old/new exact sampled indices/keys. The initial v1
  synthetic fixture omitted score_kind, so use v2 for eligibility evidence.
  The v1 real diagnostics installed actual Utility score_kind and are unaffected.

35 targeted tests cover original23 D1 tests, original7 Ring tests and5 new
optimization tests. Frozen D0/shared modules and frozen formal files are checked
unchanged. No formal seeds, Return-based selection or long development run.
The short real probe uses formal256x256 networks/pop32/1m Replay/batch256,
but one warmup and256-step episodes to bound interactions. It is not a training
performance comparison. Reconstructed sampler indices are labeled as such;
terminal key and Actor/Critic counts are checked against actual TD3 execution.

Initial measurement: original D1 builder warm~3.535s, optimized~.721s on the
same full synthetic Replay; discarded D0~2.08s per previous preflight is removed.
Combined mapping/recount~.027s. First compile/execution is reported separately.
The second real short generation is84.09s: diagnostic19.91s, source.249s,
D1 refresh3.329s, learner11.459s; two real steps do not establish steady long-run
throughput. GPU live peak~.961GiB, JAX pool~17.73GiB, process peakRSS~5.55GiB,
including old-reference validation/compilation. No long run is launched here.

Final-byte checkpoint validation: normalization from CRLF to LF changes the
existing all-implementation digest, so the supplementary cross-version load
correctly rejected the earlier probe checkpoint. The archived metadata digest
was reproduced exactly by read-only CRLF substitution of the two changed source
files, with config/environment contracts equal. No guard was bypassed. A new
formal-shaped synthetic full-state checkpoint under final LF code restores
bitwise exactly (phase-d1-2-v3-final-bytes-checkpoint). The original same-version
real resume evidence remains valid for its archived implementation bytes; it
is not advertised as cross-version compatible. Fresh-process formal-shaped
sampler compilation plus first draw is1.444s; five warm draws average1.068ms.
