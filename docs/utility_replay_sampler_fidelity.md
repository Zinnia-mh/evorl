# D0 sampler probability fidelity repair

Base: 43221c73090c92b29c230b87d7c9a8269ae93432.
Independent branch: codex/d0-sampler-fidelity.
Original seed2047 artifacts remain frozen. The -12.20% Return AUC difference
remains a result of the original implementation.

## Root cause separated experimentally

Each case fixes CP50/CP300 scores, eligibility, source/transition probabilities
and residency. Source L1 is against normalized analytical source probabilities.

| CDF / uniform grid | CP50 L1 | CP300 L1 |
|---|---:|---:|
| Original float32 /23 bits | .0157358832 | .0156460154 |
| Original float32 /52 bits | .0157361079 | .0155981152 |
| Float64 reference /23 bits | .0003091739 | .0003153026 |
| Repaired float64 /52 bits | 3.79297e-9 | 3.93128e-9 |

Float32 cumulative errors aggregate by interleaved source identity. Small
cumulative absolute error did not bound source probability error. The 23-bit
uniform grid adds a separate discretization error: CDF precision alone leaves
approximately 3.1e-4 L1.

## Minimal change

The score, rank, softmax, eligibility, residency and source/transition probability
formulas stay unchanged. Only the sampling representation changes:
- build_sampling_cdf accumulates/normalizes in float64 inside a local
  jax.enable_x64(True) context, carries valid boundaries over zero-probability
  slots, and closes at the last positive slot rather than invalid padding.
- Sampling uses a uniform52-bit grid from two uint32 26-bit words. This avoids
  JAX0.9 nested uint64 random-uniform lowering failure under x64-disabled outer
  JIT. Intervention key splitting is unchanged; EA draws intentionally change.
- Global x64 remains disabled. Models and analytical probabilities stay float32.
  EA/RL slots and the alpha0 native Uniform bypass stay unchanged.
- Initial auxiliary CDF/abstract restore templates use float64. Native restore
  uses a local x64 scope to prevent downcasting; explicit saved model dtypes
  remain unchanged. The full-state hash verifies exact restoration.

CDF normalization only corrects finite-precision total mass (original analytical
source probabilities total1.00000004697 in these states). Historical checkpoints
are audited read-only, not migrated to new training. Existing code/config
checkpoint compatibility guards remain in force.

## Exact historical-state validation

Run scripts/validate_replay_sampling_fidelity.py --output NEW_REMOTE_DIR on CPU.
It defaults to frozen seed2047 inputs, checks full restored state hashes,
requires globally disabled x64, and refuses an output inside frozen inputs.

The auditor partitions the actual JAX scan-based searchsorted decision tree,
counting the2**23/2**52 uniform grid cardinalities exactly, including old CDF
nonmonotonic boundaries. Small full-grid searches independently test the
partition, which also matched the prior full2**23 historical enumeration.
Probability pass/fail does not depend on Monte Carlo or terminal mass alone.

| Metric | CP50 old | CP50 fixed | CP300 old | CP300 fixed |
|---|---:|---:|---:|---:|
| Source L1 error | .0157358832 | 3.79297e-9 | .0156460154 | 3.93128e-9 |
| Source TV error | .0078679416 | 1.89649e-9 | .0078230077 | 1.96564e-9 |
| EA-conditional TV vs Uniform | .0226583373 | .0155664242 | .0227424831 | .0155669402 |
| Noneligible source L1 error | .0140716816 | 2.94e-12 | .0143003714 | 1.14e-10 |
| Invalid candidate rate | 2.38419e-7 | 0 | 1.19209e-7 | 0 |

The original overlay already fell back to valid baseline EA indices on invalid
candidates, so its final invalid-index rate was zero. Fixed candidates themselves
are valid. RL indices remain unchanged.

Target eligible mass:.108359132867. Fixed masses:.108359132870/.108359132981.
Uniform eligible mass:.108359133127. Exact source L1 passes both <1e-7 and
<old_error*1e-4 gates.

## Cost

RTX4090 sampler-only benchmark of original/fixed production draw functions:
20 timed repetitions after compilation/warmup,1000 sequential draws of256.

| State | Original median | Fixed median | Ratio |
|---|---:|---:|---:|
| CP50 | 279.34 ms | 265.48 ms | .9504 |
| CP300 | 266.08 ms | 267.63 ms | 1.0058 |

Ranges overlap; no training-speed improvement is claimed. Isolated full CDF
refresh medians were115.7/116.0ms original versus.553/.545ms fixed on this
compiler/hardware, not complete generation costs. A simpler pure-cumsum
prototype was~.091ms versus~.533ms for the candidate; compiled call graph matters.

At capacity1,000,000 the persistent CDF adds4,000,000 bytes (~3.81MiB).
Compiler temporary storage: fixed CDF builder~11.54MB; fixed draw loop~8.01MB
versus original~4.01MB. These temporary buffers are not resident Replay payload.
No Actor/Critic updates or environment rollouts were benchmarked.

## Verification and scope

Regression coverage: interleaved-source fidelity, zero-mass leading/trailing/
interior slots, RNG boundary values, local precision, alpha0 Uniform equivalence,
RL-index preservation, residency/overwrite, learner sampling and native
checkpoint save/restore.

Final verification: 189 distinct targeted cases passed:
88 source replay,7 source-aware buffer,13 residency,7 ring scores,
56 workflow,13 native checkpoint,5 exact-audit arithmetic.
The workflow file passed48 cases and hit8 setup errors in a shared JAX CPU
compilation context (Failed to materialize symbols: wrapped_xor). Those same
8 cases passed together in a fresh process; none were omitted. An earlier
combined process also aborted compiling the fixture. These unsuccessful attempt
logs are preserved; no unverified claim about their infrastructure root cause
is made. Final processes use eight-core CPU affinity. The genuine checkpoint
dtype mismatch was fixed and its save/restore regression passes.
The auditor's added52-bit float64 boundary case also passes.

The19 frozen top-level artifact checksums passed. Both historical native
states matched their stored full-state SHA after restoration. No original
artifact, configuration or checkpoint was modified.

This enables offline Utility-ranking repeatability experiments; it does not
establish Utility benefit or justify new training. No hyperparameter changes,
D1, new A/D0 training or replacement of the old negative result occurred.

Artifacts:
/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/analysis/d0-sampler-fidelity-v1/
