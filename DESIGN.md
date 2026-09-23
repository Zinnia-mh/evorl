# EvoRL Learner Utility Diagnostic Design

## Status and scope

This document specifies a diagnostic-only study on branch
`research/learner-utility-diagnostic`, created from EvoRL `main@387d589`.
Its sole question is whether high evolutionary fitness reliably identifies
experience that is useful to the current RL learner:

```text
High evolutionary fitness ?= high learner utility
```

This design does not change Original ERL fitness, population selection,
replay sampling, buffer eviction, the learner's real update, or the EA/RL
interaction. Utility-aware sampling, filtering, selection, multi-objective
EA, and replay algorithms are out of scope. A later branch may consider
them only if this diagnostic earns a separate go decision.

## Definitions

For evolutionary individual `i` in generation `t`:

```text
source_id = gen_<generation_id>_ind_<individual_id>
B_i       = a fixed-size source transition batch
F_i       = original baseline population episode-return fitness
```

`F_i` remains exactly the baseline definition. `source_id` is created before
population experience enters the shared replay buffer. Optional lineage
metadata (`parent_id`, mutation/crossover metadata) may be recorded only if
already directly available from the existing optimizer state; it must never
change EA behavior and is not required for the first smoke run.

For the real TD3 critic parameters `phi`, critic optimizer state `o`, and a
frozen reference batch `B_ref`:

```text
L_before = L_ref(phi, B_ref)
(updates_i, virtual_o_i) = OptaxUpdate(grad_phi L_Q(phi, B_i), clone(o))
phi_i_prime = ApplyUpdates(phi, updates_i)
L_after_i = L_ref(phi_i_prime, B_ref)
U_i = L_before - L_after_i
```

The primary definition is optimizer-faithful: it uses the same current
critic Optax transformation and cloned current critic optimizer state that
the learner would use for one critic update. `virtual_o_i` is discarded.
A simple SGD gradient-step version may be added only later as an ablation;
it is not the principal result.

Positive `U_i` means a virtual update from source `i` lowers loss on the
frozen reference; near-zero means no measurable contribution; negative
means the virtual update raises that loss. This is a learner-conditioned,
observational diagnostic, not a claim about source quality in general.

## Data flow and source provenance

```text
population actor parameters
  -> per-individual EC rollout trajectories [individual, time, env, ...]
  -> source_id plus raw per-individual trajectory
  -> fixed-size B_i for TD error and virtual update
  -> existing flatten-and-add path into shared replay (unchanged)

current replay state
  -> deterministic diagnostic key
  -> one frozen B_ref per diagnostic checkpoint
  -> common L_before and per-source L_after_i
  -> one utility record per source
```

The source grouping is established before flattening or replay mixing. The
diagnostic does not attempt to reconstruct provenance from shared replay.
The existing trajectory-to-replay conversion remains the path used by real
training; diagnostic extraction is read-only with respect to that output.

## Batch policies

Every source used in an individual comparison has exactly
`source_batch_size = M` transitions. A source with fewer than `M` valid
transitions is recorded as ineligible with its reason and is excluded from
correlation/rank statistics for that checkpoint. The first version does not
use replacement sampling because it would make small-source utility values
less directly comparable. `source_batch_size`, eligible-source count, and
excluded-source count are recorded.

The exact same `B_i` is used for both the TD-error statistics and the
virtual-update utility of source `i`:

```text
B_i^TD = B_i^Utility
```

TD-error output is `td_error_mean`, `td_error_median`, and `td_error_std`.
It is a comparison proxy only and must not alter replay priorities.

## Reference settings

### Setting A: Current Learner Reference

At a diagnostic checkpoint, sample one `B_ref_current` from the current
learner replay distribution using a diagnostic-local deterministic key.
Every eligible source at that checkpoint uses this identical reference
batch. This is the first implementation and primary result: it asks whether
source experience improves the distribution the learner is currently
fitting.

### Setting B: Anchor Reference

The later setting freezes `B_ref_anchor` across several diagnostic
checkpoints. It separates a change in source utility from variation caused
by resampling a reference batch. It is specified now but intentionally not
required for the first one-environment, one-seed smoke.

## Loss fidelity and virtual-update isolation

Diagnostic critic loss must use the learner's real critic-loss definition,
including TD3 target critic, next action, reward, discount, termination,
and all baseline target construction. The diagnostic must call the same
loss implementation as the real learner, or a shared pure loss function
extracted from it. A separately approximated squared-TD-error loss is not
permitted for utility.

The virtual update operates on functional/cloned inputs only. It must not
write back real critic parameters, critic optimizer state, target critic,
actor parameters, target actor, replay buffer, EC state, workflow metrics,
or baseline PRNG state. Diagnostic-local randomness is derived without
splitting and writing back the baseline key, for example by a deterministic
fold-in of immutable base seed, generation/checkpoint index, and a fixed
diagnostic tag.

## Diagnostic scheduling and records

The initial cadence is one checkpoint every `K = 5` generations, subject to
smoke profiling. It is deliberately not an environment-step calculation.
Each record contains:

```text
seed, env, training_step, generation, individual_id, source_id,
fitness, episode_return_mean, episode_return_std,
td_error_mean, td_error_median, td_error_std,
utility, loss_ref_before, loss_ref_after,
source_batch_size, reference_batch_size, critic_lr,
reference_setting, checkpoint, timestamp
```

The output is append-only diagnostic CSV data. Raw trajectories, replay
contents, checkpoints, and large logs remain unversioned validation
artifacts; only small report/config summaries may enter Git.

## Analysis plan

After the smoke pipeline is correct, compute per checkpoint and aggregated
statistics:

1. Pearson and Spearman `corr(F_i, U_i)`.
2. Pearson and Spearman `corr(TD_i, U_i)` using the same `B_i`.
3. Population-relative fitness/utility quadrants: high-F/high-U,
   high-F/low-U, low-F/high-U, low-F/low-U.
4. Top-20-percent fitness versus top-20-percent utility overlap.
5. Early/middle/late temporal summaries after sufficient budget exists.

Visualization priority follows correctness: CSV and statistics precede
figures. The smoke requires only the core fitness-versus-utility scatter;
TD/utility, rank, stage-distribution, and quadrant figures are added after
pipeline correctness is established.

## Correctness gates

Implementation cannot advance from tests to smoke until all gates pass:

1. Virtual update leaves real critic parameters, critic optimizer, target
   critic, actor, target actor, replay state, EA state, and baseline PRNG
   state exactly unchanged.
2. Diagnostic disabled is exact-equivalent to Original ERL under identical
   seed/config/budget.
3. Every source in one diagnostic checkpoint uses exactly the same frozen
   reference batch.
4. `source_id -> B_i` is established before shared-replay mixing.
5. Every compared source uses the same `source_batch_size`; ineligible
   short sources are explicitly recorded and excluded rather than compared.
6. TD error and utility for a source use the same `B_i`.
7. Diagnostic utility and real learner use the same critic-loss and target
   construction implementation.

## Staged execution and go/no-go

```text
tests first
  -> source tracking and functional utility instrumentation
  -> one environment x one seed smoke
  -> CSV integrity and core fitness/utility scatter
  -> at least three seeds
  -> only then consider multiple environments
```

The first go/no-go question is whether `Fitness_i` is approximately aligned
with `U_i`, based on data rather than an assumed conclusion. Evidence for
high-fitness/low-or-negative-utility and medium-or-low-fitness/high-utility
sources would motivate a separately approved replay-research branch.
Conversely, near-perfect rank/correlation agreement across seeds and
environments, with essentially no high-fitness/negative-utility sources,
means this direction must be reconsidered or closed before any replay
algorithm is implemented.
