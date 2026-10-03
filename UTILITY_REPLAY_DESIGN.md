# Learner-Utility-Aware Source Replay Pilot Design

## Document status

**Design only. No replay algorithm is implemented by this document.**

Source milestone: annotated tag `learner-utility-diagnostic-v1` at commit `df954935334e201f3e2759b848d5be2c177833f7`.

Target branch: `research/learner-utility-replay-pilot`.

The diagnostic established that evolutionary fitness contains some information about learner utility but is only weakly rank-aligned with it and is not an interchangeable source-quality criterion. It did **not** establish that utility-aware replay improves RL performance.

## 1. Research question

> Given that evolutionary fitness is not equivalent to learner-side experience utility, can learner-utility-aware source replay improve the RL learner while preserving the original ERL interaction budget?

The first intervention changes only **which EA source is replayed more often**. It does not change which experience is generated, how population fitness is calculated, how EA selection operates, or how many learner updates occur.

## 2. Scope and non-goals

### Fixed Original ERL behavior

The following remain identical across all pilot arms:

- environment and environment interaction budget;
- EA population size, mutation, crossover, selection, and elitism;
- RL learner architecture, targets, optimizer, learning rate, and update count;
- replay capacity and learner batch size;
- EA rollout and RL rollout counts;
- RL injection schedule and EA/RL interaction mechanism;
- evaluation schedule and seed;
- total training budget.

### The only intervention

Within the EA-origin portion of replay exposure, the sampler may redistribute probability among EA sources using one source-level score:

```text
uniform, fitness, TD error, or learner utility
```

### Explicitly prohibited in V1

- buffer eviction policies;
- deletion of experience;
- changing population fitness;
- changing EA selection;
- multi-objective EA;
- additional environment interaction;
- changing the number or size of learner updates;
- changing the aggregate EA-vs-RL replay exposure;
- using intervention outcomes to tune thresholds on formal evaluation data.

## 3. Design principles

1. **One causal variable:** only EA-source exposure changes.
2. **Baseline preservation:** `alpha = 0` must reduce to Original ERL sampling, not merely approximate it statistically.
3. **No source starvation:** every resident EA source retains a uniform component.
4. **Learner conditioning is temporary:** utility scores become stale and decay to neutral behavior.
5. **No circular evaluation:** data used to estimate utility cannot be the evidence used to claim algorithmic success.
6. **Common instrumentation:** all four arms produce the same audit fields and, where feasible, execute the same read-only diagnostic instrumentation.
7. **Pre-registration:** weighting version and pilot criteria are fixed before matched runs begin.

## 4. Replay data model

### 4.1 Preserve transition payloads

The baseline transition payload remains unchanged. V1 should use a pilot-specific replay state with parallel sidecar metadata rather than adding fields to every shared transition type used by the rest of EvoRL.

Conceptual state:

```text
SourceAwareReplayBufferState:
    data                       # unchanged ReplayBuffer transition pytree
    current_index              # unchanged ring-buffer pointer
    buffer_size                # unchanged valid size
    origin_type[capacity]      # EA or RL
    source_token[capacity]     # stable token for EA; sentinel for RL
    source_generation[capacity]
    score_version[capacity]    # score-table version at insertion/update
```

The sidecars use the same ring-buffer indices as `data`. Every add operation writes payload and metadata with the same mask and indices, so overwrites cannot separate a transition from its provenance.

### 4.2 Stable source identity

An EA source token identifies a population member at a specific generation:

```text
source_token = encode(generation, individual_id)
```

It is not interpreted as a persistent lineage across generations. All valid transitions produced by that individual during that rollout carry the same token. RL-origin transitions use `origin_type = RL` and a reserved source sentinel; they never receive EA-source weights.

### 4.3 Source score table

A bounded score table is keyed by `source_token` and stores:

```text
raw_score
rank_normalized_score
measured_generation
score_kind
validity
```

The table needs entries only for sources whose transitions may still be resident. Ring-buffer overwrite and the freshness rule bound the active set. V1 must not delete transitions when a score expires; expiry changes the score to neutral sampling behavior.

## 5. Provenance insertion path

### EA-origin experience

The population trajectory is kept in its pre-flattened `[population, time, env, ...]` form long enough to create exact `source_token` metadata. Payload and metadata are flattened with the same transformation and mask, then inserted together. Source ownership is never inferred from mixed replay afterward.

### RL-origin experience

RL rollout insertion is payload-identical to baseline. Sidecar metadata records only `origin_type = RL` plus the sentinel source token. The sampler never applies fitness, TD, or utility weighting to RL-origin transitions.

### Warmup

Warmup uses the same provenance rules. If source scores are unavailable or the buffer is not yet diagnosable, every EA source is neutral and sampling is uniform.

## 6. Sampling architecture: preserve the EA-vs-RL mixture

The central anti-confounding rule is implemented with a **baseline-draw overlay**:

1. Use the unchanged baseline replay sampler to draw the full batch of transition indices uniformly.
2. Preserve every position whose baseline index is RL-origin.
3. Preserve the number and positions of EA-origin draws exactly.
4. For each EA position only, conditionally replace the baseline EA index with an index sampled from the resident EA set according to the selected source weighting method.
5. Gather transition payloads from the resulting indices and run the unchanged learner update.

Consequences:

- each realized batch has exactly the same EA/RL slot count as its baseline draw;
- RL-origin samples remain baseline-uniform and untouched;
- only conditional selection among EA sources changes;
- environment interaction, learner update count, and batch size are identical;
- `alpha = 0` can return the original baseline indices directly, enabling bitwise equivalence.

This overlay is preferable to independently choosing an EA/RL mixture because the latter would introduce a second intervention. It is also preferable to globally weighting all transitions because that would change RL-origin exposure.

## 7. Source weighting

### 7.1 Rank normalization

Raw signed utility is not a probability and must not be used directly. At a measurement checkpoint, valid EA sources are ranked within that checkpoint. Ties receive a deterministic average rank. Rank is normalized to a bounded score, for example:

```text
r_i in [0, 1]
```

Higher rank means higher priority for the method being tested. The same normalization procedure is used for fitness, TD error, and utility; only the underlying raw score differs.

### 7.2 Bounded mixture with a uniform floor

Let `q_i` be the baseline conditional probability mass of source `i` among resident EA transitions. Because sources can contribute different valid trajectory lengths, `q_i` is occupancy-proportional rather than assumed to be exactly `1/N`. Let:

```text
s_i = softmax(r_i / tau)
P(source=i | EA) = (1 - alpha) * q_i + alpha * s_i
```

For equal-size sources, this reduces to the intended form:

```text
P(i) = (1-alpha)/N + alpha * softmax(rank_normalized(score_i)/tau)
```

Within a selected source, one resident transition is sampled uniformly. The `(1-alpha) * q_i` component is the starvation-prevention floor and preserves baseline occupancy effects. V1 does not lock numerical values of `alpha` or `tau`; they must be selected and frozen during a pre-pilot calibration that does not use formal evaluation outcomes.

### 7.3 Neutral and invalid scores

- missing, non-finite, insufficient-data, or expired scores are neutral;
- if every resident EA score is neutral, conditional EA sampling is baseline-uniform;
- no invalid score can become a very large positive or negative logit;
- probabilities are checked for finiteness, normalization, and a nonzero baseline component.

## 8. Four matched pilot arms

### A. Original ERL / uniform

Use the unchanged baseline replay indices. Source metadata and read-only logging may be enabled, but no source weighting is applied. This is the primary control.

### B. Fitness-aware source replay

Use population episodic fitness recorded for the source at its generation. Higher within-checkpoint fitness rank receives higher conditional EA-source weight. Fitness is not recomputed later.

### C. TD-error-aware source replay

Use the source-level TD-error statistic from the common pre-exposure measurement pipeline. The conventional comparator prioritizes higher within-checkpoint TD-error rank. Its direction is pre-registered even though the diagnostic observed a negative TD–utility association.

### D. Learner-Utility-aware source replay

Use the optimizer-faithful pre-exposure learner utility:

```text
U_i = L_ref_before - L_ref_after
```

Higher within-checkpoint utility rank receives higher conditional EA-source weight. Signed magnitude is never converted directly to probability.

### Fairness matrix

All arms use the same:

```text
environment interactions
learner update count
replay batch size
network and optimizer
EA population and operators
seed
training and evaluation budget
EA/RL slot realization from the baseline draw
logging schema
```

Where computationally feasible, the read-only source diagnostic is executed in all arms and ignored by A–C as appropriate. This equalizes measurement overhead and permits audit without changing training state. If common execution is too costly, overhead is reported separately and wall-clock time is not used as a performance outcome.

## 9. Score update lifecycle

At each configured diagnostic checkpoint:

1. obtain raw pre-mixing population trajectories with source provenance;
2. sample one `B_ref` before inserting current EA trajectories;
3. compute fitness, TD, and learner-utility source records through the read-only diagnostic;
4. rank-normalize valid sources within the checkpoint;
5. create a new score-table version;
6. insert current EA transitions and matching sidecar provenance;
7. make the new version available to later learner sampling, never to an earlier update;
8. leave baseline learner, replay payload, EC state, and training PRNG unchanged by measurement.

No same-operation feedback is allowed in which a source is sampled to estimate utility and then the same reference loss is immediately reported as algorithm success.

## 10. Stale utility

Utility is conditioned on the current learner and reference distribution. A score must not remain authoritative indefinitely.

V1 prefers a **finite freshness window** because it has simple, auditable semantics:

```text
age_i = current_generation - measured_generation_i
fresh_i = age_i <= W
```

Fresh scores use the selected rank weight. Expired scores revert to neutral baseline weight; their transitions remain in replay. `W` is selected before formal matched runs and is shared by fitness-, TD-, and utility-aware arms where a freshness rule is applicable, or the comparator's exception is explicitly justified.

An optional later version may use monotone decay to uniform:

```text
effective_alpha_i = alpha * decay(age_i)
```

but V1 should not combine a learned decay rule with the first causal pilot. The pilot logs source age, fraction of EA mass with fresh scores, expired-score sampling rate, and effective source count.

## 11. Anti-circularity and evaluation separation

`B_ref` is an estimator input, not an algorithm evaluation target. The following argument is invalid and prohibited:

```text
calculate U on B_ref
→ prioritize source by U
→ claim success because loss on the same B_ref decreases
```

### Primary outcomes

Algorithm success is evaluated using independent behavior-level targets:

- periodic deterministic evaluation return;
- evaluation-return area under the learning curve under the matched interaction budget;
- final evaluation return as a secondary endpoint.

### Independent learner-side outcome

If a loss-based endpoint is included, it uses `B_eval`, an independently sampled held-out batch that is never used to compute scores or sampling probabilities. Keys, indices/fingerprints, and update timing for `B_ref` and `B_eval` are logged and tested for separation. `B_eval` loss is secondary and cannot replace episodic evaluation return.

## 12. PRNG and state isolation

The baseline key schedule remains authoritative for environment rollout, baseline replay draw, and learner updates. Additional keys for source replacement, diagnostics, and held-out evaluation are derived from a diagnostic/intervention-local key namespace. They must not shift baseline rollout or learner keys.

The algorithm state adds only explicit pilot state:

```text
replay provenance sidecars
source score table
intervention-local PRNG state
intervention metrics
```

No diagnostic or score update writes to critic, actor, optimizer, targets, EC state, or replay payload. Only the selected replay indices affect training in B–D.

## 13. Exact-equivalence gates

Before any performance pilot, all gates below must pass.

### Gate 1: metadata alignment

After masked insertions and wraparound overwrites, payload and sidecar metadata share identical indices. EA source tokens and RL sentinels are correct.

### Gate 2: RL-origin invariance

For a fixed baseline draw, every RL slot and index is identical before and after the source-aware overlay.

### Gate 3: mixture invariance

For every batch, the number and positions of EA and RL slots exactly match the baseline draw.

### Gate 4: `alpha = 0` sampler equivalence

With `alpha = 0`, the overlay returns the original baseline indices without resampling. Sampled indices, batches, and baseline PRNG progression are bitwise identical.

### Gate 5: workflow equivalence

On a deterministic tiny run with `alpha = 0`, compare tree-wise after every step:

```text
actor and target actor
critic and target critic
optimizer state
replay payload, pointer, and size
EC state
baseline PRNG state
metrics and update counts
```

Only explicitly allowed pilot metadata and timing fields may differ.

### Gate 6: probability correctness

Weights are finite, normalized, bounded by the uniform component, deterministic for a fixed key, and neutral for missing/expired scores.

### Gate 7: anti-circularity

Tests prove that `B_ref` and `B_eval` have independent keys and fingerprints and that `B_eval` never enters source-score computation.

### Gate 8: budget equivalence

All four arms report identical environment episodes/timesteps, learner update count, replay batch size, evaluation schedule, and population operations for the same configured budget.

Failure of any exact-equivalence or anti-circularity gate blocks experiments.

## 14. Logging and audit outputs

Each run records:

- full resolved configuration and Git commit;
- environment episodes/timesteps and learner update count;
- baseline indices, final indices, and EA/RL slot count for audit samples;
- source token, origin, raw score, rank score, age, freshness, and final weight;
- source sampling counts and effective number of sampled EA sources;
- fraction of sampling mass assigned through the uniform floor;
- evaluation return and evaluation AUC inputs;
- `B_ref` and optional `B_eval` fingerprints;
- diagnostic, score-update, sampling, learner-update, evaluation, and total wall-clock timing;
- device, JAX/JAXLIB, and memory information.

Timing must distinguish measurement overhead from source-aware sampling overhead. Total step time alone must not be described as pure algorithm overhead.

## 15. Compute and memory overhead

Expected additional costs are:

- sidecar arrays proportional to replay capacity;
- a bounded source score table;
- one common diagnostic measurement at configured checkpoints;
- conditional weighted sampling for EA slots;
- optional independent held-out loss evaluation.

The pilot reports:

```text
diagnostic wall time
weight-update wall time
sampling wall time
learner-update wall time
peak device and host memory
replay metadata bytes
```

No method receives additional learner updates to compensate for overhead.

## 16. Pilot sequence

### Stage 0: unit and equivalence validation

Implement only enough to pass Gates 1–8. No performance claim is allowed.

### Stage 1: one-environment × one-seed smoke

Run a short Swimmer smoke for A–D to verify:

- all arms complete;
- budgets and update counts match;
- source weights are finite and non-degenerate;
- stale-score fallback occurs as designed;
- RL slots remain unchanged;
- evaluation data are independent of utility reference data.

The smoke is operational evidence only and cannot decide the research question.

### Stage 2: matched three-seed pilot

After smoke review, run A–D on Swimmer with the same seeds and a pre-registered budget. Runs are sequential on the fixed GPU environment. No parameter is changed after inspecting formal outcomes.

## 17. Pilot go/no-go criteria

The pilot receives **GO for broader replication** only if all conditions hold:

1. every equivalence, budget, and anti-circularity gate passes;
2. Utility-aware sampling remains non-degenerate and does not collapse to one source or uniform-only behavior;
3. median evaluation-return AUC for D minus A is positive across the three matched seeds;
4. D improves AUC over A in at least two of three matched seeds;
5. D does not show systematic final-return degradation relative to A;
6. D shows observable value beyond at least one heuristic comparator B or C;
7. the result is not explained by changed EA/RL exposure, update count, interaction budget, or evaluation leakage;
8. compute and memory overhead are measured and operationally acceptable.

The pilot receives **NO-GO / redesign** if any correctness gate fails, if D systematically degrades Original ERL, if benefits occur only on the score-estimation reference, if source weighting collapses, or if apparent gains disappear after enforcing matched budgets.

Three seeds are a pilot decision gate, not a definitive significance claim. Any broader claim requires a separately approved multi-environment replication plan.

## 18. Parameter governance

V1 intentionally leaves `alpha`, `tau`, and freshness window `W` unlocked in this design. Before formal runs:

1. define a small calibration space using only smoke behavior and sampler diagnostics;
2. reject settings that violate the uniform floor, numerical stability, or effective-source requirements;
3. select one configuration without using formal evaluation outcomes;
4. record it as a versioned configuration;
5. freeze it for all four matched arms where applicable.

Changing weighting, decay, or outcome definitions after viewing formal results creates a new pilot version and requires rerunning all arms. No environment-specific post-hoc tuning is permitted.

## 19. Planned implementation boundaries

If this design is approved, the implementation plan may introduce pilot-specific components such as:

```text
SourceAwareReplayBufferState
source-aware add/provenance adapter
baseline-draw EA overlay sampler
source score/rank updater
freshness policy
pilot workflow/configs
equivalence and anti-circularity tests
audit recorder and analysis script
```

Original ERL baseline source files should remain unchanged. The pilot should inherit or compose baseline behavior and confine intervention code to new modules/configurations wherever possible.

## 20. Review decisions required before implementation planning

The reviewer should approve or revise:

1. baseline-draw overlay as the mechanism for fixing EA/RL exposure;
2. occupancy-proportional `q_i` as the exact baseline conditional source mass;
3. finite freshness window as V1 staleness handling;
4. common diagnostic execution across arms;
5. independent `B_eval` when a learner-loss endpoint is used;
6. the direction of TD-error comparator C;
7. the go/no-go criteria;
8. the calibration and freeze procedure for `alpha`, `tau`, and `W`.

Implementation planning must not begin until this design has been reviewed and explicitly approved.
