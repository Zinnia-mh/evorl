# Learner-Utility-Aware Source Replay Pilot Design

## Document status

**Design v1.1 only. No replay algorithm is implemented by this document.**

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
3. **No source starvation or tiny-source amplification:** every resident EA transition retains a uniform component, and a source must satisfy `resident_count_i >= M_min` before its score may alter the weighted component.
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
resident_count
weighted_eligible
```

The table needs entries only for sources whose transitions may still be resident. Ring-buffer overwrite and the freshness rule bound the active set. V1 must not delete transitions when a score expires or a source falls below the resident-count threshold; either condition changes the source to neutral baseline behavior.

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
- `alpha = 0` must return the original baseline indices directly, without drawing a statistically equivalent replacement, enabling bitwise equivalence.

This overlay is preferable to independently choosing an EA/RL mixture because the latter would introduce a second intervention. It is also preferable to globally weighting all transitions because that would change RL-origin exposure.

## 7. Source weighting

### 7.1 Rank normalization

Raw signed utility is not a probability and must not be used directly. At a measurement checkpoint, valid EA sources are ranked within that checkpoint. Ties receive a deterministic average rank. Rank is normalized to a bounded score, for example:

```text
r_i in [0, 1]
```

Higher rank means higher priority for the method being tested. The same normalization procedure is used for fitness, TD error, and utility; only the underlying raw score differs.

### 7.2 Resident-source eligibility

Let `n_i = resident_count_i` be the number of currently resident EA transitions belonging to source `i`. A source may participate in score-based redistribution only when all conditions hold:

```text
n_i >= M_min
score is finite and valid
score age <= W
```

Thus:

```text
weighted_eligible_i = valid_i and fresh_i and (n_i >= M_min)
```

If `n_i < M_min`, the source remains in replay and its transitions remain sampleable through neutral baseline behavior, but its score cannot amplify its probability. This prevents a highly ranked source with only one or two surviving transitions from causing repeated sampling of those transitions. `M_min` is a pre-registered pilot parameter and cannot be adjusted using formal return, AUC, or final-return results.

The sampler records, per source and reporting window:

```text
resident_count
weighted_eligible
unique_transition_ratio
repeat_sampling_rate
```

For EA draws in a reporting window:

```text
unique_transition_ratio = number of distinct EA transition indices drawn
                          / number of EA transition draws
repeat_sampling_rate = 1 - unique_transition_ratio
```

The same definitions are used in B, C, and D.

### 7.3 Bounded mixture with a uniform floor

Let `N_EA` be the number of resident EA transitions and let:

```text
n_i = resident transition count for source i
q_i = n_i / N_EA
```

`q_i` is the exact baseline conditional probability mass of source `i`. Define the weighted-component source mass `s_i` so that ineligible sources are neutral rather than amplified:

```text
I = set of weighted-ineligible resident sources
E = set of weighted-eligible resident sources
Q_E = sum_{i in E} q_i

s_i = q_i                                           if i in I
s_i = Q_E * softmax_E(r_i / tau)                    if i in E
```

If `E` is empty, `s_i = q_i` for every source. Therefore `sum_i s_i = 1`, ineligible sources retain exactly their baseline source mass in the weighted component, and only the collective mass `Q_E` is redistributed among eligible sources.

The final source-level probability is:

```text
P(source=i | EA) = (1 - alpha) * q_i + alpha * s_i
```

For a resident transition `x_ij` belonging to source `i`, within-source sampling is uniform. Therefore:

```text
P_baseline(x_ij | EA)
  = q_i * (1 / n_i)
  = 1 / N_EA

P_weighted(x_ij | EA)
  = s_i * (1 / n_i)

P(x_ij | EA)
  = (1 - alpha) * q_i / n_i + alpha * s_i / n_i
  = (1 - alpha) / N_EA + alpha * s_i / n_i
```

For `n_i < M_min`, `i` is ineligible and `s_i = q_i`, so every one of its transitions remains exactly transition-uniform rather than being score-amplified. For equal-size, fully eligible sources, the source-level form reduces to:

```text
P(i) = (1-alpha)/N + alpha * softmax(rank_normalized(score_i)/tau)
```

The `(1-alpha) / N_EA` transition component is the starvation-prevention floor. V1 does not lock numerical values of `alpha`, `tau`, `W`, or `M_min` in this document; candidate sets and operational thresholds must be registered and frozen through the deterministic calibration protocol in Section 18 before formal matched runs.

### 7.4 Neutral and invalid scores

- missing, non-finite, insufficient-data, expired, or resident-count-ineligible scores are neutral;
- if every resident EA score is neutral, conditional EA sampling is baseline-uniform;
- no invalid score can become a very large positive or negative logit;
- probabilities are checked for finiteness, normalization, and a nonzero baseline component.
- when `alpha = 0`, the implementation bypasses all replacement sampling and directly returns the original baseline index; it must not draw a new statistically equivalent index.

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

## 10. Unified score freshness

Utility is conditioned on the current learner and reference distribution. TD error is also learner-conditioned, and even fitness ranks become an old source annotation as replay ages. To keep score type as the only causal difference among B, C, and D, V1 uses one identical freshness policy:

```text
W_fitness = W_TD = W_utility = W
```

The finite window has simple, auditable semantics:

```text
age_i = current_generation - measured_generation_i
fresh_i = age_i <= W
```

Fresh scores may use the selected rank weight. In B, C, and D, a score older than the same `W` becomes neutral baseline behavior. Its transitions remain in replay and are not deleted. No arm-specific stale handling, window, exception, or decay is allowed in V1.

An optional later version may use monotone decay to uniform:

```text
effective_alpha_i = alpha * decay(age_i)
```

but any utility-specific or score-specific decay is a future version, not V1. The pilot logs source age, fraction of EA mass with fresh scores, expired-score sampling rate, and effective source count using identical definitions in B, C, and D.

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

Let the fixed, matched environment-interaction checkpoints be:

```text
b_0 < b_1 < ... < b_K
```

and let `R[m,s,k]` be deterministic evaluation return for method `m`, matched seed `s`, and checkpoint `k`. The primary AUC is the trapezoidal integral:

```text
AUC[m,s] = sum_{k=0}^{K-1}
             0.5 * (R[m,s,k] + R[m,s,k+1]) * (b_{k+1} - b_k)
```

A, B, C, and D use exactly the same evaluation checkpoints, evaluation episode count, and interaction-budget x-axis. Evaluation frequency cannot be increased for one arm or changed to improve numerical AUC. Missing scheduled evaluations invalidate the matched AUC pair unless the pre-registered missing-run policy is applied uniformly.

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
- source token, origin, raw score, rank score, age, freshness, `resident_count`, `weighted_eligible`, and final weight;
- source sampling counts and effective number of sampled EA sources;
- unique-transition ratio and repeat-sampling rate;
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
3. `median_seed[AUC[D,s] - AUC[A,s]] > 0`;
4. at least two of three matched seeds satisfy `AUC[D,s] > AUC[A,s]`;
5. D does not show systematic final-return degradation relative to A: if all three matched seeds satisfy `FinalReturn[D,s] < FinalReturn[A,s]`, the result is an automatic NO-GO;
6. D exceeds at least one heuristic comparator under the same executable rule:

```text
(median_seed[AUC[D,s] - AUC[B,s]] > 0
 and at least 2/3 seeds satisfy AUC[D,s] > AUC[B,s])

OR

(median_seed[AUC[D,s] - AUC[C,s]] > 0
 and at least 2/3 seeds satisfy AUC[D,s] > AUC[C,s])
```

7. the result is not explained by changed EA/RL exposure, update count, interaction budget, or evaluation leakage;
8. compute and memory overhead are measured and operationally acceptable.

The pilot receives **NO-GO / redesign** if any correctness gate fails, if D systematically degrades Original ERL, if benefits occur only on the score-estimation reference, if source weighting collapses, or if apparent gains disappear after enforcing matched budgets.

Three seeds are a pilot decision gate, not a definitive significance claim. Any broader claim requires a separately approved multi-environment replication plan.

## 18. Deterministic calibration and parameter governance

V1 leaves numerical values unlocked here but fixes the selection process. Before any formal pilot, register finite ordered candidate sets:

```text
alpha in A
tau in T
W in W_set
M_min in M_set
```

The candidate sets, their order, operational thresholds, and tie-break order are versioned and timestamped before calibration. Calibration may use only sampler and systems behavior:

```text
effective source count / ESS
maximum source probability mass
uniform-floor contribution
fresh-score coverage
expired-score fraction
repeat-sampling rate
numerical stability
memory and compute overhead
```

It must not use:

```text
evaluation return
evaluation AUC
final return
formal pilot performance
```

The deterministic selection procedure is:

1. **Correctness filter:** remove any candidate that violates probability normalization, finiteness, exact mixture preservation, `alpha=0` equivalence, metadata alignment, or the pre-registered memory limit.
2. **Operational-band filter:** remove any candidate outside pre-registered acceptable bands for ESS/effective source count, maximum source mass, uniform-floor contribution, fresh-score coverage, expired-score fraction, repeat-sampling rate, or compute overhead.
3. **Sampler criterion:** among survivors, minimize the absolute distance between median ESS fraction and a pre-registered target ESS fraction. This target controls intervention strength without consulting return.
4. **Deterministic tie-break:** if candidates tie at the recorded precision, choose in order: lower maximum source probability mass, lower repeat-sampling rate, lower compute overhead, then the earliest tuple in the pre-registered lexicographic ordering of `A × T × W_set × M_set`.
5. **Freeze:** write the selected tuple, candidate-set hashes, thresholds, calibration data checksum, and selection trace to a versioned resolved config. Use that frozen tuple for the formal matched A/B/C/D pilot. A uses `alpha=0`; B/C/D use the same selected `alpha`, `tau`, `W`, and `M_min` so the scoring criterion remains the only causal difference.

If frozen diagnostic-v1 data help construct `W_set` or select `W`, they may be used only for pre-declared score stability and within-source rank-persistence calculations. No intervention return exists in diagnostic-v1, and no proxy based on future return may enter the rule. The rank-persistence statistic, threshold, and tie-break must be registered before applying them.

Changing weighting, freshness, `M_min`, calibration thresholds, or outcome definitions after viewing formal results creates a new pilot version and requires rerunning all arms. No environment-specific post-hoc tuning is permitted.

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
3. one shared finite freshness window `W_fitness = W_TD = W_utility = W` as V1 staleness handling;
4. common diagnostic execution across arms;
5. independent `B_eval` when a learner-loss endpoint is used;
6. the direction of TD-error comparator C;
7. the go/no-go criteria;
8. the calibration and freeze procedure for `alpha`, `tau`, `W`, and `M_min`.

Implementation planning must not begin until this design has been reviewed and explicitly approved.
