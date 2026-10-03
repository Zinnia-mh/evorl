# Learner-Utility Diagnostic: Final Report

## Status

**Learner-Utility Diagnostic: COMPLETE.**

This document freezes the diagnostic study at commit `76309c583002939b785d14801b2ba6181c9f9052` on `research/learner-utility-diagnostic`. It reports measurement evidence only. It does not introduce, evaluate, or claim benefit from a utility-aware replay algorithm.

## 1. Research question

Original ERL assigns evolutionary experience an implicit source value through the fitness of the policy that generated it. The diagnostic asks:

> Does evolutionary fitness reliably identify which EA source trajectories are most useful for the current RL learner?

More precisely, it tests whether the within-checkpoint ranking of population members by evolutionary fitness is interchangeable with their ranking by learner-side experience utility.

## 2. Motivation and literature boundary

ERL combines two distinct mechanisms: an EA population produces trajectories that enter shared replay, while an RL learner updates from replay and is periodically injected back into the population. Policy fitness measures behavior-level task performance. Replay value is learner-conditioned: it depends on the current critic, optimizer state, reference distribution, and training stage.

The study is motivated by this conceptual mismatch. It does **not** attempt to establish a universal definition of experience value, replace prioritized replay theory, or claim that policy quality and data quality are independent. It evaluates a concrete one-step learner-side diagnostic inside the existing ERL implementation.

## 3. Learner utility definition

For source `i`, let `B_i` be a fixed-size batch from that source's pre-mixing trajectory and let `B_ref` be one frozen reference batch shared by every eligible source at the checkpoint. Starting from the same real critic parameters and optimizer state, apply one optimizer-faithful virtual TD3 critic update on `B_i`, without writing it back. Learner utility is:

```text
U_i = L_ref_before - L_ref_after
```

Positive `U_i` means the virtual update reduced critic loss on `B_ref`; negative `U_i` means it increased that loss. This is a local, one-step, learner-conditioned proxy. It is not episodic return, long-horizon replay value, or a proof that sampling the source more often improves training.

## 4. Why measurement is pre-exposure

The diagnostic runs on raw population trajectories before those trajectories are mixed into replay for the current generation. Measuring after replay insertion or after learner updates would allow the candidate data to modify the learner or reference distribution before its utility was estimated. Pre-exposure ordering preserves the intended counterfactual: every source is evaluated from the same checkpoint state against a reference sampled before current-generation EA exposure.

## 5. Source provenance

Population rollout trajectories retain their individual identity. `source_id` is created from the generation and individual index, and `B_i` is extracted directly from that individual's trajectory. Sources are not reconstructed by flattening a mixed trajectory and guessing ownership afterward. Eligible sources use the same `source_batch_size`; insufficient valid transitions produce an explicit skip reason rather than silent padding or cross-source contamination.

## 6. Correctness gates

The measurement pipeline passed the following gates before formal replication:

1. **Optimizer-faithful virtual update:** real TD3 critic loss and the configured Optax update are used on cloned/local state.
2. **Diagnostic PRNG isolation:** diagnostic keys are derived separately and do not advance the baseline training key.
3. **Source provenance:** each `B_i` comes from the corresponding pre-mixing individual trajectory.
4. **Fixed-size source batches:** eligible sources use equal batch sizes; insufficiency is explicit.
5. **Shared reference batch:** all records within a checkpoint share one `reference_fingerprint`.
6. **Reference sampling timing:** `B_ref` is sampled once per checkpoint from current replay before current-generation EA exposure.
7. **Complete hook state invariance:** critic parameters, optimizer state, target critic, actor, target actor, replay, EC state, baseline PRNG, and workflow state are tree-wise unchanged by the diagnostic hook.
8. **Real lifecycle integration:** the hook is integrated at the real `_ec_rollout` pre-exposure point rather than tested only through a synthetic call path.
9. **Exact baseline equivalence gate:** with diagnostic effects excluded, baseline behavior and state transition remain equivalent.
10. **Record integrity:** every eligible record includes fitness, TD-error statistics, `loss_ref_before`, `loss_ref_after`, utility, source identity, reference fingerprint, and batch sizes.
11. **Focused and regression tests:** learner-utility tests and repository regression passed; the pre-existing optional MJX import/interface issue was excluded and was not counted as a learner-utility failure.
12. **GPU smoke:** the measurement path executed with JAX on `CudaDevice(id=0)` in `evorl-conflict`.

## 7. Exploratory smoke

The first 1-environment × 1-seed smoke produced 12 eligible records across three diagnostic generations. It exposed ranking inversions and high-fitness/negative-utility examples. Its pooled Spearman correlation was negative, but this result is **not** a formal conclusion: the sample is tiny, covers only three checkpoints, and pooling records across training stages mixes changing fitness and loss scales. The smoke established pipeline viability and motivated replication; it does not estimate the stable direction or magnitude of the relationship.

## 8. Three-seed replication configuration

The formal study used:

```text
environment: brax/swimmer
algorithm: Original ERL training configuration
seeds: 1, 2, 3
population: 32
total_episodes: 100000
generation budget: 3031
diagnostic interval: 10 generations
diagnostic checkpoints: 302 per seed
source batch size: 256
reference batch size: 256
reference setting: current_replay_pre_exposure
JAX/JAXLIB: 0.9.0 / 0.9.0
backend: GPU
```

The diagnostic workflow differs from Original ERL only by the read-only measurement fields and hook. Environment interaction budget, population, learner architecture, optimizer, learner update schedule, replay capacity, and EA/RL interaction settings remain at the baseline configuration.

## 9. Checkpoint-level methodology

The primary unit is a checkpoint, not a pooled source record. Each checkpoint contains 32 comparable sources evaluated against the same learner state and `B_ref`. The analysis computes within-checkpoint:

- Spearman rank correlation between fitness and utility;
- Kendall tau-b;
- normalized pairwise inversion rate;
- overlap between the top 20% by fitness and utility (`k = 7`);
- high-fitness/low-utility rate (fitness top quartile, utility bottom quartile);
- low-or-medium-fitness/high-utility recurrence;
- utility rank of the top-fitness source;
- exact top-fitness/top-utility agreement;
- negative-utility rate;
- Pearson and Spearman relationships between TD error and utility.

Checkpoint statistics are summarized within seed and then across the three seeds. Pooled raw-source plots are descriptive only.

## 10. Per-seed results

| Seed | Checkpoints | Mean Spearman(F,U) | Mean Kendall tau-b | Mean inversion rate | Mean top-20% overlap | Top-F = Top-U | Checkpoints with high-F/low-U |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 302 | 0.2480 | 0.1668 | 0.4166 | 0.4830 | 5.30% | 65.89% |
| 2 | 302 | 0.2372 | 0.1619 | 0.4190 | 0.4711 | 7.62% | 61.59% |
| 3 | 302 | 0.2476 | 0.1655 | 0.4173 | 0.5061 | 8.61% | 63.91% |

The effect is consistent across seeds: rank association is weakly positive, while disagreement remains large and exact winner agreement remains rare.

## 11. Aggregate results

```text
3 seeds
302 checkpoints / seed
32 sources / checkpoint
28992 source records

mean checkpoint Spearman(F,U) = 0.2443
mean checkpoint Kendall tau-b = 0.1647
mean normalized inversion rate = 0.4176
mean top-20% overlap = 0.4868

top-fitness == top-utility = 7.17%
top-fitness average utility rank = 6.85 / 32
checkpoints with high-F / low-U = 63.80%
checkpoints with low/medium-F / high-U = 97.02%
```

Fitness contains information about utility—the mean association is positive—but it leaves substantial source-order uncertainty. Roughly 42% of comparable source pairs have inverted fitness and utility ordering, and the fitness winner is the utility winner in only about 7% of checkpoints.

## 12. TD-error relationship

Across checkpoint-level summaries:

```text
mean Pearson corr(TD,U) = -0.4936
mean Spearman corr(TD,U) = -0.4211
mean negative-utility rate = 88.98%
```

Under this one-step reference-loss definition, higher TD error is associated with lower measured utility. This does not invalidate TD-error replay; it shows that TD magnitude and immediate held-reference loss improvement are different criteria in this setting.

## 13. Timing caveat

The recorded checkpoint timing is total instrumented training-step wall time, not isolated diagnostic overhead:

| Seed | Mean seconds | Median seconds | Total seconds over 302 checkpoints |
|---:|---:|---:|---:|
| 1 | 52.91 | 52.08 | 15977.75 |
| 2 | 53.32 | 52.50 | 16101.68 |
| 3 | 53.26 | 52.55 | 16085.21 |

Because no matched baseline timing decomposition was recorded here, these values must not be reported as pure diagnostic cost or slowdown.

## 14. Limitations

- One environment (Swimmer) and three seeds limit generalization.
- Utility is a one-step critic-loss proxy, not long-horizon learner benefit.
- `B_ref` represents current replay at one checkpoint and may not represent future training distributions.
- Utility is learner-conditioned and can become stale as parameters and replay change.
- Most measured utilities are negative; sign and magnitude depend on the learning rate, optimizer, batch construction, and reference distribution.
- The study measures association, not causal performance improvement from replay intervention.
- Raw fitness and utility scales change over training; checkpoint-level ranks are the appropriate formal comparison.

## 15. Final diagnostic conclusion

**Do NOT claim:** fitness and utility are unrelated.

**Supported claim:** fitness and learner utility are only weakly rank-aligned and are not interchangeable source-quality criteria.

Equivalent summary:

> Evolutionary fitness contains some information about learner utility, but is not a reliable substitute for learner-side experience value.

## 16. GO decision

The diagnostic receives a **GO** for a separately reviewed replay pilot. The GO is justified by cross-seed recurrence of ranking inversions, high-fitness/low-utility sources, low/medium-fitness/high-utility sources, and rare exact winner agreement. It is not a GO for deployment and is not evidence that utility-aware replay improves episodic return.

## 17. Next-stage research question

> Given that evolutionary fitness is not equivalent to learner-side experience utility, can learner-utility-aware source replay improve the RL learner while preserving the original ERL interaction budget?

The next stage must isolate source-selection effects, preserve Original ERL interaction and update budgets, address utility staleness, and evaluate performance on independent targets rather than the reference batch used to estimate utility.

## Reproducibility assets

Formal artifacts are stored outside Git under:

```text
/disk1/zmh_projects/evorl_artifacts/learner_utility_diagnostic/
```

See `ARTIFACT_MANIFEST.md` for paths, sizes, hashes, and descriptions. Git contains only compact summaries and one dashboard image; raw records, full logs, checkpoints, and the large workbook remain outside the repository.
