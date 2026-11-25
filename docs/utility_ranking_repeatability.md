# Frozen D0 Utility ranking repeatability

Base:404a01b3d796967656d8b9309d3c460a4c24301d.
Spec: the user's CP50/CP300 frozen-state repeatability task.

## Protocol and implementation plan

Reuse ERLLearnerUtilityWorkflow._evaluate_source_utility unchanged. It calls the
real TD3 squared twin-Critic loss and virtual_critic_utility: one discarded Adam
update, Utility=L_ref_before-L_ref_after, same target noise for before/after.
Step scale is1.0 only. No model, replay, environment or sampler modifications.

Use all128 weighted-eligible resident sources at each checkpoint, four complete
32-source measured-generation cohorts. Selection is token-sorted, return/score
blind. This avoids estimating32-source cohort ranks from an8-source subset.
Use8 repeats each and a fixed diagnostic root seed20261125.

A(reference): fix source minibatch and all loss keys; change uniform reference
batch. B(target): fix source minibatch and reference batch; change loss keys
(both source-update and paired reference target noise, as real D0 splits them).
C(source): fix reference batch and all loss keys; change source minibatch.
References are common across all sources in a repeat, independently sampled
across A repeats. Source/loss keys are independent per source. Repeat0 is a
shared anchor across A/B/C. B/C are conditional reliability under a fixed
reference, not independent reference replications. Source batches sample256
resident transitions without replacement, matching the original construction's
sampling rule; reference sampling reuses ReplayBuffer.sample,256 with replacement.

The checkpoint is the time boundary: no future data. Historical source scores
were made by earlier learners; this experiment remeasures their surviving
transitions with the checkpoint's current learner. In the oldest cohort only
281-282 transitions survive; C samples most of this finite population.
Do not call this a reproduction of the original pre-exposure score.

Compute raw Utility mean/variance, pairwise-repeat sign agreement, Spearman,
Kendall tau-b, top-quartile overlap (intersection/top-k), pair reversal rate with
ties reported separately. Compare cohort and global rank reliability. Reapply
the production cohort rank normalization and production source probability
builder at fixed eligibility/residency, alpha=.5,temperature=.5. Compare
repeat-to-repeat source TV and a cohort-permuted ranking null. Null rankings
preserve each measurement's within-cohort rank multiset, using independently
seeded permutations. No permutation or repeat is an independent training seed.

Plan:
- [x] Write failing tests for factor isolation, metrics/ties, cohort mapping,
      production probability sensitivity and genuine TD3 virtual-update state safety.
- [x] Implement scripts/measure_utility_ranking_repeatability.py and pass only
      tests/test_utility_ranking_repeatability.py (do not rerun189 sampler cases).
- [x] Save manifest/code hashes/source identities before any measurement, restore
      native states read-only with full hash checks, run frozen GPU calculations
      in existing remote tmux, save raw data/keys/indices/losses.
- [x] Summarize both checkpoints without selecting favorable repeats; verify
      frozen hashes, costs and input integrity; review, commit and push.

Review focus: restored first-learner/Adam tree fidelity; extras ori_obs/termination
mapping; source token/cohort/rank slot mapping; one-factor key schedule; undefined
rank correlation for ties must remain undefined rather than be called reliable.
All artifacts remain in a new remote analysis directory outside frozen inputs.


---

# Frozen Learner Utility ranking repeatability

Diagnostic base:404a01b3d796967656d8b9309d3c460a4c24301d. Training seed2047; diagnostic seed20261125.
RTX4090, real TD3/Optax at step-scale1.0. No environment interactions or training updates.
Each CP:128 distinct sources,4 complete32-source cohorts,8 repeats per factor. Across both CPs:256 source identities,6144 measurement records. Repeat0 is a shared A/B/C anchor;5632 unique source/factor schedules. These are not independent training seeds.

## Actual D0 code path

Original online path:pre-exposure raw EA trajectory -> build_source_batches (valid256 transitions without replacement) -> uniform current Replay reference256 -> ERLLearnerUtilityWorkflow._evaluate_source_utility -> td3_critic_loss -> virtual_critic_utility.
TD3 uses clipped target policy noise(sigma.2,clip.5), frozen target Actor/twin Critics, discounted min target Q, mean sum of twin squared errors. One Adam(lr.0003) update uses the native saved count and moments and is discarded. Utility is reference loss before minus after with paired identical reference target noise. Cohort ranks are production average-tie ranks/(valid_count-1). Source probabilities reuse the production builder at alpha.5,temperature.5, with fixed residency/eligibility.
Restore is read-only, learner0 and all four online/target parameter trees retained. Full native SHA matches before/after. Replay extras ori_obs/termination mapping is verified against native metadata.

## Factor isolation

A:fixed source minibatch and loss keys;8 independent uniform reference draws, shared across all128 sources per repeat.
B:fixed source minibatch and reference;8 loss-key draws, varying both update-target noise and paired reference-target noise through the unchanged D0 key split.
C:fixed reference and loss keys;8 source minibatches without replacement.
B/C measure conditional reliability under a single fixed reference. Their repeated references must not be treated as independent reference validation. A supplies the independent-reference comparison.
No interaction terms are estimated; conditional variances must not be summed as an additive total noise decomposition.

## Cohort ranking reliability

Numbers average the28 repeat-pair statistics and four cohort-level summaries; the28 pairs share measurements and are not independent trials. Top25 overlap=intersection/top-k. Reversal excludes tied pairs; strict-pair fraction and boundary ties remain in JSON.

|CP|Factor|Spearman|Kendall tau-b|Top25 overlap|Pair reversal|
|---|---|---:|---:|---:|---:|
|50|A|0.891875|0.754284|74.107%|12.286%|
|50|B|0.964548|0.864775|87.835%|6.761%|
|50|C|0.949623|0.853507|86.942%|7.325%|
|300|A|0.860959|0.710938|71.763%|14.453%|
|300|B|0.921227|0.789783|81.473%|10.511%|
|300|C|0.938996|0.826289|87.612%|8.686%|

## Raw Utility

|CP|Factor|Mean Utility|Median source mean|Mean within-source variance|Positive fraction|Repeat-pair sign agreement|
|---|---|---:|---:|---:|---:|---:|
|50|A|-0.0044441916|-0.0029807233|5.0800844e-06|19.141%|83.650%|
|50|B|-0.0045039756|-0.0027081463|1.1248814e-06|10.059%|95.229%|
|50|C|-0.0045907529|-0.0027670737|1.2592305e-06|7.715%|96.233%|
|300|A|-0.0020167573|-0.0016017018|2.9949485e-07|0.586%|98.940%|
|300|B|-0.0020923156|-0.001759048|9.6522344e-08|0.195%|99.609%|
|300|C|-0.0020754622|-0.0017492405|7.3815453e-08|0.293%|99.470%|

Most measured updates have negative Utility, especially CP300. Stable ordering is not the same as positive Utility. Production rank normalization ignores absolute sign and magnitude, and can upweight the least-negative sources.

## Cohort normalization and probability variation

Probability TV is EA-conditional, across all resident sources, using normalized analytical production probabilities. No online Replay draws or Critic inputs were changed.

|CP|Factor|Global raw Spearman|Cohort-vs-global rank Spearman|Repeat probability TV|TV vs Uniform|Cohort-vs-global probability TV|Permutation repeat TV|
|---|---|---:|---:|---:|---:|---:|---:|
|50|A|0.900595|0.973545|0.51485%|1.55662%|0.26839%|1.75619%|
|50|B|0.972854|0.966429|0.30329%|1.55664%|0.28057%|1.73536%|
|50|C|0.959490|0.969970|0.29515%|1.55664%|0.26200%|1.72658%|
|300|A|0.868997|0.978998|0.59254%|1.55659%|0.23636%|1.76219%|
|300|B|0.924565|0.977661|0.42594%|1.55659%|0.24510%|1.73932%|
|300|C|0.946832|0.976928|0.34349%|1.55659%|0.25628%|1.75280%|

Independent within-cohort permutation null preserves each repeat's rank multiset. Its Spearman is approximately zero(-.0161 to.00873), top25 overlap23.55%-26.79%, and probability TV1.73%-1.76%. Genuine ranking probability variation is smaller(.295%- .593%), while total intervention vs Uniform remains~1.5566%. The repeat-to-repeat probability movement is about19%-38% of that intervention scale. This separates stable total intervention magnitude from uncertain source identity allocation; it does not predict Return.

## Noise interpretation and limits

Reference sampling is the largest measured conditional noise factor: its mean within-source variance is4.52x/4.03x target/source factors at CP50 and3.10x/4.06x at CP300. Cohort Spearman remains.861-.892 under independent references; this is repeatability above a random permutation baseline, not perfect ordering. Reversal remains12.3%-14.5% for this factor.
The oldest cohorts(gen20/270) retain281-282 transitions, so C draws256 of nearly the whole finite population. Their C Spearman(.9917/.9857) is higher than other cohorts(.9083-.9546/.9143-.9304). This limits generalization of source-minibatch noise to full newly collected1000-transition sources; the finite population is explicitly preserved rather than padded or replenished.
Current checkpoint Critic has already trained on some resident experience. These are current-state remeasurements, not reproductions of each cohort's historical pre-exposure score. CP50 contains cohorts20/30/40/50 and CP300 contains270/280/290/300. Reference batches contain only transitions present in their own checkpoint; no future or posterior Replay is imported.
One training seed and two learner snapshots do not establish cross-seed reliability, temporal stability, ranking specificity for Return, or any performance gain.

## Recommendation

GO for further offline temporal stability and coverage analysis with faithful sampling. Do not discard D0 on measurement-noise grounds alone: these fixed-state rankings retain substantial repeatability. Do not automatically increase scoring repetitions or sampling strength. Independent-reference variability is the first uncertainty to quantify, followed by whether ordering persists over source residence/freshness horizons. The original~10% score coverage and weak bounded intervention remain open issues.
Keep the original seed2047 -12.20% AUC result and historical permutation-performance result unchanged. No new A/D0, D1, RL-derived or environment training was run.

## Cost and validation

Whole frozen diagnostic process wall time:106.20s (~0.02950 GPU-allocation hours; includes restore/CPU orchestration, not a GPU-kernel-only time).
CP50:compile/warmup11.15s; A/B/C measurement loops19.73/10.44/10.59s (including batch construction/transfers/synchronization).
CP300:compile/warmup5.10s; A/B/C measurement loops10.82/17.10/10.44s (including batch construction/transfers/synchronization).
6 new targeted tests passed;189 earlier sampler tests were not rerun. Tests cover factor schedules, reversal metrics, ties, cohort mapping, real production probability law, and genuine TD3 virtual update/read-only state.
Independent read-only review found no Critical/Important issues. Optional future robustness: an entirely constant measurement can make cohort-vs-global Spearman undefined; strict JSON then aborts. Current measurements have no undefined correlations, nonfinite losses or Utility.
Post-run audit verifies all6144 batches have correct source identities, EA origin, valid256 unique indices, no future data, eight distinct C batches per source, eight distinct A references, fixed A/B source and B/C reference batches, and Utility=before-after. Raw keys/indices/losses are retained.

Files:run1/{manifest.json,summary.json,cp50/,cp300/};per_source.csv;postrun_audit.json;red.log;green.log;run1.log.
All files are remote under /disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/analysis/utility-ranking-repeatability-v1.
