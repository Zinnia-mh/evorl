# Formal A/B/C/D protocol and permanent ABC library

The registry tool wraps existing `prepare_protocol` and `run_training`. It does
not change step/evaluate, PRNG consumption, learner code, Utility or sampling.
Only this registered launcher automatically archives a run; direct invocation
of the old runner does not silently acquire a frozen_baseline label.

## Immutable protocol

`scripts/utility_replay_baseline_registry.py prepare --frozen F --directory R`
creates 12 per-run manifests plus native four-arm manifests for seeds 1,2,3.
All initially say READY_NOT_RUN. No workflow is initialized. Configuration is
100000 requested episodes, rounded by native budget to 100013: 3031 total
generations, ten warmup, 3021 training. Evaluate every ten generations plus
3031: 303 evaluations of 128 episodes. Native evaluation PRNG consumption is
unchanged and shared by all arms. Save full state every50, keep3. D0 inherits
the frozen alpha=.5, temperature=.5, W=30, minimum residency4 and interval10.

The immutable `checksums.sha256` pins config/protocol/run manifests,
environment and source provenance. Runtime files are added separately; the
initial manifest is never rewritten as COMPLETE. `registry/.../entry.json`
is the separate completion certificate. Do not copy original calibration
YAMLs directly to launch; their old eval_interval=1 is superseded only in the
derived formal protocol, never by editing calibration files.

## Registered launch and resume (requires later training authorization)

Run commands in tmux evorl-utility:review using evorl-conflict, from repo root:

```
env -u LD_LIBRARY_PATH /home/admin001/anaconda3/envs/evorl-conflict/bin/python scripts/utility_replay_baseline_registry.py run --directory R --arm A --seed 1
env -u LD_LIBRARY_PATH /home/admin001/anaconda3/envs/evorl-conflict/bin/python scripts/utility_replay_baseline_registry.py resume --directory R --arm A --seed 1 --checkpoint FULL_CHECKPOINT_PATH
```

Each call runs exactly one arm/seed, not the matrix. Native warmup, checkpoints,
lock, journal and resume validation are reused. Every attempt gets unique
start/result JSON and full stdout/stderr (including native messages). An abrupt
hard kill without a result is preserved and its wall time is UNKNOWN, not zero.

After native COMPLETE: check actual counters, expected evaluation grid,
configuration/commit/checkpoint policy, all finite returns, report SHA and
checkpoint file seals. Only then create an immutable `frozen_baseline` (ABC)
or `frozen_D_result`. Existing entries cannot be overwritten; partial archives
remain `.pending-*` without completion status. Disk/archive failure never
means rerun training: the `archive --directory R --arm A --seed 1` command can
retry archiving the existing completed native run. It also rejects overwriting
an already frozen entry.

Layout:

```
R/experiment.json, provenance.json, checksums.sha256
R/protocols/seed_{1,2,3}/arm_{A,B,C,D}.yaml, arm_manifest.json
R/manifests/{A,B,C,D}_seed_{1,2,3}.json
R/runs/{A,B,C,D0,D1,...}/seed_N/     # native append-only segments and checkpoints
R/attempts/VERSION/ARM/seed_N/ID/   # every attempt, including failures
R/registry/baselines/{A,B,C}/seed_N/
R/registry/D_versions/{D0,D1,...}/seed_N/
R/versions/D1/                      # future D preregistration, never overwrites D0
R/comparisons/D1/ID/
```

Each frozen result copies registered/native manifests, resolved config, all
segment logs/reports and attempt histories, canonical full evaluation CSV,
last three full checkpoints with native seals, entry metrics/counters/wall time
and SHA-256 over all archive files. Original native outputs remain intact.
Archive copies require extra storage; do not delete native data to make room.
SHA seals detect later corruption; they are not a replacement for backups or
OS access control. Registry writes use exclusive creation and a freeze lock.

## Future D variants and fair comparisons

Develop on a nonformal, noncalibration seed first. Freeze a separate D commit
and configs validated by the native arm manifest. Provide development evidence
JSON with `seed`, `head`, `status` (PASS/COMPLETE), `report_path`, and
`report_sha256`; it must identify that commit. No Return threshold is used.

```
python scripts/utility_replay_baseline_registry.py register-d --directory R --version D1 --configs NEW_PROTOCOL_ROOT --development-evidence DEV_EVIDENCE.json
python scripts/utility_replay_baseline_registry.py run --directory R --arm D --seed 1 --version D1
python scripts/utility_replay_baseline_registry.py compare --directory R --version D1
```

New protocol root contains `seed_1/arm_D.yaml` etc with valid native arm
manifest paths. Registration validates/pins these, requires shared conditions
to match original ABC, and allocates a new output directory. Neither ABC
training nor old D output is rerun/overwritten. All D successes and failures
remain; compare has no "choose best D" option and never tunes parameters.

Compatibility is strict on environment, seed matching, episode/step budget,
network, TD3/EA updates, Replay capacity/FIFO, baseline/exploration PRNG
settings, evaluation schedule/episode count and shared source hashes.
Administrative fields and explicit D sampler/scorer fields are stored but
excluded from the common comparison contract. All shared Python files under
evorl plus the native training CLI are pinned. Only the Utility scorer
`erl_learner_utility.py` and dedicated `utility_variants/` modules are excluded
from shared source identity; their hashes/commit are still recorded per run.
Changing any mixed/shared workflow file is conservatively rejected: it needs
a new common-protocol review or a new ABC library, not an automatic exemption
labelled "D only". Dependency or common protocol changes also reject reuse.
This tooling does not prove an arbitrary new D scorer is read-only; that
algorithm still requires its own development gates before registration.

Comparison requires all twelve successful frozen results and identical
evaluation step grids. It reads ABC curves directly and emits versioned
per-seed and across-seed metrics plus every curve. AUC uses trapezoids over
observed evaluation environment steps only (no invented return before first
evaluation). Late mean uses the last five scheduled evaluations, fixed before
results. Cross-seed mean and sample SD use the three actual independent seeds;
late-checkpoint SD is labelled separately, never used as independent repeats.

## Validation scope

Only new CPU synthetic registry tests are needed: actual budget/evaluation
rejection, compatibility and dependency rejection, immutable file/entry
behavior, checkpoint integrity, completed archive, and 12-result comparison.
No synthetic values are written to the real registry or used to select D.
Existing learning/sampler/checkpoint gates are reused. Actual automatic
archival will receive its first production-data validation when authorized
formal runs complete; nothing is called frozen_baseline in advance.
