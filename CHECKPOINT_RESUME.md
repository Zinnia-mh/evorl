# Utility Replay full-state checkpoint and explicit resume

## Existing framework

Workflow.init always calls setup (including warmup). scripts/train.py has no
restore branch. Orbax helpers already support atomic PyTree checkpoints and
interval/retention policies, but checkpoint.enable defaults to false. Native
ERL learn strips Replay if save_replay_buffer=false. LogRecorder opens files
with mode=w, so simply re-entering the old command can destroy prior logs.

No Original ERL, sampler, Utility, PRNG or shared checkpoint implementation is
modified. The dedicated Pilot runner reuses native Orbax Composite commits:
one item is the ENTIRE State, the other the actual budget/evaluation cursor.
The legacy Workflow checkpoint manager stays disabled to avoid competing
writers; the runner-owned native manager is enabled by its CLI policy.

## Contract

- Includes Replay payload, ring pointer and provenance; EA population/key;
  Actor/Critic/Targets; optimizer states; Utility score table/distribution,
  score versions, measured generations, intervention key and baseline key.
- Restore builds an abstract template from native initialization components;
  never calls setup, warmup, random prefill, rollout or evaluate.
- Check configuration, implementation, package/backend/PRNG settings, file
  hashes and exact serialized State fingerprint. Training/evaluation settings
  and the original target budget cannot change on resume.
- A sealed shape/dtype/path inventory restores the three source-probability
  arrays which expand from length one to capacity after scoring. Other shape
  changes are rejected. Full-state SHA validation follows the native restore.
- Save AFTER the scheduled evaluation, including its advanced key and curve
  cursor. Pausing does not force an extra evaluation. Resume continues from the
  next absolute generation, not an additional full budget or another warmup.
- Keep >=2 checkpoints using Orbax atomic commits. Default: interval 50,
  keep 3. Completed generation paths cannot be overwritten. Explicit resume
  creates a new segment/checkpoint store; parent checkpoints/logs stay intact.
  The external checksum seal is written after the native commit. A crash in
  that short window can leave the newest directory unsealed; restore rejects
  it. Select an earlier sealed checkpoint instead. Retention preserves at
  least one prior valid checkpoint, rather than trusting directory presence.
- Append-only JSONL event segments. Canonical evaluation CSV/report is rebuilt
  from the checkpoint's committed prefix plus new points. Uncommitted old tails
  remain for provenance, but never become duplicated canonical evaluations.
  Work after the last committed checkpoint may be recomputed after a crash;
  its discarded physical work is separate from the logical training budget.
- A completed run cannot be resumed accidentally. A file lock rejects concurrent
  writers. Final reports include actual environment/update/population/evaluation
  counters, full curve, AUC over its observed step domain, and final return.
  Native evaluate currently returns means per RL actor, not raw per-episode
  evaluation returns; CSV columns state that aggregation explicitly.

## Proposed common formal protocol (configuration preparation only)

3031 total generations = 10 warmup + 3021 training generations. Existing ceil
rounding yields 100013 episodes for total_episodes=100000. Keep full formal
rl_updates_frac=1.0, one-million Replay, frozen alpha/tau/W/M_min and diagnostic
interval. Evaluate every 10 generations and at final generation 3031, 128
episodes each: 303 evaluations. Native evaluate advances state.key; keep this
unchanged and identical for A/B/C/D. Changing evaluation cadence creates a new
common protocol and is not equivalent to downsampling historical curves.

Run ONLY with later training authorization, inside evorl-utility:review:

```bash
cd /disk1/zmh_projects/evorl
PYTHON_BIN=/home/admin001/anaconda3/envs/evorl-conflict/bin/python
FROZEN=/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/calibration/phase11b-selected-v2-head-dbcb4f9-seed104729
STUDY=/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/formal_resume_v1
env -u LD_LIBRARY_PATH "$PYTHON_BIN" scripts/run_utility_replay_training.py prepare \
  --frozen "$FROZEN" --directory "$STUDY/protocol_seed1" --seed 1 \
  --eval-interval 10 --eval-episodes 128 --total-episodes 100000
env -u LD_LIBRARY_PATH "$PYTHON_BIN" scripts/run_utility_replay_training.py run \
  --config "$STUDY/protocol_seed1/arm_A.yaml" --directory "$STUDY/seed_1/A" \
  --checkpoint-interval 50 --keep 3
# Replace CHECKPOINT with the exact last_checkpoint path printed in the report,
# or a sealed segments/<id>/checkpoints/<generation> path after interruption.
# Its segment/checkpoint_checksums/<generation>.json must exist; resume verifies
# that seal against every saved file before loading. Native commit alone is
# insufficient if interrupted while writing the seal.
env -u LD_LIBRARY_PATH "$PYTHON_BIN" scripts/run_utility_replay_training.py resume \
  --directory "$STUDY/seed_1/A" --checkpoint "$CHECKPOINT" \
  --checkpoint-interval 50 --keep 3
```

prepare does not train. B/C/D use their respective resolved YAMLs and separate
run directories. Prepare separate manifests for seeds 2/3. Output includes
immutable config/manifest, append-only segment events, evaluation CSV/report,
native checkpoint state/runtime items and file seals. Do not use the old
12-generation Smoke entry point for full-budget training.

## Validation scope

Small real Swimmer test: ten warmup generations, then continuous six versus
three plus explicit resume three, both Uniform and Utility. Test-only horizon 4,
population 4, 8x8 networks and diagnostic interval 2 ensure the saved midpoint
contains active Utility scores and advanced intervention PRNG. This does not
alter any formal frozen configuration. Restore must be byte-exact; independent
post-resume Brax trajectories are compared numerically, with exact integer
metadata, keys, counters and evaluation schedule.

The one-off GPU million-capacity I/O probe uses native formal network/population
shapes with a fully occupied SYNTHETIC random transition payload. It performs
no environment interaction or optimizer update and is not a performance result.
It saves/loads the actual full State once, not merely model parameters. Report
actual bytes and API wall time including transfer/hash/commit, plus logical
state size and free disk. Budget four checkpoint sizes for the active segment
(three retained plus a pending save), in addition to older preserved segments.

Historical stable-generation measurements are approximately 22--24 seconds.
An interval of 50 bounds discarded logical work to roughly 18--20 minutes;
an interval of 10 is an optional higher-frequency policy after comparing the
measured I/O cost. This changes storage policy only, not training/evaluation.

## Verified results (2026-11-03)

- Native storage/configuration/protocol tests: 14 passed.
- Real small Swimmer Uniform six vs three+resume-three: passed (139.61 s).
- Real small Swimmer active Utility six vs three+resume-three: passed (270.98 s).
  Both run on CPU with JIT; resume forbids setup/warmup and verifies exact saved
  state, budgets, PRNG and evaluation checkpoints 12,14,16. Old event tails
  remain byte-identical and do not duplicate canonical curve records.
- One RTX4090 million-capacity native full-state I/O round trip: PASS,
  83,506,193 stored bytes versus 140,263,862 logical State bytes; write
  2.0888 seconds, load 0.6759 seconds. Transfer, hashes and synchronous native
  commit included; load used warm filesystem cache. No environment rollout.
- Validation report/checkpoint (outside Git):
  `/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/checkpoint_validation/20261103-native-resume-v1/`
  Evidence logs: `.codex_validation/utility_replay_pilot/checkpoint_native_protocol_v2.log`,
  `checkpoint_resume_uniform_final.log`, `checkpoint_resume_utility_v2.log`,
  `checkpoint_validate_final.log`, `checkpoint_million_native.log` and `.time`.

The stored size reflects this initialized-network/synthetic-Replay fixture;
formal trained checkpoints can compress differently. Reserve at least four
logical State sizes (about 0.56 GB) plus metadata/headroom and all archived
segments. Existing free space at the probe was 4.89 TB. No old Smoke,
calibration or training artifacts were overwritten.
