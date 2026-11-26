# M-Late fixed seed3083 experiment

Base1f16fb2. Reuse unchanged online RL-seeded GA helper and prior GPU Smoke;
no old GA/TD3/Replay changes or full test rerun. New branch/output only.

Shared parent MUST be original Uniform A CP100 under the frozen baseline
runs/A tree, not old M. Verify sealed native full-state SHA, configuration,
learner/target/Adam/GA/Replay/PRNG/runtime and9 evaluation-prefix points.
A already has CP150; do not train A again.

First intervention110, then120/130/140/150 before EC rollout; population32,
tail8=4 pairs currentRLActor+/-0.005*epsilon, other24 positions unchanged.
Normal GA selection/breeding/mutation and original periodic single-Actor
injection continue. Uniform Replay only; real rollout-to-Replay receipts and
genealogy reuse the existing tested implementation.

Fixed primary: trapezoidal Return AUC100..150 on3.29M..4.94M steps, including
the shared CP100 evaluation endpoint. Every10g evaluate128episodes. Retain
all110..150 Returns, final Return, fresh8/remaining24 Fitness, breeding and
real timing. +/-2% is a prospective descriptive near band, not statistical
significance/equivalence. No interim tuning or early stopping.
If comparing old M, recompute its AUC on100..150; its old60..150 percent
cannot be directly compared with this different interval.

New physical budget:50gens x33k=1.65M training,5x128x1000=640k evaluation,
800k Actor/1.6M Critic updates. Final logical counters match A4.94M steps,
2.24M Actor/4.48M Critic updates. Time reports actual new segment wall/step
seconds and also charges measured common A prefix through100 to each arm.

Optional Utility: reuse existing diagnostic at150 only via explicit root
and generation arguments. Same real TD3/Optax update; frozen M-Late learner,
3 references256 transitions, actual same-generation A/M-Late sources.
Post-update phase; cross-run A exposure differs. Source means aggregate3
references; no future data, no replay writes and no additional environment.
The existing diagnostic defaults remain unchanged for older experiments.

Files: new scripts/run_rl_guided_late_pilot.py and
scripts/analyze_rl_guided_late_pilot.py; only parameterize existing diagnostic
entrypoint root/generation selection (two lines). New two focused tests pin
original-A100 provenance and primary endpoint. No new algorithm framework.

Artifacts exclusively remote:
    /disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot/algorithm/rl-guided-late-seed3083-v1

Decision: clear positive signal -> examine Actor quality and ancestry before
Utility coupling; near -> late start may avoid early M harm, no innovation
claim; negative -> redesign selection/proximal variation rather than keep
moving the start generation. One development seed is descriptive evidence.
