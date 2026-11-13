"""Frozen six-run D0 ranking ablation driver."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import jax
from omegaconf import OmegaConf

from evorl.algorithms.erl.erl_td3.d0_placebo import Ablation
from evorl.algorithms.erl.erl_td3.utility_replay_checkpoint import environment_contract, mechanism_digest
from evorl.algorithms.erl.erl_td3.utility_replay_training import prepare_protocol, run_training

BASE = '228a3014902948251998bc8a39cad128c910ec6e'
SEEDS = (2081, 2099)
ARMS = {'A': 'A', 'D0': 'D', 'P': 'D'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)


def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def prepare(root, frozen):
    if (root / 'plan.json').exists():
        raise FileExistsError(root / 'plan.json')
    if json.loads((root / 'preflight/preflight.json').read_text())['status'] != 'PASS':
        raise ValueError('missing zero-step preflight')
    protocols = {}
    for seed in SEEDS:
        directory = root / 'protocols' / f'seed_{seed}'
        prepared = prepare_protocol(frozen, directory, seed=seed, eval_interval=10,
                                    eval_episodes=128, total_episodes=3950)
        assert prepared['final_generation'] == 120 and prepared['effective_total_episodes'] == 3950
        protocols[str(seed)] = {name: sha(directory / name) for name in prepared['config_checksums']}
        for label in ARMS.values():
            c = OmegaConf.load(directory / f'arm_{label}.yaml')
            assert c.env.max_episode_steps == 1000 and c.warmup_iters == 10
            assert c.eval_interval == 10 and c.eval_episodes == c.num_eval_envs == 128
            assert c.diagnostic_interval == 10 and c.replay_temperature == .5
            assert c.freshness_window == 30 and c.min_resident == 4
            assert c.alpha == (0.0 if label == 'A' else .5)
    write(root / 'plan.json', dict(
        base=BASE, commit=git('rev-parse', 'HEAD'), mechanism_digest=mechanism_digest(),
        environment=environment_contract(), seeds=SEEDS, arms=list(ARMS),
        run_order=[[seed, arm] for seed in SEEDS for arm in ARMS],
        frozen_source=str(frozen.resolve()), frozen_sha256=sha(frozen / 'checksums.sha256'),
        protocol_checksums=protocols, preflight_sha256=sha(root / 'preflight/preflight.json'),
        permutation='PCG64 SeedSequence([0x50304430, seed, measured_generation, resident_count]); uniform independent permutation of eligible sources within each exact age/count group sorted by token, singleton retained; effective ranks only',
        scope='Brax Swimmer, 10 warmup + 110 learning generations, identical native evaluation RNG and TD3 update schedule',
        checkpoint_policy=dict(interval=1, keep=3),
        learning_generations=110, actor_updates=1760000, critic_updates=3520000,
        evaluation_generations=list(range(20, 121, 10)), evaluation_episodes_per_point=128,
        training_steps_per_arm=3950000, evaluation_steps_per_arm=1408000,
        aggregate_limit=32148000, primary='paired within-seed D0-P trapezoidal AUC, CP20-120, 650000-3950000 training steps',
        secondary=['D0-A and P-A AUC', 'first sampled evaluation >= 350', 'CP120 return', 'same time curves', 'sampling TVD/ESS/eligible mass and diagnostic costs'],
        decision='GO only if D0 exceeds P in both seeds with comparable observed budgets, intervention strength and cost; otherwise HOLD',
    ))
    with (root / 'plan.sha256').open('x') as stream:
        stream.write(f"{sha(root / 'plan.json')}  plan.json\n")


def run(root, seed, arm, checkpoint):
    plan_path = root / 'plan.json'
    plan = json.loads(plan_path.read_text())
    plan_hash = sha(plan_path)
    if (root / 'plan.sha256').read_text().split()[0] != plan_hash:
        raise ValueError('plan seal mismatch')
    if plan['commit'] != git('rev-parse', 'HEAD') or plan['mechanism_digest'] != mechanism_digest():
        raise ValueError('wrong experiment code')
    if plan['environment'] != environment_contract() or jax.default_backend() != 'gpu':
        raise ValueError('wrong experiment environment')
    protocol = root / 'protocols' / f'seed_{seed}'
    for name, digest in plan['protocol_checksums'][str(seed)].items():
        if sha(protocol / name) != digest:
            raise ValueError(f'protocol checksum mismatch: {name}')
    config = OmegaConf.load(protocol / f'arm_{ARMS[arm]}.yaml')
    directory = root / f'seed_{seed}' / arm
    directory.parent.mkdir(exist_ok=True)
    log = directory.parent / f'{arm}-attempt-{time.time_ns()}.json'
    started = time.perf_counter()
    try:
        _, report = run_training(config, directory, checkpoint_interval=1, keep=3,
                                 resume_checkpoint=checkpoint,
                                 ablation=Ablation(arm, seed, plan_hash))
        assert report['status'] == 'COMPLETE' and report['final']['generation'] == 120
        assert [r['generation'] for r in report['evaluations']] == plan['evaluation_generations']
        assert report['final']['timesteps'] == plan['training_steps_per_arm']
    finally:
        write(log, dict(wall_seconds=time.perf_counter() - started))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('root', type=Path)
    p.add_argument('frozen', type=Path)
    r = sub.add_parser('run')
    r.add_argument('root', type=Path)
    r.add_argument('seed', type=int, choices=SEEDS)
    r.add_argument('arm', choices=tuple(ARMS))
    r.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    if git('status', '--porcelain') or git('branch', '--show-current') != 'experiment/d0-utility-placebo':
        raise ValueError('expected clean isolated experiment branch')
    root = args.root.resolve()
    if args.command == 'prepare':
        prepare(root, args.frozen)
    else:
        run(root, args.seed, args.arm, args.checkpoint)


if __name__ == '__main__':
    main()

