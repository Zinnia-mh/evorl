"""Explicit common-prefix, fork and same-arm resume entry points."""
import argparse
import json
from pathlib import Path
import subprocess

from omegaconf import OmegaConf

from evorl.algorithms.erl.erl_td3.d0_shared_start import SharedStartAblation, seal_parent
from evorl.algorithms.erl.erl_td3.utility_replay_training import run_training

# One shared 617,000-step prefix + 2 * (3,333,000 train + 1,408,000 eval).
ENVIRONMENT_STEPS_PER_SEED = 10_099_000
TWO_SEED_ENVIRONMENT_STEPS = 2 * ENVIRONMENT_STEPS_PER_SEED


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prefix','fork','resume'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--arm',choices=('D0','P'))
    parser.add_argument('--resume-checkpoint',type=Path)
    parser.add_argument('--stop-after-generation',type=int)
    args=parser.parse_args()
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        parser.error('a clean, versioned experiment worktree is required')
    root=args.root.resolve()
    repo=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip())
    if root.is_relative_to(repo):
        parser.error('experiment root must be outside the worktree')
    config=OmegaConf.load(args.config)
    expected=dict(total_episodes=3950, warmup_iters=10, eval_interval=10, eval_episodes=128,
                  num_eval_envs=128, diagnostic_interval=10, replay_temperature=.5,
                  freshness_window=30, min_resident=4, alpha=.5, arm='utility',
                  score_kind='utility', diagnostic_enabled=True, batch_size=256,
                  pop_size=32,num_rl_agents=1,episodes_for_fitness=1,rollout_episodes=1,
                  replay_buffer_capacity=1000000, actor_update_interval=2,
                  rl_updates_frac=1., rl_updates_mode='iter')
    if (any(config.get(k)!=v for k,v in expected.items())
            or config.env.env_type!='brax' or config.env.env_name!='swimmer' or config.env.max_episode_steps!=1000):
        parser.error('configuration differs from the shared-start protocol')
    parent=root/'permanent-cp19'
    if args.operation=='prefix':
        if args.arm or args.resume_checkpoint or args.stop_after_generation is not None:
            parser.error('prefix saves exactly one shared CP19')
        root.mkdir(parents=True,exist_ok=True)
        state,report=run_training(config,parent,stop_after_generation=19)
        cp=Path(report['last_checkpoint'])
        if (report['final']!=dict(generation=19,episodes=617,timesteps=617000,
                                  actor_optimizer_count=144000,critic_optimizer_count=288000)
                or report['evaluations']):
            raise ValueError('shared prefix budget mismatch')
        runtime=json.loads((cp/'runtime/metadata').read_text())['runtime']
        seal_parent(cp,config,state,runtime)
    else:
        if not args.arm or (args.operation=='resume') != bool(args.resume_checkpoint):
            parser.error('specify an arm and give a checkpoint only for resume')
        report_path,=parent.glob('segments/*/report.json')
        cp=Path(json.loads(report_path.read_text())['last_checkpoint']).resolve()
        if not cp.is_relative_to(parent):
            raise ValueError('parent checkpoint escapes permanent archive')
        experiment=SharedStartAblation(args.arm,int(config.seed),cp)
        if args.stop_after_generation is not None and not 20 <= args.stop_after_generation <= 120:
            parser.error('child stop must be between CP20 and CP120')
        _,report=run_training(config,root/args.arm,ablation=experiment,
                              fork=args.operation=='fork',resume_checkpoint=args.resume_checkpoint,
                              stop_after_generation=args.stop_after_generation,
                              checkpoint_interval=1,keep=102)
    print(json.dumps(report))


if __name__=='__main__':
    main()

