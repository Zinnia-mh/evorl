"""Explicit durable Pilot prepare/run/resume. Does not launch a matrix."""
import argparse,json,subprocess
from pathlib import Path
from omegaconf import OmegaConf
from evorl.algorithms.erl.erl_td3.utility_replay_training import run_training,prepare_protocol

ROOT=Path('/disk1/zmh_projects/evorl_artifacts/learner_utility_replay_pilot')

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('prepare');s.add_argument('--frozen',required=True);s.add_argument('--directory',required=True);s.add_argument('--seed',type=int,default=1);s.add_argument('--eval-interval',type=int,default=10);s.add_argument('--eval-episodes',type=int,default=128);s.add_argument('--total-episodes',type=int,default=100000)
    for command in ['run','resume']:
        s=sub.add_parser(command);s.add_argument('--directory',required=True);s.add_argument('--config',required=command=='run');s.add_argument('--checkpoint-interval',type=int,default=50);s.add_argument('--keep',type=int,default=3);s.add_argument('--stop-after-generation',type=int)
        if command=='resume':s.add_argument('--checkpoint',required=True)
    args=p.parse_args();directory=Path(args.directory).resolve()
    if not directory.is_relative_to(ROOT):raise ValueError('training artifacts must stay outside Git under Pilot artifact root')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('durable training requires a clean committed working tree')
    if args.command=='prepare':
        result=prepare_protocol(args.frozen,directory,seed=args.seed,eval_interval=args.eval_interval,eval_episodes=args.eval_episodes,total_episodes=args.total_episodes)
    else:
        config=OmegaConf.load(args.config or directory/'resolved_config.yaml')
        _,result=run_training(config,directory,resume_checkpoint=args.checkpoint if args.command=='resume' else None,stop_after_generation=args.stop_after_generation,checkpoint_interval=args.checkpoint_interval,keep=args.keep)
    print(json.dumps(result,sort_keys=True),flush=True)

if __name__=='__main__':main()
