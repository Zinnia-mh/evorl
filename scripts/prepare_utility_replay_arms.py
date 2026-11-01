"""Freeze four resolved Pilot configs before launching independent arm runs."""

import argparse

from omegaconf import OmegaConf

from evorl.algorithms.erl.erl_td3.utility_replay_budget import prepare_arm_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    for arm in ("uniform", "fitness", "td", "utility"):
        parser.add_argument(f"--{arm}", required=True, help="resolved YAML for this arm")
    args = parser.parse_args()
    configs = [OmegaConf.load(getattr(args, arm)) for arm in ("uniform", "fitness", "td", "utility")]
    prepare_arm_manifest(configs, args.output)


if __name__ == "__main__":
    main()
