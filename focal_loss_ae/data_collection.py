from env import make_atari
from utils import run_disk_data_collection
from typing import Optional
from pi import GymRandomPolicy
import argparse
import numpy as np


def run(env, obs_size, output_dir: str, max_steps: int = np.inf, max_episodes: int = np.inf):
    atari_env = make_atari(env, obs_size=obs_size)
    policy = GymRandomPolicy(atari_env.action_space)
    print("Data collection")
    run_disk_data_collection(policy, atari_env, output_dir, target_episodes=max_episodes,
                             target_steps=max_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect data on specified environments using random policy")

    parser.add_argument(
        '--env', type=str, help="the name of environment", default="PongNoFrameskip-v4")
    parser.add_argument('--output-dir', type=str,
                        help="the output directory for the disk saving")
    parser.add_argument('--max-steps', type=int,
                        help="the maximum steps to collect data", default=np.inf)
    parser.add_argument('--max-episodes', type=int,
                        help="the maximum episodes to collect data", default=np.inf)
    parser.add_argument('--obs-size', type=int, nargs="+",
                        help="the observation size", default=(64, 64))
    args = parser.parse_args()
    args.obs_size = tuple(args.obs_size)
    args = vars(args)
    print(args)
    run(**args)
