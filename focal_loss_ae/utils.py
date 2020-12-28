import os
import math
import numpy as np

MAX_NUM_EPISODE = 1e6
MAX_EPISODE_DIGITS = round(math.log(MAX_NUM_EPISODE, 10))


def save_obs_disk(episode_id: int, output_dir_root: str, max_episode_horizon: int = 1e6):
    horizon_length = round(math.log(max_episode_horizon, 10))
    output_dir = f'{output_dir_root}/%0{MAX_EPISODE_DIGITS}d' % episode_id
    os.makedirs(output_dir, exist_ok=True)

    def _save_obs_call_back(step: int, obs: np.ndarray, *args):
        output_path = f'{output_dir}/%0{horizon_length}d.npy' % step
        np.save(output_path, obs)

    return _save_obs_call_back


def save_obs_replay_buffer(obs, action, reward, done, info):
    raise NotImplementedError


def run_disk_data_collection(pi, env, output_dir_root: str, start_episode_id: int = 0, start_steps: int = 0,
                             target_episodes: int = np.inf, target_steps: int = np.inf):

    num_episodes = start_episode_id
    num_steps = start_steps

    def _termination(num_episodes, num_steps):
        return (num_episodes >= target_episodes) or (num_steps >= target_steps)

    while not _termination(num_episodes, num_steps):
        increased_steps, _ = run_episode(
            pi, env, save_obs_disk(num_episodes, output_dir_root))
        num_steps += increased_steps
        num_episodes += 1
        print(
            f"{num_episodes} out of {target_episodes}, {num_steps} out of {target_steps}", end="\r", flush=True)
    return


# TODO write a obs class to support DQN.
def run_episode(pi, env, call_back=None) -> int:

    last_obs = env.reset()
    step = 0
    done = False
    episode_return = 0

    while not done:
        action = pi.act(last_obs)
        obs, reward, done, info = env.step(action)

        if call_back is not None:
            call_back(step, last_obs, action, reward, done, info)

        step += 1
        episode_return += reward
        last_obs = obs

    if call_back is not None:
        call_back(step, last_obs, None, None, None, None)

    return step, episode_return
