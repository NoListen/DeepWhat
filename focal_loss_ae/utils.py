import os
import torch
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


def binary_decode(output):
    zero_indices = output < 0.5
    img = np.zeros_like(output, dtype=np.uint8)
    img[zero_indices] = 0
    one_indices = not zero_indices
    img[one_indices] = 1
    return img


def focal_loss(output, target):
    # TODO(lisheng) SOME ERRORS NEED TO FIX
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    # neg_weights = torch.pow(1-target, 4)

    loss = 0

    pos_loss = torch.log(output) * torch.pow(1-output, 2) * pos_inds
    neg_loss = torch.log(1-output) * torch.pow(output, 2) * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + negloss) / num_pos
    return loss
