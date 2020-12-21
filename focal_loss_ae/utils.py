import os
import math
import numpy as np

MAX_NUM_EPISODE = 1e6
MAX_EPISODE_DIGITS = round(math.log(MAX_NUM_EPISODE, 10))

def save_obs_disk(episode_id: int, output_dir_root: str, max_episode_horizon:int = 1e6):
    horizon_length = round(math.log(max_episode_horizon, 10))
    output_path_template = f'{output_dir_root}/%0{MAX_EPISODE_DIGITS}d/%0{horizon_length}d.npy'
    os.makedirs(output_dir, exist_ok=True)

    def _save_obs_call_back(step: int, obs: np.ndarray, *args):
        output_path = output_path_template % (episode_id, step)
        np.save(output_path, obs)
    
    return _save_obs_call_back

def save_obs_replay_buffer(obs, action, reward, done, info):
    raise NotImplementedError


def run_data_collection(pi, env, start_episode_id: int = 0, target_episodes: int = None, target_steps: int = None, call_back = None):
    if target_episodes is not None and target_steps is not None:
        raise ValueError("Either set value for target episodes or target_steps")
    
    num_episode = 0
    num_steps = 0

    def _termination(num_episodes, num_steps):
        if target_episodes is not None:
            return num_episodes < target_episodes
        if target_steps is not None:
            return num_steps < target_steps
        return False
    
    while not _termination(num_episodes, num_steps):
        increased_steps, _ = run_episode(pi, env, call_back)
        steps += increased_steps
    return 



# TODO write a obs class to support DQN.
def run_episode(pi, env, call_back=None) -> int:

    last_obs = env.reset()
    step = 0
    done = False
    episode_return = 0

    while not done:
        action = pi.act(obs)
        obs, reward, done, info = env.step(action)

        if call_back is not None:
            call_back(step, last_obs, action, reward, done, info)

        step += 1
        episode_return += reward
        last_obs = obs
    
    if call_back is not None:
        call_back(step, last_obs, None, None, None, None)
    
    return step, episode_return

